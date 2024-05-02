#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Evaluating langauge for conditional language modelling tasks.
"""
# You can also adapt this script for your own distillation tasks. Pointers for this are left as comments.

import logging
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Union

import datasets
import numpy as np
import torch
import torch.nn as nn
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    concatenate_datasets,
    interleave_datasets,
    load_dataset,
)
from huggingface_hub import create_repo, get_full_repo_name, upload_folder
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    BitsAndBytesConfig,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Seq2SeqTrainingArguments,
    get_scheduler,
    set_seed,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.34.0.dev0")

require_version("datasets>=2.14.6", "To fix: `pip install --upgrade datasets`")

logger = get_logger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to distill from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained Whisper model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    subfolder: str = field(
        default="",
        metadata={
            "help": "In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can"
            "specify the folder name here."
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Which attention implementation to use in the encoder and decoder attention layers. Can be one of:\n"
                "1. `eager` or `None`: default Transformers attention implementation.\n"
                "2. `sdpa`: Flash Attention through PyTorch SDPA. Requires `torch>=2.1`. Recommended for hardware where Flash Attention 2 is not supported, e.g. Turing GPUs, (T4, RTX 2080).\n"
                "3. `flash_attn_2`: Flash Attention 2 through the Flash Attention package https://github.com/Dao-AILab/flash-attention. **Always** recommended on supported hardware (Ampere, Ada, or Hopper GPUs, e.g., A100, RTX 3090, RTX 4090, H100)."
            )
        },
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "Use 8 bit precision for inference."})
    load_in_4bit: bool = field(default=False, metadata={"help": "Use 4 bit precision for inference."})
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "Quantization type if the teacher is quantized (fp4 or nf4)"}
    )
    use_bnb_nested_quant: bool = field(default=False, metadata={"help": "Whether or not to use nested quantization."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    eval_dataset_name: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "The name of the evaluation dataset to use (via the datasets library). Defaults to the training "
            "dataset name if unspecified. Load multiple evaluation datasets by separating dataset "
            "ids by a '+' symbol."
        },
    )
    eval_dataset_config_name: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "The configuration name of the evaluation dataset to use (via the datasets library). Defaults to the "
            "training dataset config name if unspecified."
        },
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to cache directory for saving and loading datasets"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing if using non-streaming mode."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."
            )
        },
    )
    eval_text_column_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset column containing the generated text data in the evaluation set."},
    )
    eval_prompt_column_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset column containing the prompt data in the evaluation set."},
    )
    max_label_length: int = field(
        default=4096,
        metadata={"help": "Truncate target labels that are longer `max_label_length` tokens."},
    )
    pad_target_to_multiple_of: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "If set will pad the target sequence to a multiple of the provided value. This is important to "
                "avoid triggering recompilations when using torch compile. If unspecified, will default to padding "
                "the targets to max length."
            )
        },
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is especially useful when data "
                "preprocessing errors out in distributed training due to timeout. In this case, one should run the "
                "preprocessing in a non-distributed setup with `preprocessing_only=True` so that the cached datasets "
                "can consequently be loaded in distributed training"
            )
        },
    )
    eval_split_name: Optional[List[str]] = field(
        default=lambda: ["validation"],
        metadata={
            "help": (
                "The name of the evaluation data set split to use (via the datasets library). Defaults to 'validation'"
            )
        },
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Whether to use Datasets' streaming mode to load and pre-process the data."},
    )
    wandb_project: str = field(
        default="distil-mixtral",
        metadata={"help": "The name of the wandb project."},
    )

@dataclass
class DistillationTrainingArguments(Seq2SeqTrainingArguments):
    output_router_logits: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to return the router logits in the forward pass. Enabling this will "
            "also configure the model to compute the auxiliary loss."
        },
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "The data type (dtype) in which to run training. One of `float32` (full-precision), "
                "`float16` or `bfloat16` (both half-precision)."
            )
        },
    )

@dataclass
class DataCollatorCausalLMWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        tokenizer ([`PreTrainedTokenizer`])
            The tokenizer used for tokenizing the data.
        target_padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned target sequences (according to the model's padding side and padding index).
            See above for details.
        max_target_length (:obj:`int`, `optional`):
            Maximum length of the ``labels`` of the returned list and optionally padding length (see above).
    """

    tokenizer: PreTrainedTokenizerBase
    target_padding: Union[bool, str] = "max_length"
    max_target_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> BatchEncoding:
        # dataloader returns a list of features which we convert to a dict
        label_features = {"input_ids": [feature["labels"] for feature in features]}
        prompt_lengths = [feature["prompt_length"] for feature in features]

        batch = self.tokenizer.pad(
            label_features,
            max_length=self.max_target_length,
            padding=self.target_padding,
            return_tensors="pt",
        )

        labels_mask = batch["attention_mask"]

        # don't include prompts in loss calculation
        for idx in range(len(prompt_lengths)):
            labels_mask[idx, : prompt_lengths[idx]] = 0

        # replace padding with -100 to ignore loss correctly
        labels = batch["input_ids"].masked_fill(labels_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


def log_metric(
    accelerator,
    metrics: Dict,
    prefix: str = "train",
):
    """Helper function to log all training/evaluation metrics with the correct prefixes and styling."""
    log_metrics = {}
    for k, v in metrics.items():
        log_metrics[f"{prefix}/{k}"] = v
    accelerator.log(log_metrics)


def log_pred(
    accelerator,
    pred_str: List[str],
    label_str: List[str],
    prefix: str = "eval",
    num_lines: int = 200000,
):
    """Helper function to log target/predicted transcriptions to weights and biases (wandb)."""
    if accelerator.is_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")
        table_name = f"predictions/{prefix.replace('/', '-')}"

        # convert str data to a wandb compatible format
        str_data = [[label_str[i], pred_str[i]] for i in range(len(pred_str))]
        # log as a table with the appropriate headers
        wandb_tracker.log_table(
            table_name=table_name,
            columns=["Target", "Pred"],
            data=str_data[:num_lines],
        )


def convert_dataset_str_to_list(
    dataset_names,
    dataset_config_names,
    splits=None,
    text_column_names=None,
    prompt_column_names=None,
    dataset_samples=None,
    default_split="train",
) -> List[Dict]:
    """
    Given three lists of dataset names, configs and splits, this function groups the corresponding
    names/configs/splits. Each dataset is assigned a unique dictionary with these metadata values, and the
    function returns a list of dictionaries, one for each dataset.
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
        splits = [splits] if splits else None
        text_column_names = [text_column_names] if text_column_names else None
        prompt_column_names = [prompt_column_names] if prompt_column_names else None
    if isinstance(dataset_config_names, str):
        dataset_config_names = [dataset_config_names]

    if len(dataset_names) == 1 and len(dataset_config_names) > 1:
        dataset_names = len(dataset_config_names) * dataset_names

    if isinstance(splits, list) and len(splits) == 1 and len(dataset_config_names) > 1:
        splits = len(dataset_config_names) * splits

    # basic checks to ensure we've got the right number of datasets/configs/splits/columns/probs
    if dataset_config_names is not None and len(dataset_names) != len(dataset_config_names):
        raise ValueError(
            f"Ensure one config is passed for each dataset, got {len(dataset_names)} datasets and"
            f" {len(dataset_config_names)} configs."
        )

    if splits is not None and len(splits) != len(dataset_names):
        raise ValueError(
            f"Ensure one split is passed for each dataset, got {len(dataset_names)} datasets and {len(splits)} splits."
        )

    if text_column_names is not None and len(text_column_names) != len(dataset_names):
        raise ValueError(
            f"Ensure one text column name is passed for each dataset, got {len(dataset_names)} datasets and"
            f" {len(text_column_names)} text column names."
        )

    if prompt_column_names is not None and len(prompt_column_names) != len(dataset_names):
        raise ValueError(
            f"Ensure one prompt column name is passed for each dataset, got {len(dataset_names)} datasets and"
            f" {len(prompt_column_names)} prompt column names."
        )

    if dataset_samples is not None:
        if len(dataset_samples) != len(dataset_names):
            raise ValueError(
                f"Ensure one sample is passed for each dataset, got {len(dataset_names)} datasets and "
                f"{len(dataset_samples)} samples."
            )
        dataset_samples = [float(ds_sample) for ds_sample in dataset_samples]
    else:
        dataset_samples = [None] * len(dataset_names)

    dataset_config_names = (
        dataset_config_names if dataset_config_names is not None else ["default" for _ in range(len(dataset_names))]
    )
    text_column_names = (
        text_column_names if text_column_names is not None else ["text" for _ in range(len(dataset_names))]
    )
    prompt_column_names = (
        prompt_column_names if prompt_column_names is not None else ["prompt" for _ in range(len(dataset_names))]
    )
    splits = splits if splits is not None else [default_split for _ in range(len(dataset_names))]

    dataset_names_dict = []
    for i, ds_name in enumerate(dataset_names):
        dataset_names_dict.append(
            {
                "name": ds_name,
                "config": dataset_config_names[i],
                "split": splits[i],
                "eval_text_column_name": text_column_names[i],
                "eval_prompt_column_name": prompt_column_names[i],
                "samples": dataset_samples[i],
            }
        )
    return dataset_names_dict


def get_quantization_config(
    model_args: ModelArguments, torch_dtype: torch.dtype
) -> tuple[BitsAndBytesConfig | None, BitsAndBytesConfig | None]:
    if model_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            # For consistency with model weights, we use the same value as `teacher_dtype`
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = None

    return quantization_config


def main():
    # 1. Parse input arguments
    # We keep distinct sets of args, for cleaner separation of model/data/training related args
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, DistillationTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # If we pass only one argument to the script and it's the path to a yaml file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. Initialize the accelerator
    # We will let the accelerator handle device placement for us in this example
    accelerator = Accelerator(
        log_with=training_args.report_to,
        project_dir=training_args.output_dir,
    )

    accelerator.init_trackers(project_name=data_args.wandb_project)

    # 3. Set-up basic logging
    # Create one log on every process with the configuration for debugging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Log a small summary on each proces
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    logger.info("Training/evaluation parameters %s", training_args)

    # 6. Load dataset - either streaming or non-streaming (offline)
    raw_datasets = IterableDatasetDict() if data_args.streaming else DatasetDict()

    # set seed for determinism
    set_seed(training_args.seed)
    dataset_names_dict = convert_dataset_str_to_list(
        data_args.eval_dataset_name,
        data_args.eval_dataset_config_name,
        splits=data_args.eval_split_name,
        text_column_names=data_args.eval_text_column_name,
        prompt_column_names=data_args.eval_prompt_column_name,
    )
    all_eval_splits = []
    if len(dataset_names_dict) == 1:
        # load a single eval set
        dataset_dict = dataset_names_dict[0]
        all_eval_splits.append("eval")
        raw_datasets["eval"] = load_dataset(
            dataset_dict["name"],
            dataset_dict["config"],
            split=dataset_dict["split"],
            cache_dir=data_args.dataset_cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
        )
        if dataset_dict["eval_text_column_name"] != "text":
            raw_datasets["eval"] = raw_datasets["eval"].rename_column(data_args.eval_text_column_name, "text")
        if dataset_dict["eval_prompt_column_name"] != "prompt":
            raw_datasets["eval"] = raw_datasets["eval"].rename_column(data_args.eval_prompt_column_name, "prompt")
    else:
        # load multiple eval sets
        for dataset_dict in dataset_names_dict:
            pretty_name = f"{dataset_dict['name'].split('/')[-1]}/{dataset_dict['config'].replace('.', '-')}"
            all_eval_splits.append(pretty_name)
            raw_datasets[pretty_name] = load_dataset(
                dataset_dict["name"],
                dataset_dict["config"],
                split=dataset_dict["split"],
                cache_dir=data_args.dataset_cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
            )
            # make column names consistent (text, audio)
            if dataset_dict["eval_text_column_name"] != "text":
                raw_datasets[pretty_name] = raw_datasets[pretty_name].rename_column(
                    dataset_dict["eval_text_column_name"], "text"
                )
            if dataset_dict["eval_prompt_column_name"] != "prompt":
                raw_datasets[pretty_name] = raw_datasets[pretty_name].rename_column(
                    dataset_dict["eval_prompt_column_name"], "prompt"
                )
            raw_datasets[pretty_name] = raw_datasets[pretty_name].remove_columns(
                set(raw_datasets[pretty_name].features.keys()) - {"text", "prompt"}
            )

    # 7. Load pretrained model, tokenizer, and feature extractor
    config = AutoConfig.from_pretrained(
        (model_args.config_name if model_args.config_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
    )
    if training_args.output_router_logits:
        config.output_router_logits = True

    tokenizer = AutoTokenizer.from_pretrained(
        (model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = getattr(torch, training_args.dtype, "float32")
    quantization_config = get_quantization_config(model_args, torch_dtype=torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        subfolder=model_args.subfolder,
        token=model_args.token,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
        quantization_config=quantization_config,
    )

    if model.generation_config.bos_token_id is None:
        raise ValueError(
            "Make sure that `generation_config.bos_token_id` is correctly defined for the "
            f"teacher model. Got {model.generation_config.bos_token_id}."
        )

    model.generation_config.max_length = data_args.max_label_length

    # 10. Preprocessing the datasets: we need to combine the prompt and generations and tokenize the targets.
    # 10.1: Define the pre-processing constants
    max_label_length = (
        data_args.max_label_length if data_args.max_label_length is not None else config.max_length
    )
    num_workers = data_args.preprocessing_num_workers
    dataloader_num_workers = training_args.dataloader_num_workers
    prefetch_factor = training_args.dataloader_prefetch_factor
    eos_token_id = tokenizer.eos_token_id

    # 10.2: filter based on maximum number of evaluation samples
    if data_args.max_eval_samples is not None:
        for eval_split in all_eval_splits:
            raw_datasets[eval_split] = (
                raw_datasets[eval_split].take(data_args.max_eval_samples)
                if data_args.streaming
                else raw_datasets[eval_split].select(range(data_args.max_eval_samples))
            )

    # 10.4: pre-process training/evaluation datasets
    def prepare_datasets(example):
        prompt_ids = tokenizer(example["prompt"]).input_ids
        gen_ids = tokenizer(example["text"], add_special_tokens=False).input_ids + [eos_token_id]
        if prompt_ids[-1] == eos_token_id:
            prompt_ids = prompt_ids[:-1]
        example["labels"] = prompt_ids + gen_ids
        example["prompt_length"] = len(prompt_ids)
        return example

    vectorized_datasets = IterableDatasetDict() if data_args.streaming else DatasetDict()
    for eval_split in all_eval_splits:
        raw_datasets_eval_features = list(raw_datasets[eval_split].features.keys())
        map_fn_eval = partial(
            raw_datasets[eval_split].map, function=prepare_datasets, remove_columns=raw_datasets_eval_features
        )
        with accelerator.main_process_first():
            vectorized_datasets[eval_split] = (
                map_fn_eval(num_proc=num_workers, desc="preprocess eval dataset")
                if not data_args.streaming
                else map_fn_eval()
            )

    # 10.6: Filter training data with labels longer than `max_label_length`
    def is_labels_in_length_range(labels):
        return 0 < len(labels) <= max_label_length

    filter_by_labels_fn = partial(
        vectorized_datasets.filter, function=is_labels_in_length_range, input_columns=["labels"]
    )
    with accelerator.main_process_first():
        vectorized_datasets = (
            filter_by_labels_fn(num_proc=num_workers, desc="filtering train dataset")
            if not data_args.streaming
            else filter_by_labels_fn()
        )

    # Pre-processing complete!
    # For large datasets it is advised to run the preprocessing on a
    # single machine first with `--preprocessing_only` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step, `--preprocessing_only` can then be set to `False` to load the
    # cached dataset
    if data_args.preprocessing_only:
        if data_args.streaming:
            raise ValueError(
                "When using streaming mode, dataset pre-processing is performed on the fly, hence there is no notion"
                "of a cached pre-processed dataset. Remove the argument `--preprocessing_only` to run pre-processing "
                "on the fly with streaming mode."
            )
        cache = {k: v.cache_files for k, v in vectorized_datasets.items()}
        logger.info(f"Data preprocessing finished. Files cached at {cache}.")
        return

    # 11. Define Evaluation Metrics
    def compute_metrics(preds, labels):
        # TODO(SG): better metrics for performance?
        # replace padded labels by the padding token
        for idx in range(len(labels)):
            labels[idx][labels[idx] == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return pred_str, label_str

    # 12. Define Training Schedule
    # Store some constants
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)

    data_collator = DataCollatorCausalLMWithPadding(
        tokenizer=tokenizer,
        target_padding="max_length",
        max_target_length=max_label_length,
    )

    # 14. Define generation arguments - we need to do this before we wrap the models in DDP
    # so that we can still access the configs
    num_beams = (
        training_args.generation_num_beams
        if training_args.generation_num_beams is not None
        else getattr(model.generation_config, "num_beams", 1)
    )

    # 15. Prepare everything with accelerate
    model = accelerator.prepare(model)

    # Define eval fn
    def eval_step(batch):
        model.eval()
        with torch.no_grad():
            loss = model(**batch).loss
        return {"ce_loss": loss}

    def generate_step(batch):
        model.eval()
        output_ids = accelerator.unwrap_model(model).generate(
            **batch, max_length=max_label_length, num_beams=num_beams
        )
        output_ids = accelerator.pad_across_processes(output_ids, dim=1, pad_index=tokenizer.pad_token_id)
        return output_ids

    model.eval()
    steps_trained_progress_bar = tqdm(
        range(len(all_eval_splits)), desc="Eval dataset ... ", position=0, disable=not accelerator.is_local_main_process
    )
    # ======================== Evaluating ==============================
    for idx, eval_split in enumerate(all_eval_splits):
        eval_metrics = []
        eval_preds = []
        eval_labels = []

        validation_dataloader = DataLoader(
            vectorized_datasets[eval_split],
            collate_fn=data_collator,
            batch_size=per_device_eval_batch_size,
            drop_last=False,
            num_workers=dataloader_num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=training_args.dataloader_pin_memory,
        )
        validation_dataloader = accelerator.prepare(validation_dataloader)

        for batch in tqdm(
            validation_dataloader,
            desc=f"Evaluating {eval_split}...",
            position=2,
            disable=not accelerator.is_local_main_process,
        ):
            # Model forward
            eval_metric = eval_step(batch)
            eval_metric = accelerator.gather_for_metrics(eval_metric)
            eval_metrics.append(eval_metric)

            # generation
            if training_args.predict_with_generate:
                generated_ids = generate_step(batch)
                # Gather all predictions and targets
                generated_ids, labels = accelerator.gather_for_metrics(
                    (generated_ids, batch["labels"])
                )
                eval_preds.extend(generated_ids)
                eval_labels.extend(labels)

        # normalize eval metrics
        stack = torch.stack if accelerator.num_processes == 1 else torch.concatenate
        # normalize eval metrics
        eval_metrics = {
            key: torch.mean(stack([d[key] for d in eval_metrics])) for key in eval_metrics[0]
        }
        try:
            eval_metrics["perplexity"] = math.exp(eval_metrics["ce_loss"])
        except OverflowError:
            eval_metrics["perplexity"] = float("inf")

        if training_args.predict_with_generate:
            pred_str, label_str = compute_metrics(eval_preds, eval_labels)
            log_pred(
                accelerator,
                pred_str,
                label_str,
                prefix=eval_split,
            )

        # Print metrics and update progress bar
        logger_desc = " ".join([f"Eval {key}: {value} |" for key, value in eval_metrics.items()])
        steps_trained_progress_bar.update(idx + 1)
        steps_trained_progress_bar.write(
            f"Eval results for split {eval_split} | {logger_desc}"
        )

        log_metric(
            accelerator,
            metrics=eval_metrics,
            prefix=eval_split,
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()

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
Training langauge models for conditional language modelling tasks via teacher-student distillation.
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
    set_seed, is_bitsandbytes_available,
)
from transformers.training_args import OptimizerNames
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
    teacher_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained teacher model or model identifier from huggingface.co/models"}
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
    load_teacher_in_8bit: bool = field(default=False, metadata={"help": "Use 8 bit precision for the teacher model."})
    load_teacher_in_4bit: bool = field(default=False, metadata={"help": "Use 4 bit precision for the teacher model."})
    load_student_in_8bit: bool = field(default=False, metadata={"help": "Use 8 bit precision for the student model."})
    load_student_in_4bit: bool = field(default=False, metadata={"help": "Use 4 bit precision for the student model."})
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "Quantization type if the teacher is quantized (fp4 or nf4)"}
    )
    use_bnb_nested_quant: bool = field(default=False, metadata={"help": "Whether or not to use nested quantization."})
    lora_r: Optional[int] = field(
        default=16,
        metadata={"help": "LoRA R value."},
    )
    lora_alpha: Optional[int] = field(
        default=32,
        metadata={"help": "LoRA alpha."},
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": "LoRA dropout."},
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "LoRA target modules."},
    )
    lora_modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Model layers to unfreeze & train"},
    )
    instruction_model: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether or not the pre-trained model is instruction tuned"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_dataset_name: List[str] = field(
        default=None,
        metadata={
            "help": "The name of the training dataset to use (via the datasets library). Load and combine "
            "multiple datasets by separating dataset ids by a '+' symbol. For example, to load LibriSpeech "
            "and Common Voice, set `train_dataset_name='librispeech_asr+common_voice'`."
        },
    )
    train_dataset_config_name: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "The configuration name of the training dataset to use (via the datasets library). Load and combine "
            "multiple datasets by separating dataset configs by a '+' symbol. Note that the order of the configs should "
            "match the order of the datasets."
        },
    )
    train_dataset_samples: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Number of samples in each dataset when loading multiple datasets with streaming mode. "
            "Not required when using one dataset or non-streaming mode. The sample values provide the sampling "
            "probability for each dataset. Setting them equal to the number of sample values ensures that every "
            "sample from every dataset is used once per epoch."
        },
    )
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
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."
            )
        },
    )
    text_column_name: Optional[List[str]] = field(
        default=None,
        metadata={"help": "The name of the dataset column containing the generated text data in the training set."},
    )
    prompt_column_name: Optional[List[str]]  = field(
        default=None,
        metadata={"help": "The name of the dataset column containing the prompt data. Defaults to 'prompt'"},
    )
    eval_text_column_name: Optional[List[str]]  = field(
        default=None,
        metadata={"help": "The name of the dataset column containing the generated text data in the evaluation set."},
    )
    eval_prompt_column_name: Optional[List[str]]  = field(
        default=None,
        metadata={"help": "The name of the dataset column containing the prompt data in the evaluation set."},
    )
    max_label_length: Optional[int] = field(
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
    train_split_name: Optional[List[str]] = field(
        default=lambda: ["train"],
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
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
        default="distil-mistral",
        metadata={"help": "The name of the wandb project."},
    )


@dataclass
class DistillationTrainingArguments(Seq2SeqTrainingArguments):
    freeze_embeddings: Optional[bool] = field(
        default=False, metadata={"help": "Whether to freeze the input and output embeddings of the student model."}
    )
    freeze_n_layers: Optional[int] = field(
        default=None, metadata={"help": "Freeze the first n layers of the student model."}
    )
    temperature: Optional[float] = field(
        default=2.0, metadata={"help": "Temperature to anneal the logits when computing the softmax."}
    )
    kl_weight: Optional[float] = field(
        default=1.0,
        metadata={
            "help": (
                "Weighting assigned to the MSE loss in the KD formulation. MSE loss is "
                "computed between the teacher-student hidden states and attentions."
            )
        },
    )
    output_router_logits: Optional[bool] = field(
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
    completions_only: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to train only on the target completions, or the prompt + completions."
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
        completions_only (:obj:`bool`, `optional`):
            Whether to train on the assistant responses (completions) only, or the combination of prompt + responses.
    """

    tokenizer: PreTrainedTokenizerBase
    target_padding: Union[bool, str] = "max_length"
    max_target_length: Optional[int] = None
    completions_only: Optional[bool] = False

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> BatchEncoding:
        # dataloader returns a list of features which we convert to a dict
        label_features = {"input_ids": [feature["labels"] for feature in features]}
        label_lengths = [len(feature["labels"]) for feature in features]
        prompt_lengths = [feature["prompt_length"] for feature in features]

        batch = self.tokenizer.pad(
            label_features,
            max_length=self.max_target_length,
            padding=self.target_padding,
            return_tensors="pt",
        )

        labels_mask = batch["attention_mask"]

        if self.completions_only:
            # don't include prompts in loss calculation
            for idx in range(len(prompt_lengths)):
                padding_length = labels_mask.shape[1] - label_lengths[idx]
                labels_mask[idx, padding_length : padding_length + prompt_lengths[idx]] = 0

        # replace padding with -100 to ignore loss correctly
        labels = batch["input_ids"].masked_fill(labels_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


def log_metric(
    accelerator,
    metrics: Dict,
    train_time: float,
    step: int,
    epoch: int,
    learning_rate: float = None,
    prefix: str = "train",
):
    """Helper function to log all training/evaluation metrics with the correct prefixes and styling."""
    log_metrics = {}
    for k, v in metrics.items():
        log_metrics[f"{prefix}/{k}"] = v
    log_metrics[f"{prefix}/time"] = train_time
    log_metrics[f"{prefix}/epoch"] = epoch
    if learning_rate is not None:
        log_metrics[f"{prefix}/learning_rate"] = learning_rate
    accelerator.log(log_metrics, step=step)


def log_pred(
    accelerator,
    pred_str: List[str],
    label_str: List[str],
    step: int,
    epoch: int,
    evaluation_strategy: str,
    prefix: str = "eval",
    num_lines: int = 200000,
):
    """Helper function to log target/predicted transcriptions to weights and biases (wandb)."""
    if accelerator.is_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")
        # pretty name for current step: step 50000 -> step 50k
        cur_step_pretty = f"{int(step // 1000)}k" if step > 1000 else step
        prefix_pretty = prefix.replace("/", "-")

        if evaluation_strategy == "epoch":
            table_name = f"predictions/{prefix_pretty}-epoch-{epoch}"
        else:
            table_name = f"predictions/{prefix_pretty}-step-{cur_step_pretty}"

        # convert str data to a wandb compatible format
        str_data = [[label_str[i], pred_str[i]] for i in range(len(pred_str))]
        # log as a table with the appropriate headers
        wandb_tracker.log_table(
            table_name=table_name,
            columns=["Target", "Pred"],
            data=str_data[:num_lines],
            step=step,
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

    if isinstance(text_column_names, list) and len(text_column_names) == 1 and len(dataset_config_names) > 1:
        text_column_names = len(dataset_config_names) * text_column_names

    if isinstance(prompt_column_names, list) and len(prompt_column_names) == 1 and len(dataset_config_names) > 1:
        prompt_column_names = len(dataset_config_names) * prompt_column_names

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
        prompt_column_names if prompt_column_names is not None else [None for _ in range(len(dataset_names))]
    )
    splits = splits if splits is not None else [default_split for _ in range(len(dataset_names))]

    dataset_names_dict = []
    for i, ds_name in enumerate(dataset_names):
        dataset_names_dict.append(
            {
                "name": ds_name,
                "config": dataset_config_names[i],
                "split": splits[i],
                "text_column_name": text_column_names[i],
                "prompt_column_name": prompt_column_names[i],
                "samples": dataset_samples[i],
            }
        )
    return dataset_names_dict


def load_multiple_datasets(
    dataset_names: Union[List, str],
    dataset_config_names: Union[List, str],
    splits: Optional[Union[List, str]] = None,
    text_column_names: Optional[List] = None,
    prompt_column_names: Optional[List] = None,
    stopping_strategy: Optional[str] = "first_exhausted",
    dataset_samples: Optional[Union[List, np.array]] = None,
    streaming: Optional[bool] = False,
    seed: Optional[int] = None,
    accelerator: Optional[Accelerator] = None,
    **kwargs,
) -> Union[Dataset, IterableDataset]:
    dataset_names_dict = convert_dataset_str_to_list(
        dataset_names, dataset_config_names, splits, text_column_names, prompt_column_names, dataset_samples
    )

    if dataset_samples is not None:
        dataset_samples = [ds_dict["samples"] for ds_dict in dataset_names_dict]
        probabilities = np.array(dataset_samples) / np.sum(dataset_samples)
    else:
        probabilities = None

    all_datasets = []
    # iterate over the datasets we want to interleave
    for dataset_dict in tqdm(
        dataset_names_dict,
        desc="Combining datasets...",
        disable=not accelerator.is_main_process,
    ):
        dataset = load_dataset(
            dataset_dict["name"],
            dataset_dict["config"],
            split=dataset_dict["split"],
            streaming=streaming,
            **kwargs,
        )

        columns_to_keep = {"text"}
        dataset_features = dataset.features.keys()

        if dataset_dict["text_column_name"] not in dataset_features:
            raise ValueError(
                f"Text column name {dataset_dict['text_column_name']} not found in dataset"
                f" '{dataset_dict['name']}'. Make sure to set `--text_column_name` to the"
                f" correct text column - one of {', '.join(dataset_features)}."
            )

        # blanket renaming of all transcription columns to text
        if dataset_dict["text_column_name"] != "text":
            dataset = dataset.rename_column(dataset_dict["text_column_name"], "text")

        # blanket renaming of all prompt columns to prompt
        if dataset_dict["prompt_column_name"] is not None:
            if dataset_dict["prompt_column_name"] not in dataset_features:
                raise ValueError(
                    f"Prompt column name {dataset_dict['prompt_column_name']} not found in dataset"
                    f" '{dataset_dict['name']}'. Make sure to set `--prompt_column_name` to the"
                    f" correct prompt column - one of {', '.join(dataset_features)}."
                )
            elif dataset_dict["prompt_column_name"] != "prompt":
                dataset = dataset.rename_column(dataset_dict["prompt_column_name"], "prompt")
            columns_to_keep.add("prompt")

        dataset = dataset.remove_columns(set(dataset_features - columns_to_keep))
        all_datasets.append(dataset)

    if len(all_datasets) == 1:
        # we have a single dataset so just return it as is
        return all_datasets[0]

    if streaming:
        interleaved_dataset = interleave_datasets(
            all_datasets,
            stopping_strategy=stopping_strategy,
            probabilities=probabilities,
            seed=seed,
        )
    else:
        interleaved_dataset = concatenate_datasets(all_datasets)

    # shuffle mixed dataset prior to potentially truncating it
    interleaved_dataset = interleaved_dataset.shuffle(seed)
    return interleaved_dataset


def sorted_checkpoints(output_dir=None, checkpoint_prefix="checkpoint") -> List[str]:
    """Helper function to sort saved checkpoints from oldest to newest."""
    ordering_and_checkpoint_path = []

    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*") if os.path.isdir(x)]

    for path in glob_checkpoints:
        regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
        if regex_match is not None and regex_match.groups() is not None:
            ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def rotate_checkpoints(save_total_limit=None, output_dir=None, checkpoint_prefix="checkpoint") -> Union[List, None]:
    """Helper function to delete old checkpoints."""
    if save_total_limit is None or save_total_limit <= 0:
        return
    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = sorted_checkpoints(output_dir=output_dir, checkpoint_prefix=checkpoint_prefix)
    if len(checkpoints_sorted) <= save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
        shutil.rmtree(checkpoint, ignore_errors=True)
    checkpoints_to_be_deleted = [f"*{Path(checkpoint).absolute().name}*"  for checkpoint in checkpoints_to_be_deleted]
    return checkpoints_to_be_deleted


_RE_CHECKPOINT = re.compile(r"^checkpoint-(\d+)-epoch-(\d+)$")


def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _RE_CHECKPOINT.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_RE_CHECKPOINT.search(x).groups()[0])))


def get_parameter_names(model, forbidden_layer_types, forbidden_module=None):
    """
    Returns the names of the model parameters that are not inside a forbidden layer or forbidden module.
    Can be used to get a subset of parameter names for decay masks, or to exclude parameters from an optimiser
    (e.g. if the module is frozen).
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types, forbidden_module)
            if not (
                isinstance(child, tuple(forbidden_layer_types))
                or (child in tuple(forbidden_module) if forbidden_module is not None else False)
            )
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def get_quantization_config(
    model_args: ModelArguments, torch_dtype: torch.dtype
) -> tuple[BitsAndBytesConfig | None, BitsAndBytesConfig | None]:
    if model_args.load_teacher_in_4bit:
        quantization_config_teacher = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
        )
    elif model_args.load_teacher_in_8bit:
        quantization_config_teacher = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config_teacher = None

    if model_args.load_student_in_4bit:
        quantization_config_student = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
        )
    elif model_args.load_student_in_8bit:
        quantization_config_student = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config_student = None

    return quantization_config_teacher, quantization_config_student


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
    # We simply have to specify the training precision and any trackers being used
    # We'll use the same dtype arguments as our JAX/Flax training script and convert
    # it to accelerate format
    if training_args.dtype == "float16":
        mixed_precision = "fp16"
        teacher_dtype = torch.float16
    elif training_args.dtype == "bfloat16":
        mixed_precision = "bf16"
        teacher_dtype = torch.bfloat16
    else:
        mixed_precision = "no"
        teacher_dtype = torch.float32

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=training_args.report_to,
        project_dir=training_args.output_dir,
    )

    accelerator.init_trackers(
        project_name=data_args.wandb_project,
        config={
            "learning_rate": training_args.learning_rate,
            "model_name_or_path": model_args.model_name_or_path,
            "teacher_name_or_path": model_args.teacher_model_name_or_path,
            "num_train_epochs": training_args.num_train_epochs,
            "max_steps": training_args.max_steps,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "global_batch_size": training_args.per_device_train_batch_size * accelerator.num_processes,
            "mixed_precision": mixed_precision,
            "lr_scheduler_type": training_args.lr_scheduler_type,
            "warmup_steps": training_args.warmup_steps,
            "weight_decay": training_args.weight_decay,
            "adam_beta1": training_args.adam_beta1,
            "adam_beta2": training_args.adam_beta2,
            "temperature": training_args.temperature,
        },
    )

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

    # 4. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(sorted_checkpoints(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 5. Handle the repository creation
    if accelerator.is_main_process:
        if training_args.output_dir is not None:
            os.makedirs(training_args.output_dir, exist_ok=True)
        if training_args.push_to_hub:
            if training_args.hub_model_id is None:
                repo_name = get_full_repo_name(
                    Path(training_args.output_dir).absolute().name,
                    token=training_args.hub_token,
                )
            else:
                repo_name = training_args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=training_args.hub_token)

            with open(os.path.join(training_args.output_dir, ".gitignore"), "w+") as gitignore:
                if "wandb" not in gitignore:
                    gitignore.write("wandb\n")
    accelerator.wait_for_everyone()

    # 6. Load dataset - either streaming or non-streaming (offline)
    raw_datasets = IterableDatasetDict() if data_args.streaming else DatasetDict()

    # set seed for determinism
    set_seed(training_args.seed)

    if training_args.do_train:
        raw_datasets["train"] = load_multiple_datasets(
            data_args.train_dataset_name,
            data_args.train_dataset_config_name,
            splits=data_args.train_split_name,
            text_column_names=data_args.text_column_name,
            prompt_column_names=data_args.prompt_column_name,
            streaming=data_args.streaming,
            dataset_samples=data_args.train_dataset_samples,
            seed=training_args.seed,
            accelerator=accelerator,
            cache_dir=data_args.dataset_cache_dir,
            token=model_args.token,
            num_proc=data_args.preprocessing_num_workers,
        )
        raw_datasets_train_features = set(raw_datasets["train"].features.keys())

    if training_args.do_eval:
        dataset_names_dict = convert_dataset_str_to_list(
            data_args.eval_dataset_name if data_args.eval_dataset_name else data_args.train_dataset_name,
            (
                data_args.eval_dataset_config_name
                if data_args.eval_dataset_config_name
                else data_args.train_dataset_config_name
            ),
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
            if dataset_dict["text_column_name"] != "text":
                raw_datasets["eval"] = raw_datasets["eval"].rename_column(data_args.eval_text_column_name, "text")
            if dataset_dict["prompt_column_name"] and dataset_dict["prompt_column_name"] != "prompt":
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
                # make column names consistent (text, prompt)
                columns_to_keep = {"text"}
                if dataset_dict["text_column_name"] != "text":
                    raw_datasets[pretty_name] = raw_datasets[pretty_name].rename_column(
                        dataset_dict["text_column_name"], "text"
                    )
                if dataset_dict["prompt_column_name"]:
                    if dataset_dict["prompt_column_name"] != "prompt":
                        raw_datasets[pretty_name] = raw_datasets[pretty_name].rename_column(
                            dataset_dict["prompt_column_name"], "prompt"
                    )
                    columns_to_keep.add("prompt")
                raw_datasets[pretty_name] = raw_datasets[pretty_name].remove_columns(
                    set(raw_datasets[pretty_name].features.keys()) - columns_to_keep
                )

    if not training_args.do_train and not training_args.do_eval:
        raise ValueError(
            "Cannot not train and not do evaluation. At least one of training or evaluation has to be performed."
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

    quantization_config_teacher, quantization_config_student = get_quantization_config(
        model_args, torch_dtype=teacher_dtype
    )

    if model_args.teacher_model_name_or_path:
        # The teacher model can safely be cast to the dtype of training since we don't
        # update the params
        teacher_model = AutoModelForCausalLM.from_pretrained(
            model_args.teacher_model_name_or_path,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            low_cpu_mem_usage=True,
            torch_dtype=teacher_dtype,
            attn_implementation=model_args.attn_implementation,
            quantization_config=quantization_config_teacher,
        ).eval()
    else:
        teacher_model = None

    student_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        subfolder=model_args.subfolder,
        token=model_args.token,
        low_cpu_mem_usage=True,
        attn_implementation=model_args.attn_implementation,
        quantization_config=quantization_config_student,
    )

    if quantization_config_student is not None:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.lora_target_modules,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        student_model = get_peft_model(student_model, lora_config)

    if student_model.generation_config.bos_token_id is None or (teacher_model and teacher_model.generation_config.bos_token_id is None):
        student_error = f"Make sure that `generation_config.bos_token_id` is correctly defined. Got {student_model.generation_config.bos_token_id} for the student."
        teacher_error = f"Got {teacher_model.generation_config.bos_token_id} for the teacher." if teacher_model else None
        raise ValueError(student_error + teacher_error)

    def set_trainable_parameters(module, requires_grad=False):
        for param in module.parameters():
            param.requires_grad = requires_grad
        module._requires_grad = requires_grad

    forbidden_module = []
    # freeze student embeddings if necessary
    if training_args.freeze_embeddings:
        set_trainable_parameters(student_model.get_output_embeddings(), requires_grad=False)
        set_trainable_parameters(student_model.get_input_embeddings(), requires_grad=False)
        forbidden_module.extend([student_model.get_output_embeddings(), student_model.get_input_embeddings()])

    if training_args.freeze_n_layers:
        for i in range(int(training_args.freeze_n_layers)):
            set_trainable_parameters(student_model.model.layers[i], requires_grad=False)
            forbidden_module.extend([student_model.model.layers[i]])

    # enable gradient checkpointing if necessary
    if training_args.gradient_checkpointing:
        if training_args.freeze_embeddings or training_args.freeze_n_layers:
            raise ValueError(
                "Gradient checkpointing is not compatible with `--freeze_embeddings` or `--freeze_n_layers`. "
                "Either un-freeze these layers, or set `--gradient_checkpointing=False`."
            )
        student_model.gradient_checkpointing_enable()

    student_model.generation_config.max_length = data_args.max_label_length

    # 8. Save all pre-processed tokenizers/config/generation configs
    if accelerator.is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)
        # save the config and generation config as well
        config.save_pretrained(training_args.output_dir)
        student_model.generation_config.save_pretrained(training_args.output_dir)

    accelerator.wait_for_everyone()


    # 10. Preprocessing the datasets: we need to combine the prompt and generations and tokenize the targets.
    # 10.1: Define the pre-processing constants
    max_label_length = (
        data_args.max_label_length if data_args.max_label_length is not None else config.max_length
    )
    num_workers = data_args.preprocessing_num_workers
    dataloader_num_workers = training_args.dataloader_num_workers
    prefetch_factor = training_args.dataloader_prefetch_factor
    eos_token_id = tokenizer.eos_token_id
    if model_args.instruction_model is not None:
        instruction_model = model_args.instruction_model
    else:
        instruction_model = "instruct" in model_args.model_name_or_path.lower()
    if instruction_model and "prompt" not in raw_datasets_train_features:
        raise ValueError(
            "Distilling an instruction model, but `--prompt_column_name` is set to None. "
            "Ensure `--prompt_column_name` is set according to the dataset features."
        )

    # 10.2: filter based on maximum number of training/evaluation samples
    if training_args.do_train and data_args.max_train_samples is not None:
        raw_datasets["train"] = (
            raw_datasets["train"].take(data_args.max_train_samples)
            if data_args.streaming
            else raw_datasets["train"].select(range(data_args.max_train_samples))
        )

    if training_args.do_eval and data_args.max_eval_samples is not None:
        for eval_split in all_eval_splits:
            raw_datasets[eval_split] = (
                raw_datasets[eval_split].take(data_args.max_eval_samples)
                if data_args.streaming
                else raw_datasets[eval_split].select(range(data_args.max_eval_samples))
            )

    # 10.3: pre-process training/evaluation datasets
    def prepare_dataset(example):
        prompt = example.get("prompt")
        target_text = prompt + example["text"] if prompt is not None else example["text"]
        example["labels"] = tokenizer(target_text).input_ids
        if example["labels"][-1] != eos_token_id:
            example["labels"] += [eos_token_id]
        example["prompt_length"] = len(tokenizer(prompt).input_ids) if prompt else 0
        return example

    def prepare_instruction_dataset(example):
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["text"]},
        ]
        example["labels"] = tokenizer.apply_chat_template(messages)
        if example["labels"][-1] != eos_token_id:
            example["labels"] = example["labels"][:-1]

        example["prompt_length"] = len(tokenizer.apply_chat_template([messages[0]]))
        return example

    prepare_dataset = prepare_instruction_dataset if instruction_model else prepare_dataset
    vectorized_datasets = IterableDatasetDict() if data_args.streaming else DatasetDict()
    if training_args.do_train:
        # with streaming mode we can only have 1 worker, whereas with non-streaming
        # we can use `num_workers` (which is much faster)
        # We gate the pre-processing function accordingly
        map_fn_train = partial(
            raw_datasets["train"].map,
            function=prepare_dataset,
            remove_columns=raw_datasets_train_features,
        )
        with accelerator.main_process_first():
            vectorized_datasets["train"] = (
                map_fn_train(num_proc=num_workers, desc="preprocess train dataset")
                if not data_args.streaming
                else map_fn_train()
            )
    if training_args.do_eval:
        for eval_split in all_eval_splits:
            raw_datasets_eval_features = list(raw_datasets[eval_split].features.keys())
            map_fn_eval = partial(
                raw_datasets[eval_split].map, function=prepare_dataset, remove_columns=raw_datasets_eval_features
            )
            with accelerator.main_process_first():
                vectorized_datasets[eval_split] = (
                    map_fn_eval(num_proc=num_workers, desc="preprocess eval dataset")
                    if not data_args.streaming
                    else map_fn_eval()
                )

    # 10.4: Filter training data with labels longer than `max_label_length`
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
    # 12.1: Store some constants
    per_device_train_batch_size = int(training_args.per_device_train_batch_size)
    train_batch_size = per_device_train_batch_size * accelerator.num_processes
    gradient_accumulation_steps = int(training_args.gradient_accumulation_steps)
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)
    temperature = training_args.temperature

    # 12.2: Set max training steps
    if not data_args.streaming and training_args.max_steps < 0:
        num_epochs = int(training_args.num_train_epochs)
        steps_per_epoch = len(vectorized_datasets["train"]) // (train_batch_size * gradient_accumulation_steps)
        total_train_steps = steps_per_epoch * num_epochs
    elif training_args.max_steps > 0:
        logger.info("max_steps is given, it will override any value given in num_train_epochs")
        total_train_steps = int(training_args.max_steps)
        if not data_args.streaming:
            steps_per_epoch = len(vectorized_datasets["train"]) // (train_batch_size * gradient_accumulation_steps)
            num_epochs = int(np.ceil(total_train_steps / steps_per_epoch))
        else:
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_epochs = sys.maxsize
            steps_per_epoch = total_train_steps
    else:
        raise ValueError("max_steps must be specified when training with a streaming (iterable) dataset")

    # 12.3: Set evaluation steps
    if training_args.evaluation_strategy == "epoch":
        eval_steps = steps_per_epoch
    elif training_args.eval_steps is None:
        logger.info(
            f"eval_steps is not set, evaluating at the end of {'each epoch' if not data_args.streaming else 'training'}"
        )
        eval_steps = steps_per_epoch
    else:
        eval_steps = training_args.eval_steps

    # 12.4: Set save steps
    if training_args.save_strategy == "epoch":
        save_steps = steps_per_epoch
    elif training_args.save_strategy == "steps":
        save_steps = training_args.save_steps
    else:
        save_steps = sys.maxsize

    # 13. Define optimizer, LR scheduler, collator
    decay_parameters = get_parameter_names(
        student_model,
        [nn.LayerNorm],
        forbidden_module,
    )

    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [param for name, param in student_model.named_parameters() if name in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [param for name, param in student_model.named_parameters() if name not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    if training_args.optim == OptimizerNames.ADAMW_TORCH:
        optim_cls = torch.optim.AdamW
    elif training_args.optim == OptimizerNames.ADAMW_BNB:
        if not is_bitsandbytes_available():
            raise ValueError(
                "bitsandbytes package required for Adam8bit. Install via: `pip install --upgrade bitsandbytes`"
            )
        import bitsandbytes as bnb

        optim_cls = bnb.optim.Adam8bit
    else:
        raise ValueError(
            f"Got invalid `--optim` {training_args.optim}, should be one of `['adam_torch', 'adamw_bnb_8bit']`."
        )

    optimizer = optim_cls(
        params = optimizer_grouped_parameters,
        lr = training_args.learning_rate,
        betas = (training_args.adam_beta1, training_args.adam_beta2),
        eps = training_args.adam_epsilon,
    )

    # LR scheduler gets stepped by `num_processes` each time -> account for this in warmup / total steps
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps * accelerator.num_processes,
        num_training_steps=total_train_steps * accelerator.num_processes,
    )

    data_collator = DataCollatorCausalLMWithPadding(
        tokenizer=tokenizer,
        target_padding="max_length",
        max_target_length=max_label_length,
        completions_only=training_args.completions_only,
    )

    # 14. Define generation arguments - we need to do this before we wrap the models in DDP
    # so that we can still access the configs
    num_beams = (
        training_args.generation_num_beams
        if training_args.generation_num_beams is not None
        else getattr(student_model.generation_config, "num_beams", 1)
    )

    # 15. Prepare everything with accelerate
    student_model, optimizer, lr_scheduler = accelerator.prepare(student_model, optimizer, lr_scheduler)
    teacher_model = accelerator.prepare(teacher_model) if teacher_model else None

    def kl_divergence(target_distribution, log_predicted_distribution, labels):
        kl_loss = nn.KLDivLoss(reduction="none")
        divergence = kl_loss(log_predicted_distribution, target_distribution)
        # ignore padded tokens from divergence, i.e. where labels are not set to -100
        padding_mask = labels >= 0
        padding_mask = padding_mask.unsqueeze(-1)
        divergence = divergence * padding_mask
        # take the average over the mini-batch
        divergence = divergence.sum() / padding_mask.sum()
        return divergence

    # Define gradient update step fn
    def train_step(batch):
        student_model.train()
        student_outputs = student_model(**batch)

        # CE (data) loss
        ce_loss = student_outputs.loss
        metrics = {"ce_loss": ce_loss}

        if teacher_model:
            with torch.no_grad():
                teacher_outputs = teacher_model(**batch)
            # rescale distribution by temperature to ensure gradients scale correctly
            teacher_distribution = nn.functional.softmax(teacher_outputs.logits / temperature, dim=-1)
            # log softmax of student predictions for numerical stability
            student_distribution = nn.functional.log_softmax(student_outputs.logits / temperature, dim=-1)
            # KL-divergence loss (scaled by temperature)
            kl_loss = kl_divergence(teacher_distribution, student_distribution, batch["labels"]) * temperature ** 2
            # use Distil-Whisper formulation (fix weight of CE loss and tune KL weight)
            loss = 0.8 * ce_loss + training_args.kl_weight * kl_loss
            metrics["kl_loss"] = kl_loss
        else:
            loss = ce_loss

        metrics["loss"] = loss
        return loss, metrics

    # Define eval fn
    @torch.no_grad()
    def eval_step(batch):
        student_model.eval()

        # CE (data) loss
        student_outputs = student_model(**batch)
        ce_loss = student_outputs.loss
        metrics = {"ce_loss": ce_loss}

        if teacher_model:
            teacher_outputs = teacher_model(**batch)
            # log softmax / softmax for numerical stability
            student_distribution = nn.functional.log_softmax(student_outputs.logits, dim=-1)
            teacher_distribution = nn.functional.softmax(teacher_outputs.logits, dim=-1)
            # temperature is always 1 for eval
            kl_loss = kl_divergence(teacher_distribution, student_distribution, batch["labels"])
            # use Distil-Whisper formulation (fix weight of CE loss and tune KL weight)
            loss = 0.8 * ce_loss + training_args.kl_weight * kl_loss
            metrics["kl_loss"] = kl_loss
        else:
            loss = ce_loss

        metrics["loss"] = loss
        return metrics

    def generate_step(batch):
        output_ids = accelerator.unwrap_model(student_model).generate(
            **batch, max_length=max_label_length, num_beams=num_beams
        )
        output_ids = accelerator.pad_across_processes(output_ids, dim=1, pad_index=tokenizer.pad_token_id)
        return output_ids

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {total_train_steps * train_batch_size * gradient_accumulation_steps}")
    if not data_args.streaming:
        logger.info(f"  Num epochs = {num_epochs}")
    logger.info("  Instantaneous batch size per device =" f" {training_args.per_device_train_batch_size}")
    logger.info("  Gradient accumulation steps =" f" {gradient_accumulation_steps}")
    logger.info(
        f"  Total train batch size (w. parallel & distributed) = {train_batch_size * gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {total_train_steps}")

    # ======================== Training ================================
    train_time = 0
    train_start = time.time()
    steps_trained_progress_bar = tqdm(
        range(total_train_steps), desc="Train steps ... ", position=0, disable=not accelerator.is_local_main_process
    )
    continue_training = True
    epochs_trained = 0
    cur_step = 0

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if checkpoint is not None:
        accelerator.load_state(checkpoint)
        # Find num steps and epoch from saved state string pattern
        pattern = r"checkpoint-(\d+)-epoch-(\d+)"
        match = re.search(pattern, checkpoint)
        cur_step = int(match.group(1))
        epochs_trained = int(match.group(2))

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info(f"  Continuing training from epoch {epochs_trained}")
        logger.info(f"  Continuing training from global step {cur_step}")

        steps_trained_progress_bar.update(cur_step)

        for epoch in range(0, epochs_trained):
            vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(training_args.seed)

        if not data_args.streaming and training_args.max_steps < 0:
            # we know exactly the number of steps per epoch, so can skip through the required number of batches
            resume_step = (cur_step - epochs_trained * steps_per_epoch) * gradient_accumulation_steps
        else:
            # Currently we don't know how many steps we've taken in the current epoch
            # So we just shuffle the dataset one extra time and start from a fresh epoch
            # This is "good enough" for our purposes but not fully correct
            resume_step = None
            vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(training_args.seed)
    else:
        resume_step = None

    for epoch in range(epochs_trained, num_epochs):
        vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(training_args.seed)
        train_dataloader = DataLoader(
            vectorized_datasets["train"],
            collate_fn=data_collator,
            batch_size=per_device_train_batch_size,
            num_workers=dataloader_num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=training_args.dataloader_pin_memory,
        )
        train_dataloader = accelerator.prepare(train_dataloader)
        if hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(epoch)

        if resume_step is not None:
            # Skip the first N batches in the dataloader when resuming from a checkpoint
            train_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            resume_step = None

        for batch in train_dataloader:
            with accelerator.accumulate(student_model):
                loss, train_metric = train_step(batch)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(student_model.parameters(), training_args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Check if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                steps_trained_progress_bar.update(1)
                cur_step += 1

                if cur_step % training_args.logging_steps == 0:
                    steps_trained_progress_bar.write(
                        f"Step... ({cur_step} / {total_train_steps} | Loss:"
                        f" {train_metric['loss']}, Learning Rate:"
                        f" {lr_scheduler.get_last_lr()[0]})"
                    )
                    log_metric(
                        accelerator,
                        metrics=train_metric,
                        learning_rate=lr_scheduler.get_last_lr()[0],
                        train_time=train_time + time.time() - train_start,
                        step=cur_step,
                        epoch=epoch if data_args.streaming else epoch + (cur_step - epoch * steps_per_epoch) / steps_per_epoch,
                        prefix="train",
                    )

                # save checkpoint and weights after each save_steps and at the end of training
                if (cur_step % save_steps == 0) or cur_step == total_train_steps:
                    accelerator.wait_for_everyone()
                    intermediate_dir = os.path.join(training_args.output_dir, f"checkpoint-{cur_step}-epoch-{epoch}")
                    accelerator.save_state(output_dir=intermediate_dir)
                    unwrapped_model = accelerator.unwrap_model(student_model)
                    unwrapped_model.save_pretrained(
                        intermediate_dir,
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                    )
                    if accelerator.is_main_process:
                        checkpoint_to_be_deleted = rotate_checkpoints(training_args.save_total_limit, output_dir=training_args.output_dir)
                        if training_args.push_to_hub:
                            upload_folder(
                                folder_path=training_args.output_dir,
                                repo_id=repo_name,
                                repo_type="model",
                                commit_message=f"Saving train state of step {cur_step}",
                                delete_patterns=checkpoint_to_be_deleted,
                            )

                if training_args.do_eval and (cur_step % eval_steps == 0 or cur_step == total_train_steps):
                    train_time += time.time() - train_start
                    student_model.eval()
                    # ======================== Evaluating ==============================
                    for eval_split in all_eval_splits:
                        eval_metrics = []
                        eval_preds = []
                        eval_labels = []
                        eval_start = time.time()

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

                        eval_time = time.time() - eval_start
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
                                step=cur_step,
                                epoch=epoch,
                                evaluation_strategy=training_args.evaluation_strategy,
                                prefix=eval_split,
                            )

                        # Print metrics and update progress bar
                        logger_desc = " ".join([f"Eval {key}: {value} |" for key, value in eval_metrics.items()])
                        steps_trained_progress_bar.write(
                            f"Eval results for step ({cur_step} / {total_train_steps} | {logger_desc}"
                        )

                        log_metric(
                            accelerator,
                            metrics=eval_metrics,
                            train_time=eval_time,
                            step=cur_step,
                            epoch=epoch if data_args.streaming else epoch + (cur_step - epoch * steps_per_epoch) / steps_per_epoch,
                            prefix=eval_split,
                        )

                    # flush the train metrics
                    train_start = time.time()

                # break condition
                if cur_step == total_train_steps:
                    accelerator.wait_for_everyone()
                    # un-wrap student model for save
                    student_model = accelerator.unwrap_model(student_model)
                    student_model.save_pretrained(
                        training_args.output_dir,
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                    )
                    if training_args.push_to_hub and accelerator.is_main_process:
                        upload_folder(
                            folder_path=training_args.output_dir,
                            repo_id=repo_name,
                            repo_type="model",
                            commit_message=f"Saving final weights of step {cur_step}",
                        )
                    continue_training = False
                    break

        if not continue_training:
            break

    accelerator.end_training()


if __name__ == "__main__":
    main()

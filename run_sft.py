# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
import os
import sys
from dataclasses import field, dataclass
from typing import Optional, Union, List, Dict

import numpy as np
from tqdm import tqdm
from transformers.trainer_utils import get_last_checkpoint
from trl.commands.cli_utils import TrlParser
import torch
from datasets import load_dataset, DatasetDict, Dataset, IterableDataset, interleave_datasets, concatenate_datasets
import datasets
import transformers
from transformers import AutoTokenizer, TrainingArguments, set_seed, AutoConfig
from trl import (
    ModelConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map, DataCollatorForCompletionOnlyLM,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments(ModelConfig):
    output_router_logits: bool = field(
        default=True, metadata={"help": "Whether or not to return the router logits in the forward pass. Enabling this will also configure the model to compute the auxiliary loss."}
    )

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_dataset_name: str = field(
        default=None,
        metadata={
            "help": "The name of the training dataset to use (via the datasets library). Load and combine "
            "multiple datasets by separating dataset ids by a '+' symbol. For example, to load LibriSpeech "
            "and Common Voice, set `train_dataset_name='librispeech_asr+common_voice'`."
        },
    )
    train_dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the training dataset to use (via the datasets library). Load and combine "
            "multiple datasets by separating dataset configs by a '+' symbol. Note that the order of the configs should "
            "match the order of the datasets."
        },
    )
    train_dataset_samples: str = field(
        default=None,
        metadata={
            "help": "Number of samples in each dataset when loading multiple datasets with streaming mode. "
            "Not required when using one dataset or non-streaming mode. The sample values provide the sampling "
            "probability for each dataset. Setting them equal to the number of sample values ensures that every "
            "sample from every dataset is used once per epoch."
        },
    )
    eval_dataset_name: str = field(
        default=None,
        metadata={
            "help": "The name of the evaluation dataset to use (via the datasets library). Defaults to the training "
            "dataset name if unspecified. Load multiple evaluation datasets by separating dataset "
            "ids by a '+' symbol."
        },
    )
    eval_dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the evaluation dataset to use (via the datasets library). Defaults to the "
            "training dataset config name if unspecified."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    prompt_column_name: str = field(
        default="prompt",
        metadata={"help": "The name of the dataset column containing the prompt data. Defaults to 'prompt'"},
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    max_seq_length: int = field(default=2048, metadata={"help": "The maximum sequence length for the SFT Trainer."})
    packing: bool = field(default=False, metadata={"help": "Whether to apply data packing or not during training"})
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
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
        dataset_names = dataset_names.split("+")
        dataset_config_names = dataset_config_names.split("+") if dataset_config_names is not None else None
        splits = splits.split("+") if splits is not None else None
        text_column_names = text_column_names.split("+") if text_column_names is not None else None
        prompt_column_names = prompt_column_names.split("+") if prompt_column_names is not None else None
        dataset_samples = dataset_samples.split("+") if dataset_samples is not None else None

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
    training_args: Optional[TrainingArguments] = None,
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
        disable=not training_args.distributed_state.is_main_process,
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

    return interleaved_dataset

def main():
    # 1. Parse input arguments
    parser = TrlParser((DataArguments, TrainingArguments, ModelArguments))
    data_args, training_args, model_args = parser.parse_args_and_config()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 4. Load dataset
    raw_datasets = DatasetDict()

    if training_args.do_train:
        raw_datasets["train"] = load_multiple_datasets(
            data_args.train_dataset_name,
            data_args.train_dataset_config_name,
            splits=data_args.train_split_name,
            text_column_names=data_args.text_column_name,
            prompt_column_names=data_args.prompt_column_name,
            dataset_samples=data_args.train_dataset_samples,
            seed=training_args.seed,
            training_args=training_args,
            cache_dir=data_args.cache_dir,
            token=data_args.token,
        )

    if training_args.do_eval:
        raw_datasets["eval"] = load_dataset(
            data_args.eval_dataset_name if data_args.eval_dataset_name else data_args.train_dataset_name,
            data_args.eval_dataset_config_name if data_args.eval_dataset_config_name else data_args.train_dataset_config_name,
            split=data_args.eval_split_name,
            cache_dir=data_args.cache_dir,
            token=data_args.token,
            num_proc=data_args.preprocessing_num_workers,
        )

    split_for_features = "train" if training_args.do_train else "eval"
    raw_datasets_features = list(raw_datasets[split_for_features].features.keys())

    if data_args.max_train_samples is not None:
        raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))

    if data_args.max_eval_samples is not None:
        raw_datasets["eval"] = raw_datasets["eval"].select(range(data_args.max_eval_samples))

    # 5. Define pre-trained model args and load tokenizer
    # When performing distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        cache_dir=data_args.cache_dir,
    )
    if model_args.output_router_logits:
        model_kwargs["output_router_logits"] = True

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer.padding_side = "right"
    tokenizer.model_max_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    instruction_template = "### Human:"
    response_template = "### Assistant:"

    def prepare_dataset(batch):
        all_examples = []
        for prompt, text in zip(batch["prompt"], batch["text"]):
            all_examples.append(f"{instruction_template} {prompt}\n {response_template} {text}")
        return all_examples

    data_collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)

    # 6. Initialize Trainer
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["eval"],
        dataset_text_field=data_args.text_column_name,
        max_seq_length=data_args.max_seq_length,
        tokenizer=tokenizer,
        formatting_func=prepare_dataset,
        data_collator=data_collator,
        packing=data_args.packing,
        dataset_num_proc=data_args.preprocessing_num_workers,
        peft_config=get_peft_config(model_args),
    )

    # 7. Training
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # 8. Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["eval"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 9. Write Training Stats
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    train_dataset_name = data_args.train_dataset_name.split("+")[0] if data_args.train_dataset_name else None
    train_dataset_config_name = data_args.train_dataset_config_name.split("+")[0] if data_args.train_dataset_config_name else None
    if train_dataset_name is not None:
        kwargs["dataset_tags"] = data_args.train_dataset_name
        if train_dataset_config_name is not None:
            kwargs["dataset_args"] = train_dataset_config_name
            kwargs["dataset"] = f"{train_dataset_name} {train_dataset_config_name}"
        else:
            kwargs["dataset"] = train_dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

if __name__ == "__main__":
    main()


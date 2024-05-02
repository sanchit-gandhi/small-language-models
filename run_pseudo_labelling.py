import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import torch
from accelerate import Accelerator, skip_first_batches
from accelerate.logging import get_logger
from datasets import DatasetDict, load_dataset
from huggingface_hub import create_repo, get_full_repo_name
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser, PreTrainedTokenizerBase, BatchEncoding,
)


logger = get_logger(__name__, log_level="INFO")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to what data we are going to input our model for eval.
    """

    model_name_or_path: str = field(
        metadata={"help": "The name of the model to use (via the transformers library) for the prompt annotation."},
    )
    per_device_eval_batch_size: int = field(
        metadata={"help": "The per-device batch size to use for inference."},
    )
    model_variant: str = field(
        default=None,
        metadata={"help": "If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin. "},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    dtype: Optional[str] = field(
        default="float16",
        metadata={
            "help": (
                "Floating-point format in which the model weights should be initialized"
                " and the computations run. Choose one of `[float32, float16, bfloat16]`."
            )
        },
    )
    attn_implementation: Optional[str] = field(
        default="sdpa",
        metadata={"help": "Which attn type to use: ['eager', 'sdpa', 'flash_attention_2']"},
    )
    load_in_8bit: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use 8-bit precision for inference."}
    )
    load_in_4bit: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use 4-bit precision for inference."}
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"}
    )
    use_bnb_nested_quant: Optional[bool] = field(default=False, metadata={"help": "use nested quantization"})
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=True, metadata={"help": "Use fast tokenizer for encoding/decoding input ids"}
    )
    token: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether or not to use an authentication token when loading/uploading from the Hugging Face Hub"
        },
    )
    temperature: Optional[float] = field(default=0.6, metadata={"help": "Temperature for sampling-based generation"})
    torch_compile: Optional[bool] = field(
        default=False, metadata={"help": "Whether to compile the forward pass (not sampling) in generate."}
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    output_dir: str = field(
        metadata={
            "help": "Where to save the processed dataset to disk. If unspecified, uses a 'pretty' version of the "
            "original dataset name. E.g. 'facebook/voxpopuli' will be saved under 'voxpopuli'."
        },
    )
    dataset_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)"},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    dataset_split_name: Optional[str] = field(
        default=None,
        metadata={"help": "The split name of the dataset to use (via the datasets library)."},
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to cache directory for saving and loading datasets"},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples for generation - use for debugging purposes."},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the generated text data in the training set."},
    )
    prompt_column_name: str = field(
        default="prompt",
        metadata={"help": "The name of the dataset column containing the prompt data. Defaults to 'prompt'"},
    )
    max_label_length: int = field(
        default=4096,
        metadata={"help": "Truncate target labels that are longer `max_label_length` tokens."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    dataloader_num_workers: Optional[int] = field(
        default=0,
        metadata={"help": "The number of processes to use for the dataloader."},
    )
    push_to_hub: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to push the processed dataset to the Hub."},
    )
    hub_dataset_id: Optional[str] = field(
        default=None,
        metadata={"help": "Repository namespace if pushing to the Hugging Face Hub."},
    )
    overwrite_output_dir: Optional[bool] = field(
        default=False,
        metadata={"help": "Overwrite the content of the output directory each time the script is run."},
    )
    save_steps: Optional[int] = field(
        default=500,
        metadata={"help": "Save the generated prompts every save_steps."},
    )
    save_total_limit: Optional[int] = field(
        default=1, metadata={"help": "If a value is passed, will limit the total number of saved checkpoints"}
    )
    report_to: Union[None, str, List[str]] = field(
        default=None, metadata={"help": "The list of integrations to report the results and logs to."}
    )
    wandb_project: Optional[str] = field(
        default="distil-mistral",
        metadata={"help": "The name of the wandb project."},
    )


def get_quantization_config(model_args: ModelArguments) -> Union[BitsAndBytesConfig, None]:
    if model_args.load_in_4bit:
        compute_dtype = torch.float16
        if model_args.dtype not in {"auto", None}:
            compute_dtype = getattr(torch, model_args.dtype)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    return quantization_config


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
        # don't include prompts in log-probs calculation
        for idx in range(len(prompt_lengths)):
            labels_mask[idx, : prompt_lengths[idx]] = 0
        # replace padding with -100 to ignore from log-probs correctly
        labels = batch["input_ids"].masked_fill(labels_mask.ne(1), -100)
        batch["labels"] = labels

        return batch

CHECKPOINT_PREFIX = "checkpoint"
_RE_CHECKPOINT = re.compile(r"^checkpoint-(\d+).json$")


def save_checkpoint(output_dir, all_logprobs, step):
    checkpoint_path = f"{CHECKPOINT_PREFIX}-{step}.json"
    output_path = os.path.join(output_dir, checkpoint_path)
    all_logprobs = [float(logprob) for logprob in all_logprobs]
    with open(output_path, "w") as file:
        json.dump(all_logprobs, file)


def load_checkpoint(checkpoint_path):
    with open(checkpoint_path, "r") as file:
        all_logprobs = json.load(file)
    return all_logprobs

def clean_checkpoints(output_dir):
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        # Check if it's a directory
        if os.path.isdir(item_path):
            # Recursively clean subdirectories
            clean_checkpoints(item_path)
            # Remove the empty directory
            shutil.rmtree(item_path)

def sorted_checkpoints(output_dir=None) -> List[str]:
    """Helper function to sort saved checkpoints from oldest to newest."""
    ordering_and_checkpoint_path = []

    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{CHECKPOINT_PREFIX}-*")]

    for path in glob_checkpoints:
        regex_match = re.match(f".*{CHECKPOINT_PREFIX}-([0-9]+)", path)
        if regex_match is not None and regex_match.groups() is not None:
            ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def rotate_checkpoints(save_total_limit=None, output_dir=None) -> None:
    """Helper function to delete old checkpoints."""
    if save_total_limit is None or save_total_limit <= 0:
        return
    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = sorted_checkpoints(output_dir=output_dir)
    if len(checkpoints_sorted) <= save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
        os.remove(checkpoint)


def get_last_checkpoint(folder) -> Tuple[List, int]:
    if not os.path.exists(folder) or not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
        return [], 0
    content = os.listdir(folder)
    checkpoints = [path for path in content if _RE_CHECKPOINT.search(path) is not None]
    if len(checkpoints) == 0:
        return [], 0
    last_checkpoint = os.path.join(folder, max(checkpoints, key=lambda x: int(_RE_CHECKPOINT.search(x).groups()[0])))
    # Find num steps saved state string pattern
    pattern = r"checkpoint-(\d+).json"
    match = re.search(pattern, last_checkpoint)
    cur_step = int(match.group(1))
    # load corresponding generated ids
    all_logprobs = load_checkpoint(last_checkpoint)
    return all_logprobs, cur_step


def main():
    # 1. Parse input arguments
    parser = HfArgumentParser((ModelArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()

    # 2. Setup logging
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    accelerator = Accelerator(log_with=data_args.report_to, project_dir=data_args.output_dir,)
    accelerator.init_trackers(project_name=data_args.wandb_project)

    if data_args.overwrite_output_dir and os.path.exists(data_args.output_dir):
        logger.info("Cleaning output dir from previous run...")
        clean_checkpoints(data_args.output_dir)

    # 5. Handle the repository creation
    if accelerator.is_main_process:
        if data_args.output_dir is not None:
            os.makedirs(data_args.output_dir, exist_ok=True)
        if data_args.push_to_hub:
            if data_args.hub_dataset_id is None:
                repo_name = get_full_repo_name(
                    Path(data_args.output_dir).absolute().name,
                    token=model_args.token,
                )
            else:
                repo_name = data_args.hub_dataset_id
            create_repo(repo_name, exist_ok=True, token=model_args.token)
            with open(os.path.join(data_args.output_dir, ".gitignore"), "w+") as gitignore:
                if "wandb" not in gitignore:
                    gitignore.write("wandb\n")
    accelerator.wait_for_everyone()

    # 3. Load annotated dataset
    logger.info("*** Load annotated dataset ***")
    if data_args.dataset_split_name is not None:
        raw_datasets = DatasetDict()
        data_splits = data_args.dataset_split_name.split("+")
        # load on a split-wise basis
        for split in data_splits:
            raw_datasets[split] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=split,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                num_proc=data_args.preprocessing_num_workers,
            )
    else:
        # load all splits for annotation
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            num_proc=data_args.preprocessing_num_workers,
        )

    if data_args.max_eval_samples is not None:
        for split in raw_datasets:
            raw_datasets[split] = raw_datasets[split].select(range(data_args.max_eval_samples))

    # 4. Load pre-trained model
    logger.info("*** Load pretrained model ***")
    dtype = (
        model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    )
    quantization_config = get_quantization_config(model_args)
    current_device = Accelerator().local_process_index if torch.cuda.is_available() else "cpu"
    device_map = {"": current_device} if (torch.cuda.is_available() and quantization_config) else None

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        variant=model_args.model_variant,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=dtype,
        device_map=device_map,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        token=model_args.token,
    ).eval()

    if model_args.torch_compile:
        # torch compile only compatible with gemma and llama
        if not callable(getattr(model, "_setup_cache", None)):
            raise ValueError(
                f"Static k/v cache is not compatible with the model {model.__class__.__name__}. Set `--torch_compile=False"
                "for dynamic k/v cache"
            )
        model.generation_config.cache_implementation = "static"
        # compile the forward pass (but not the top-{p,k} sampling)
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=model_args.use_fast_tokenizer,
        padding_side="left",
    )

    # define some constants
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.bos_token_id
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    max_label_length = (
        data_args.max_label_length if data_args.max_label_length is not None else model.config.max_length
    )
    text_column_name = data_args.text_column_name
    prompt_column_name = data_args.prompt_column_name
    rescale_temperature = model_args.temperature if model_args.temperature > 0.0 else 1

    def prepare_instruction_dataset(example):
        messages = [
            {"role": "user", "content": example[prompt_column_name]},
            {"role": "assistant", "content": example[text_column_name]},
        ]
        example["labels"] = tokenizer.apply_chat_template(messages)
        example["prompt_length"] = len(tokenizer.apply_chat_template([messages[0]]))
        return example

    def is_labels_in_length_range(labels):
        return 0 < len(labels) <= max_label_length

    with accelerator.main_process_first():
        vectorized_datasets = raw_datasets.map(
            prepare_instruction_dataset,
            num_proc=data_args.preprocessing_num_workers,
            desc="Tokenizing dataset...",
        )
        vectorized_datasets = vectorized_datasets.filter(
            is_labels_in_length_range, num_proc=data_args.preprocessing_num_workers, desc="Filtering dataset..."
        )

    # Prepare everything with our `accelerator`
    model = accelerator.prepare(model)
    data_collator = DataCollatorCausalLMWithPadding(
        tokenizer,
        target_padding="max_length",
        max_target_length=max_label_length,
    )

    @torch.no_grad()
    def eval_step(batch):
        shifted_input_ids = batch.get("input_ids")[..., 1:]
        shifted_labels = batch.pop("labels")[..., 1:]
        shifted_logits = model(**batch).logits[..., :-1, :]

        logprobs = nn.functional.log_softmax((shifted_logits * rescale_temperature).float(), dim=-1).to(shifted_logits.dtype)
        logprobs = torch.gather(logprobs, dim=2, index=shifted_input_ids.unsqueeze(-1))

        sum_logprobs = (logprobs.squeeze(-1) * (shifted_labels != -100)).sum(-1)
        lengths = (shifted_labels != -100).sum(-1)
        avg_logprobs = sum_logprobs / lengths
        return avg_logprobs

    for split in vectorized_datasets:
        data_loader = DataLoader(
            vectorized_datasets[split],
            batch_size=model_args.per_device_eval_batch_size,
            collate_fn=data_collator,
            num_workers=data_args.dataloader_num_workers,
            pin_memory=True,
        )
        data_loader = accelerator.prepare(data_loader)
        total_inference_steps = len(data_loader)
        progress_bar = tqdm(
            range(total_inference_steps), desc=" ... ", position=0, disable=not accelerator.is_local_main_process
        )
    
        split_output_dir = os.path.join(data_args.output_dir, split)
        all_logprobs, cur_step = get_last_checkpoint(split_output_dir)
    
        if cur_step > 0:
            logger.info(f"Resuming {split} from step {cur_step}")
            # efficiently skip the first n batches
            data_loader = skip_first_batches(data_loader, cur_step)
            progress_bar.update(cur_step)
    
        while cur_step < total_inference_steps:
            for batch in data_loader:
                logprobs = eval_step(batch)
                logprobs = accelerator.gather_for_metrics(logprobs)
                all_logprobs.extend(logprobs.cpu().numpy())

                cur_step += 1
                progress_bar.update(1)

                if (cur_step % data_args.save_steps == 0) or (cur_step == total_inference_steps):
                    save_checkpoint(split_output_dir, all_logprobs, cur_step)
                    rotate_checkpoints(data_args.save_total_limit, output_dir=split_output_dir)

        vectorized_datasets[split] = vectorized_datasets[split].add_column("logprobs", all_logprobs)

    if accelerator.is_main_process:
        vectorized_datasets.save_to_disk(data_args.output_dir)
        if data_args.push_to_hub:
            vectorized_datasets.push_to_hub(
                repo_name,
                config_name=data_args.dataset_config_name if data_args.dataset_config_name is not None else "default",
                token=model_args.token,
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()

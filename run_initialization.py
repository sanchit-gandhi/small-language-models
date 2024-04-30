import copy
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from huggingface_hub import create_repo, get_full_repo_name, upload_folder
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """

    model_name_or_path: Optional[str] = field(
        metadata={"help": "The teacher checkpoint for weights initialization"},
    )
    output_dir: str = field(
        metadata={"help": "The output directory where the student checkpoint will be written."},
    )
    model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The specific teacher model version to use (can be a branch name, tag name or commit id)."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pre-trained models downloaded from huggingface.co"},
    )
    subfolder: Optional[str] = field(
        default="",
        metadata={
            "help": "In case the relevant files are located inside a subfolder of the teacher model repo on huggingface.co, you can"
            "specify the folder name here."
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the teacher model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    trust_remote_code: Optional[bool] = field(
        default=False, metadata={"help": "Trust remote code when loading a model."}
    )
    token: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` necessary to use this script with private models)."
        },
    )
    num_hidden_layers: Optional[int] = field(
        default=6,
        metadata={"help": "The number of hidden layers in the Transformer decoder."},
    )
    push_to_hub: Optional[bool] = field(
        default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
    )
    hub_model_id: Optional[str] = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    low_cpu_mem_usage: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Create the teacher model as an empty shell, and only materialize its parameters when the pretrained weights are loaded. "
            "Significantly benefits loading time and RAM consumption."
        },
    )
    initialization_strategy: Optional[str] = field(
        default="maximally_spaced",
        metadata={
            "help": "The weight initialization strategy for the decoder weights. Either `first_n`, or `maximally_spaced`."
        },
    )


def main():
    # 1. Parse input arguments
    parser = HfArgumentParser(ModelArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        model_args = parser.parse_args_into_dataclasses()[0]

    logger.info(f"Model parameters {model_args}")

    logger.info("*** Load pretrained teacher model ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    # quantization_config = get_quantization_config(model_args)

    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        revision=model_args.model_revision,
        cache_dir=model_args.cache_dir,
        subfolder=model_args.subfolder,
        trust_remote_code=model_args.trust_remote_code,
        token=model_args.token,
        # device_map=get_kbit_device_map() if quantization_config is not None else None,
        # quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    generation_config = teacher_model.generation_config
    teacher_config = teacher_model.config

    logger.info("*** Teacher model loaded! ***")

    student_config = copy.deepcopy(teacher_config)
    student_config.num_hidden_layers = model_args.num_hidden_layers
    teacher_hidden_layers = teacher_config.num_hidden_layers

    if model_args.initialization_strategy == "maximally_spaced":
        decoder_mapping = np.linspace(0, teacher_hidden_layers - 1, student_config.num_hidden_layers, dtype=int)
        decoder_mapping[-1] = teacher_hidden_layers - 1
    elif model_args.initialization_strategy == "first_n":
        decoder_mapping = np.arange(0, student_config.num_hidden_layers)
    else:
        raise ValueError(
            f"Got invalid initialization_strategy strategy '{model_args.initialization_strategy}', should be one of "
            "'maximally_spaced` or `first_n`."
        )

    decoder_map = {}
    for student_layer, teacher_layer in enumerate(decoder_mapping):
        decoder_map[teacher_layer] = student_layer

    # init the student params from the teacher model
    logger.info("*** Load and initialise student model ***")
    student_model = AutoModelForCausalLM.from_config(student_config)
    missing_keys, unexpected_keys = student_model.load_state_dict(teacher_model.state_dict(), strict=False)
    student_model.to(dtype=torch_dtype)
    if len(missing_keys) > 0:
        raise RuntimeError(
            f"Error(s) in loading state_dict for {student_model.__class__.__name__}. \n"
            f"Missing key(s) in state_dict: {missing_keys}"
        )
    if student_config.num_hidden_layers == teacher_hidden_layers:
        decoder_keys = [key for key in unexpected_keys if "model.layers" in key]
        if len(decoder_keys) > 0:
            raise RuntimeError(
                f"Error(s) in loading state_dict for {student_model.__class__.__name__}. \n"
                f"Unexpected key(s) in state_dict: {decoder_keys}"
            )

    for layer in range(teacher_hidden_layers):
        if layer in decoder_map:
            # re-introduce pre-defined layers from the teacher
            student_model.model.layers[decoder_map[layer]].load_state_dict(
                teacher_model.model.layers[layer].state_dict()
            )

    logger.info("*** Student model loaded! ***")

    # remove the teacher params and model
    del teacher_model

    # save the converted weights and model
    if model_args.output_dir is not None:
        student_model.save_pretrained(model_args.output_dir)
        # we also need to correctly save the processor and generation config
        tokenizer.save_pretrained(model_args.output_dir)
        generation_config.save_pretrained(model_args.output_dir)

    if model_args.push_to_hub:
        if model_args.hub_model_id is None:
            repo_name = get_full_repo_name(
                Path(model_args.output_dir).absolute().name,
                token=model_args.token,
            )
        else:
            repo_name = model_args.hub_model_id
        create_repo(repo_name, exist_ok=True, token=model_args.token)
        upload_folder(
            repo_id=repo_name,
            folder_path=model_args.output_dir,
            commit_description="Uploading initialised weights and configs",
        )


if __name__ == "__main__":
    main()

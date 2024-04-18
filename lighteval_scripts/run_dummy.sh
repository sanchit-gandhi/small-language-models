accelerate launch run_evals_accelerate.py \
    --model_args "pretrained=hf-internal-testing/tiny-random-MistralForCausalLM" \
    --tasks "./all_cosmo_tasks.txt" \
    --override_batch_size 8 \
    --output_dir "./"
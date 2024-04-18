accelerate launch run_evals_accelerate.py \
    --model_args "pretrained=sanchit-gandhi/distil-mistral-1.5B-v0.1" \
    --tasks ./all_cosmo_tasks.txt \
    --override_batch_size 8 \
    --output_dir "./"

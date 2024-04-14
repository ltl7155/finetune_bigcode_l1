# need to give deepspeed config file as argument
if [ -z "$1" ]
  then
    echo "No dataset name"
    exit 1
fi

if [ -z "$2" ]
  then
    echo "No output_dir"
    exit 1
fi

if [ -z "$3" ]
  then
    echo "No checkpoint"
    exit 1
fi

python3 -m torch.distributed.launch \
        --nproc_per_node 8 \
        main_l1.py \
        --deepspeed="deepspeed_cfgs/deepspeed_z3_config_bf16_andy.json" \
        --model_path="$3" \
        --dataset_name="$1" \
        --output_dir="$2" \
        --seq_length 2048 \
        --epochs 2 \
        --batch_size 1 \
        --gradient_accumulation_steps 2 \
        --learning_rate $5 \
        --num_warmup_steps 10 \
        --num_workers=$(expr $(nproc --all) - 4) \
        --no_fp16 \
        --bf16 \
        --perc_valid_set 0.0025 \
        --save_total_limit 20 \
        --humaneval_eval_loss \
        --lam $4

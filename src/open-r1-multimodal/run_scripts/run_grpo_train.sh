# Please run this script under `PhotoFramer-Assessment/src/open-r1-multimodal`

export PYTHONPATH=./src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATA_DIR=../../Datasets
QWEN_DIR=../../ModelZoo/Qwen2.5-VL-7B-Instruct
export DEBUG_MODE="true"  # if you do not want debug log, set to false
RUN_NAME="Qwen2.5-VL-7B-GRPO-Composition-Score-Class"
export LOG_PATH="./debug_log_$RUN_NAME.txt"

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo_composition.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path $QWEN_DIR \
    --dataset_name "" \
    --data_paths $DATA_DIR/GAIC/metas/train_gaic_48k_sampled.json \
                 $DATA_DIR/CADB_Dataset/metas/train_cadb_score_8.5k.json \
                 $DATA_DIR/CADB_Dataset/metas/train_cadb_class_8.5k.json \
                 $DATA_DIR/KU_PCP_Dataset/metas/train_kupcp_class_3k_resize1024.json \
                 $DATA_DIR/AVA_dataset/metas/train_ava_sampled_43k.json \
    --weights 1 5 5 5 1 \
    --tasks composition_score composition_score composition_class composition_class composition_score \
    --image_root $DATA_DIR \
    --max_prompt_length 1024 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 141 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 1000 \
    --save_only_model true \
    --freeze_vision_modules false \
    # --beta 0.0 \  # weight of kl divergence
    # --epsilon 0.2 \  # clip parameter [1 - epsilon, 1 + epsilon]
    # --epsilon_high 0.28 \  # higher clip parameter if given

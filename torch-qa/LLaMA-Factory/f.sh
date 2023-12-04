MODEL_ID=microsoft_Orca-2-7b
MODEL_ID=meta_llama-2-7b-chat-hf
MODEL_ID=NousResearch_llama-2-7b-chat-hf
MODEL_ID=Mistral-7B-Instruct-v0.1
DATASET=qa
TRAIN_BATCH_SIZE=1
NUM_TRAIN_EPOCHS=6
SAVE_STEPS=100
#LORA_TARGET=q_proj,v_proj
LORA_TARGET=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head
#LORA_RANK=8
LORA_RANK=64
LEARNING_RATE=1e-4

if [ -e outputs/$MODEL_ID/adapter_config.json ]; then
   echo "resume from adapter_config.json"
   CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
      --stage sft \
      --model_name_or_path ../models/$MODEL_ID \
      --do_train \
      --dataset $DATASET \
      --template llama2 \
      --finetuning_type lora \
      --lora_target $LORA_TARGET \
      --resume_lora_training True \
      --output_dir outputs/$MODEL_ID \
      --checkpoint_dir outputs/$MODEL_ID \
      --overwrite_output_dir true \
      --per_device_train_batch_size $TRAIN_BATCH_SIZE \
      --num_train_epochs $NUM_TRAIN_EPOCHS \
      --lr_scheduler_type cosine \
      --gradient_accumulation_steps 4 \
      --save_steps $SAVE_STEPS \
      --lora_alpha 16 \
      --lora_dropout 0.1 \
      --lora_rank $LORA_RANK \
      --logging_steps 10 \
      --learning_rate $LEARNING_RATE \
      --quantization_bit 4 \
      --double_quantization True \
      --quantization_type nf4 \
      --overwrite_cache \
      --fp16 \
      --plot_loss
else
   echo "train"
   CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
      --stage sft \
      --model_name_or_path ../models/$MODEL_ID \
      --do_train \
      --dataset $DATASET \
      --template llama2 \
      --finetuning_type lora \
      --lora_target $LORA_TARGET \
      --resume_lora_training True \
      --output_dir outputs/$MODEL_ID \
      --per_device_train_batch_size $TRAIN_BATCH_SIZE \
      --num_train_epochs $NUM_TRAIN_EPOCHS \
      --lr_scheduler_type cosine \
      --gradient_accumulation_steps 4 \
      --save_steps $SAVE_STEPS \
      --lora_alpha 16 \
      --lora_dropout 0.1 \
      --lora_rank $LORA_RANK \
      --logging_steps 10 \
      --learning_rate $LEARNING_RATE \
      --quantization_bit 4 \
      --double_quantization True \
      --quantization_type nf4 \
      --overwrite_cache \
      --fp16 \
      --plot_loss
fi
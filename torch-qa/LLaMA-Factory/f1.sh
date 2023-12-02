# llama-2-13b-chat-hf
#--lora_target q_proj,v_proj \
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
--stage sft \
--model_name_or_path ../models/llama-2-7b-chat-hf \
--do_train \
--dataset qa \
--template default \
--finetuning_type lora \
--lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head \
--lora_alpha 16 \
--lora_rank 64 \
--lora_dropout 0.1 \
--resume_lora_training True \
--output_dir outputs \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 4 \
--lr_scheduler_type cosine \
--logging_steps 10 \
--save_steps 5 \
--learning_rate 1e-5 \
--num_train_epochs 1.0 \
--quantization_bit 4 \
--plot_loss 
#--checkpoint_dir outputs/checkpoint \
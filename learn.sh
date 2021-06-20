#!/bin/bash

code=main.py
experiment_name=052_reformer_nocut_singletext_noweight_resnet50
data_name=014_flat_seq
ckpt_name=ckpt
step_load=30000
step_max=30000
batch_size=64
batch_size_val=32 #8
g_steps=8
num_workers=4
fp16_opt_level=O1
max_grad_norm=1.0
loss_scale=0
step_save=10000
step_valid=5000
step_log=1000

#mode=train
#mode=test
mode=embed
dbg=1
#dbg=0
#log_level=DEBUG
log_level=INFO
log=${experiment_name}.log
GPU=-1


# debug
if [ ${dbg} == 1 ]; then
    CUDA_VISIBLE_DEVICES=${GPU} python ${code} \
	--experiment_name ${experiment_name} \
	--data_name ${data_name} \
	--ckpt_name ${ckpt_name} \
	--mode ${mode} \
	--num_workers ${num_workers} \
	--step_load ${step_load} \
	--step_max ${step_max} \
	--batch_size ${batch_size} \
	--batch_size_val ${batch_size_val} \
	--gradient_accumulation_steps ${g_steps} \
	--fp16_opt_level ${fp16_opt_level} \
	--max_grad_norm ${max_grad_norm} \
	--loss_scale ${loss_scale} \
	--log_level ${log_level} \
	--step_save ${step_save} \
	--step_valid ${step_valid} \
	--step_log ${step_log} \
	--fp16 \
	#--use_pretrain \
	#--resnet_cpu \
    

# no debug
else
    CUDA_VISIBLE_DEVICES=${GPU} nohup python ${code} \
	--experiment_name ${experiment_name} \
	--data_name ${data_name} \
	--ckpt_name ${ckpt_name} \
	--mode ${mode} \
	--num_workers ${num_workers} \
	--step_load ${step_load} \
	--step_max ${step_max} \
	--batch_size ${batch_size} \
	--batch_size_val ${batch_size_val} \
	--gradient_accumulation_steps ${g_steps} \
	--fp16_opt_level ${fp16_opt_level} \
	--max_grad_norm ${max_grad_norm} \
	--loss_scale ${loss_scale} \
	--log_level ${log_level} \
	--step_save ${step_save} \
	--step_valid ${step_valid} \
	--step_log ${step_log} \
	--fp16 \
	>> ${log} &
	#--resnet_cpu \
	#--use_pretrain \
fi

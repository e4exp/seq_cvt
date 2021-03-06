#!/bin/bash

code=main.py
experiment_name=031_reformer_ds
data_name=014_flat_seq
ckpt_name=ckpt
step_load=0
step_max=10000
batch_size=16
batch_size_val=8
g_steps=1
num_workers=4
fp16_opt_level=O2
max_grad_norm=1.0
loss_scale=0
step_save=1000
step_log=10

mode=train
#mode=test
#mode=extract
dbg=1
#dbg=0
#log_level=DEBUG
log_level=INFO
log=${experiment_name}.log

# debug
if [ ${dbg} == 1 ]; then
    python ${code} \
	--experiment_name ${experiment_name} \
	--data_name ${data_name} \
	--ckpt_name ${ckpt_name} \
	--mode ${mode} \
	--step_load ${step_load} \
	--step_max ${step_max} \
	--batch_size ${batch_size} \
	--batch_size_val ${batch_size_val} \
	--gradient_accumulation_steps ${g_steps} \
	--num_workers ${num_workers} \
	--fp16_opt_level ${fp16_opt_level} \
	--max_grad_norm ${max_grad_norm} \
	--loss_scale ${loss_scale} \
	--log_level ${log_level} \
	--step_save ${step_save} \
	--step_log ${step_log} \
	#--use_pretrain \
    #--fp16 \

# no debug
else
    nohup python ${code} \
	--experiment_name ${experiment_name} \
	--data_name ${data_name} \
	--ckpt_name ${ckpt_name} \
	--mode ${mode} \
	--step_load ${step_load} \
	--step_max ${step_max} \
	--batch_size ${batch_size} \
	--batch_size_val ${batch_size_val} \
	--gradient_accumulation_steps ${g_steps} \
	--num_workers ${num_workers} \
	--fp16_opt_level ${fp16_opt_level} \
	--max_grad_norm ${max_grad_norm} \
	--loss_scale ${loss_scale} \
	--log_level ${log_level} \
	--step_save ${step_save} \
	--step_log ${step_log} \
	>> ${log} &
	#--use_pretrain \
fi

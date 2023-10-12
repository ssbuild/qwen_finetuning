#!/usr/bin/env bash

export trainer_backend=pl

train_config="./config/train_${trainer_backend}.yaml"

# 强制覆盖配置文件
export train_config=${train_config}
export enable_deepspeed=false
export enable_ptv2=true
export enable_lora=false
export load_in_bit=0

mode="train"
while getopts m: opt
do
	case "${opt}" in
		m) mode=${OPTARG};;
	esac
done


if [[ "${mode}" == "dataset" ]]
then
    python ../data_utils.py
else
  if [[ "${trainer_backend}" == "pl" ]]
  then
     # pl 多卡 修改配置文件 devices
    python ../train.py
  else
    # 多机多卡
    # --nproc_per_node=1 nnodes=1 --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT ../train.py
    torchrun --nproc_per_node 1 --nnodes 1 ../train.py
  fi
fi
# -*- coding: utf-8 -*-
# @Time:  20:50
# @Author: tk
# @File：convert_lora_to_peft
import gc
import os
import re

import torch
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser, GenerationConfig
from data_utils import train_info_args, NN_DataHelper, global_args
from aigc_zoo.model_zoo.qwen.llm_model import MyTransformer, QWenTokenizer, setup_model_profile, QWenConfig, \
    EffiArguments,LoraModel



def convert_to_peft(ckpt_dir,output_dir = './peft_lora'):
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(train_info_args, allow_extra_keys=True)
    setup_model_profile()
    dataHelper = NN_DataHelper(model_args, )
    tokenizer: QWenTokenizer
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=QWenTokenizer, config_class_name=QWenConfig)


    config = QWenConfig.from_pretrained(ckpt_dir)
    lora_args = EffiArguments.from_pretrained(ckpt_dir)
    # 非推理模式
    lora_args.inference_mode = False

    new_num_tokens = config.vocab_size
    if config.task_specific_params is not None and config.task_specific_params.get('vocab_size', None) is not None:
        config.vocab_size = config.task_specific_params['vocab_size']

    pl_model = MyTransformer(config=config, model_args=model_args, lora_args=lora_args,
                             torch_dtype=torch.float16,
                             new_num_tokens=new_num_tokens,#扩充词
                             # load_in_8bit=global_args["load_in_8bit"],
                             # # device_map="auto",
                             # device_map = {"":0} # 第一块卡
                             )
    # 加载lora权重
    pl_model.load_sft_weight(ckpt_dir, is_trainable=True)

    pl_model.save_peft_weight(output_dir)
    return tokenizer,config,lora_args



def get_base_model(ckpt_dir):
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(train_info_args, allow_extra_keys=True)
    setup_model_profile()
    dataHelper = NN_DataHelper(model_args, )
    tokenizer: QWenTokenizer
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=QWenTokenizer, config_class_name=QWenConfig)
    config = QWenConfig.from_pretrained(ckpt_dir)
    pl_model = MyTransformer(config=config, model_args=model_args, torch_dtype=torch.float16, )
    return pl_model.get_llm_model()

if __name__ == '__main__':
    ckpt_dir = './best_ckpt/last'  # lora 路径
    output_peft_dir = './peft_lora' # peft 路径
    tokenizer,config,lora_args = convert_to_peft(ckpt_dir, output_peft_dir)

    # 验证peft
    from transformers import AutoModelForCausalLM
    from peft import get_peft_config, get_peft_model, LoraConfig, TaskType,PeftModel



    model = get_base_model(ckpt_dir)
    # model = get_peft_model(model, peft_config)
    model = PeftModel.from_pretrained(model, output_peft_dir)
    model.print_trainable_parameters()

    model.eval().half().cuda()
    text_list = [
        "写一个诗歌，关于冬天",
        "晚上睡不着应该怎么办",
    ]
    generation_config = GenerationConfig(**{
        "chat_format": "chatml",
        "eos_token_id": 151643,
        "max_new_tokens": 512,
        "pad_token_id": 151643,
        # "stop_words_ids": [[151643]],
        "do_sample": True,
        "top_k": 0,
        "top_p": 0.8,
    })
    for input in text_list:
        response, history = model.chat(tokenizer, input, history=[], generation_config=generation_config)
        print("input", input)
        print("response", response)
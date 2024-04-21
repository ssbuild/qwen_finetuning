# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 15:29
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import torch
from deep_training.data_helper import ModelArguments, DataArguments
from transformers import HfArgumentParser, GenerationConfig, AutoConfig
from data_utils import config_args, NN_DataHelper, global_args
from deep_training.zoo.model_zoo.llm.llm_model import MyTransformer, PetlArguments,PetlModel
from data_processer import make_context

if __name__ == '__main__':
    config_args['seed'] = None
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(config_args, allow_extra_keys=True)
    dataHelper = NN_DataHelper(model_args)
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config()

    ckpt_dir = './best_ckpt/last'
    config = AutoConfig.from_pretrained(ckpt_dir)

    lora_args = PetlArguments.from_pretrained(ckpt_dir)

    assert lora_args.inference_mode == True

    # new_num_tokens = config.vocab_size
    # if config.task_specific_params is not None and config.task_specific_params.get('vocab_size', None) is not None:
    #     config.vocab_size = config.task_specific_params['vocab_size']

    pl_model = MyTransformer(config=config, model_args=model_args, lora_args=lora_args,
                             torch_dtype=torch.float16,
                             # new_num_tokens=new_num_tokens,#扩充词
                             
                             # # device_map="auto",
                             # device_map = {"":0} # 第一块卡
                             )
    # 加载多个lora权重
    pl_model.load_sft_weight(ckpt_dir,adapter_name="default")

    # 加载多个lora权重
    #pl_model.load_sft_weight(ckpt_dir, adapter_name="yourname")

    # 加载多个lora权重
    #pl_model.load_sft_weight(ckpt_dir, adapter_name="yourname")


    pl_model.eval().half().cuda()

    # backbone model replaced PetlModel
    lora_model: PetlModel = pl_model.backbone

    text_list = [
        "写一个诗歌，关于冬天",
        "晚上睡不着应该怎么办",
    ]
    generation_config = GenerationConfig(**{
        "chat_format": "chatml",
        "eos_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 512,
        "pad_token_id": tokenizer.eos_token_id,
        #"stop_words_ids": [[151643]],
        "do_sample": True,
        "top_k": 0,
        "top_p": 0.8,
    })
    # 基准模型推理
    with lora_model.disable_adapter():
        for input in text_list:
            #lora_model 调用子对象方法
            response, history = lora_model.chat(tokenizer, input, history=[],generation_config=generation_config )
            print("input", input)
            print("response", response)

    lora_model.set_adapter(adapter_name='default')

    for input in text_list:
        _, input_ids = make_context(tokenizer, input)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.unsqueeze(0)
        response = lora_model.generate(inputs=input_ids.cuda(), )
        outputs = response.tolist()[0][len(input_ids[0]):]
        response = tokenizer.decode(outputs, skip_special_tokens=True)

        print("input", input)
        print("response", response)


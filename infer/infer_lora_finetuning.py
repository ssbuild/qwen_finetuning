# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 15:29
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import torch
from deep_training.data_helper import ModelArguments, DataArguments
from transformers import HfArgumentParser, GenerationConfig,AutoConfig
from data_utils import config_args, NN_DataHelper,global_args
from deep_training.zoo.model_zoo.llm.llm_model import MyTransformer,PetlArguments
from data_processer import make_context


if __name__ == '__main__':
    config_args['seed'] = None
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(config_args, allow_extra_keys=True)
    dataHelper = NN_DataHelper(model_args)
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config()

    weight_dir = '../scripts/best_ckpt'
    lora_weight_dir = os.path.join(weight_dir, "last")

    config = AutoConfig.from_pretrained(weight_dir)
    lora_args = PetlArguments.from_pretrained(lora_weight_dir)

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
    # 加载lora权重
    pl_model.load_sft_weight(lora_weight_dir)

    pl_model.eval().half().cuda()

    enable_merge_weight = False
    if enable_merge_weight:
        # 合并lora 权重 保存
        pl_model.save_sft_weight(os.path.join(lora_weight_dir,'pytorch_model_merge.bin'),merge_lora_weight=True)

    else:
        model = pl_model.get_llm_model()

        text_list = [
            "写一个诗歌，关于冬天",
            "晚上睡不着应该怎么办",
        ]

        model.generation_config = GenerationConfig(**{
            "chat_format": "chatml",
            "eos_token_id": tokenizer.eos_token_id,
            "max_new_tokens": 512,
            "pad_token_id": tokenizer.eos_token_id,
            #"stop_words_ids": [[151643]],
            "do_sample": True,
            "top_k": 0,
            "top_p": 0.8,
        })
        for input in text_list:
            _, input_ids = make_context(tokenizer, input)
            input_ids = torch.tensor(input_ids)
            input_ids = input_ids.unsqueeze(0)
            response = model.generate(inputs = input_ids.cuda(),)
            outputs = response.tolist()[0][len(input_ids[0]):]
            response = tokenizer.decode(outputs, skip_special_tokens=True)

            print("input", input)
            print("response", response)




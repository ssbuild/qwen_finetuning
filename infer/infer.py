# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 15:29
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import torch
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from transformers import HfArgumentParser, BitsAndBytesConfig, GenerationConfig, PreTrainedTokenizer
from data_utils import config_args, NN_DataHelper
from deep_training.zoo.model_zoo.llm.llm_model import MyTransformer,PetlArguments
from data_processer import make_context



if __name__ == '__main__':
    config_args['seed'] = None
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(config_args, allow_extra_keys=True)

    dataHelper = NN_DataHelper(model_args)
    tokenizer: PreTrainedTokenizer
    tokenizer, config, _,_ = dataHelper.load_tokenizer_and_config()

    # quantization configuration for NF4 (4 bits)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # # quantization configuration for Int8 (8 bits)
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)


    pl_model = MyTransformer(config=config, model_args=model_args,
                             torch_dtype=torch.float16,
                             # device_map="cuda:0",
                             # quantization_config=quantization_config,
                             )

    model = pl_model.get_llm_model()

    # if hasattr(model,'is_loaded_in_4bit') or hasattr(model,'is_loaded_in_8bit'):
    #     model.eval().cuda()
    # else:
    #     model.half().eval().cuda()

    model = model.eval()
    model.requires_grad_(False)

    model.half().cuda()
    model = model.eval()

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

    for input in text_list:
        _, input_ids = make_context(tokenizer, input)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.unsqueeze(0)
        response = model.generate(inputs=input_ids.cuda(), )
        outputs = response.tolist()[0][len(input_ids[0]):]
        response = tokenizer.decode(outputs, skip_special_tokens=True)

        print("input", input)
        print("response", response)

    # response, history = base_model.chat(tokenizer, "写一个诗歌，关于冬天", history=[],max_length=30)
    # print('写一个诗歌，关于冬天',' ',response)


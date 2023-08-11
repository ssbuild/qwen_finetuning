# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 15:29
import torch
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from transformers import HfArgumentParser, BitsAndBytesConfig, GenerationConfig
from data_utils import train_info_args, NN_DataHelper
from aigc_zoo.model_zoo.qwen.llm_model import MyTransformer,QWenTokenizer,LoraArguments,setup_model_profile, QWenConfig



if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(train_info_args, allow_extra_keys=True)

    setup_model_profile()

    dataHelper = NN_DataHelper(model_args)
    tokenizer: QWenTokenizer
    tokenizer, config, _,_ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=QWenTokenizer, config_class_name=QWenConfig)

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
    if not model.quantized:
        # 按需修改，目前只支持 4/8 bit 量化 ， 可以保存量化模型
        model.half().quantize(4).cuda()
        # 保存量化权重
        # model.save_pretrained('qwen-chat-7b-int4',max_shard_size="4GB")
        # exit(0)
    else:
        # 已经量化
        model.half().cuda()

    text_list = [
        "写一个诗歌，关于冬天",
        "晚上睡不着应该怎么办",
    ]
    generation_config = GenerationConfig(**{
        "chat_format": "chatml",
        "eos_token_id": 151643,
        "max_new_tokens": 512,
        "pad_token_id": 151643,
        #"stop_words_ids": [[151643]],
        "do_sample": True,
        "top_k": 0,
        "top_p": 0.8,
    })

    for input in text_list:
        response, history = model.chat(tokenizer, input, history=[],generation_config=generation_config )
        print("input", input)
        print("response", response)

    # response, history = base_model.chat(tokenizer, "写一个诗歌，关于冬天", history=[],max_length=30)
    # print('写一个诗歌，关于冬天',' ',response)


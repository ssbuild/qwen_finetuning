# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 15:29
import torch
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from transformers import HfArgumentParser, GenerationConfig
from data_utils import train_info_args, NN_DataHelper, get_deepspeed_config
from aigc_zoo.model_zoo.qwen.llm_model import MyTransformer,QWenTokenizer,setup_model_profile, QWenConfig,LoraArguments

deep_config = get_deepspeed_config()


if __name__ == '__main__':
    train_info_args['seed'] = None
    train_info_args['model_name_or_path'] = None

    parser = HfArgumentParser((ModelArguments,  DataArguments))
    model_args,data_args= parser.parse_dict(train_info_args,allow_extra_keys=True)

    setup_model_profile()

    dataHelper = NN_DataHelper(model_args, None, data_args)
    tokenizer: QWenTokenizer
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=QWenTokenizer, config_class_name=QWenConfig)

    ###################### 注意 选最新权重
    #选择最新的权重 ， 根据时间排序 选最新的
    config = QWenConfig.from_pretrained('./best_ckpt')

    # new_num_tokens = config.vocab_size
    # if config.task_specific_params is not None and config.task_specific_params.get('vocab_size', None) is not None:
    #     config.vocab_size = config.task_specific_params['vocab_size']

    pl_model = MyTransformer(config=config, model_args=model_args,
                             torch_dtype=torch.float16,
                             # new_num_tokens=new_num_tokens,#扩充词
                             )
    if deep_config is None:
        train_weight = './best_ckpt/last-v3.ckpt'
    else:
        #使用转换脚本命令 生成 ./best_ckpt/last/best.pt 权重文件
        # cd best_ckpt/last
        # python zero_to_fp32.py . best.pt
        train_weight = './best_ckpt/last/best.pt'

    #加载微调权重
    pl_model.load_sft_weight(train_weight,strict=True)

    model = pl_model.get_llm_model()
    #保存hf权重
    #config.save_pretrained('convert/')

    # 保存sft p-tuning-v2 权重
    #  pl_model.save_sft_weight('convert/pytorch_model_sft_ptv2.bin')

    #保存sft权重
    # pl_model.save_sft_weight('convert/pytorch_model_sft.bin')


    model.half().cuda()
    model = model.eval()

    text_list = [
        "写一个诗歌，关于冬天",
        "晚上睡不着应该怎么办",
    ]
    generation_config = GenerationConfig(**{
        "chat_format": "chatml",
        "decay_bound": 0.0,
        "decay_factor": 1.0,
        "eos_token_id": 151643,
        "factual_nucleus_sampling": False,
        "max_context_size": 1024,
        "max_generate_size": 512,
        "max_new_tokens": 512,
        "pad_token_id": 151643,
        #"stop_words_ids": [[151643]],
        "do_sample": True,
        "top_k": 0,
        "top_p": 0.8,
    })
    for input in text_list:
        response, history = model.chat(tokenizer, input, history=[],generation_config=generation_config)
        print("input",input)
        print("response", response)


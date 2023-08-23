# -*- coding: utf-8 -*-
# @Time:  23:20
# @Author: tk
# @File：model_maps



train_info_models = {
    'Qwen-7B': {
        'model_type': 'qwen',
        'model_name_or_path': '/data/nlp/pre_models/torch/qwen/Qwen-7B',
        'config_name': '/data/nlp/pre_models/torch/qwen/Qwen-7B/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/qwen/Qwen-7B/',
    },

    'Qwen-7B-Chat': {
        'model_type': 'qwen',
        'model_name_or_path': '/data/nlp/pre_models/torch/qwen/Qwen-7B-Chat',
        'config_name': '/data/nlp/pre_models/torch/qwen/Qwen-7B-Chat/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/qwen/Qwen-7B-Chat/',
    },

    'qwen-7b-chat-int4': {
        'model_type': 'qwen',
        'model_name_or_path': '/data/nlp/pre_models/torch/qwen/qwen-7b-chat-int4',
        'config_name': '/data/nlp/pre_models/torch/qwen/qwen-7b-chat-int4/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/qwen/qwen-7b-chat-int4/',
    },

}

# 'target_modules': ['query_key_value'],  # bloom,gpt_neox
# 'target_modules': ["q_proj", "v_proj"], #llama,opt,gptj,gpt_neo
# 'target_modules': ['c_attn'], #gpt2
# 'target_modules': ['project_q','project_v'] # cpmant

train_target_modules_maps = {
    'qwen': ['c_attn'],
    'moss': ['qkv_proj'],
    'chatglm': ['query_key_value'],
    'chatglm2': ['query_key_value'],
    'bloom' : ['query_key_value'],
    'gpt_neox' : ['query_key_value'],
    'llama' : ["q_proj", "v_proj"],
    'opt' : ["q_proj", "v_proj"],
    'gptj' : ["q_proj", "v_proj"],
    'gpt_neo' : ["q_proj", "v_proj"],
    'gpt2' : ['c_attn'],
    'cpmant' : ['project_q','project_v'],
    'rwkv' : ['key','value','receptance'],
}

# 量化权重不支持此模式训练
train_model_config = train_info_models['Qwen-7B-Chat']
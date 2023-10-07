# -*- coding: utf-8 -*-
# @Time:  23:20
# @Author: tk
# @File：model_maps

from aigc_zoo.constants.define import (TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING)

__all__ = [
    "TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING",
    "train_model_config"
]

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

# 按需修改
# TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING

# 量化权重不支持此模式训练
train_model_config = train_info_models['Qwen-7B-Chat']
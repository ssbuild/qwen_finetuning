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
    "MODELS_MAP"
]

MODELS_MAP = {

    'Qwen-1_8B-Chat': {
        'model_type': 'qwen',
        'model_name_or_path': '/data/nlp/pre_models/torch/qwen/Qwen-1_8B-Chat',
        'config_name': '/data/nlp/pre_models/torch/qwen/Qwen-1_8B-Chat',
        'tokenizer_name': '/data/nlp/pre_models/torch/qwen/Qwen-1_8B-Chat',
    },

    'Qwen-1_8B': {
        'model_type': 'qwen',
        'model_name_or_path': '/data/nlp/pre_models/torch/qwen/Qwen-1_8B',
        'config_name': '/data/nlp/pre_models/torch/qwen/Qwen-1_8B',
        'tokenizer_name': '/data/nlp/pre_models/torch/qwen/Qwen-1_8B',
    },


    'Qwen-7B-Chat': {
        'model_type': 'qwen',
        'model_name_or_path': '/data/nlp/pre_models/torch/qwen/Qwen-7B-Chat',
        'config_name': '/data/nlp/pre_models/torch/qwen/Qwen-7B-Chat',
        'tokenizer_name': '/data/nlp/pre_models/torch/qwen/Qwen-7B-Chat',
    },
    
    'Qwen-7B': {
        'model_type': 'qwen',
        'model_name_or_path': '/data/nlp/pre_models/torch/qwen/Qwen-7B',
        'config_name': '/data/nlp/pre_models/torch/qwen/Qwen-7B',
        'tokenizer_name': '/data/nlp/pre_models/torch/qwen/Qwen-7B',
    },

    'Qwen-14B-Chat': {
        'model_type': 'qwen',
        'model_name_or_path': '/data/nlp/pre_models/torch/qwen/Qwen-14B-Chat',
        'config_name': '/data/nlp/pre_models/torch/qwen/Qwen-14B-Chat',
        'tokenizer_name': '/data/nlp/pre_models/torch/qwen/Qwen-14B-Chat',
    },
    
    'Qwen-14B': {
        'model_type': 'qwen',
        'model_name_or_path': '/data/nlp/pre_models/torch/qwen/Qwen-14B',
        'config_name': '/data/nlp/pre_models/torch/qwen/Qwen-14',
        'tokenizer_name': '/data/nlp/pre_models/torch/qwen/Qwen-14B',
    },

    'Qwen-72B-Chat': {
        'model_type': 'qwen',
        'model_name_or_path': '/data/nlp/pre_models/torch/qwen/Qwen-72B-Chat',
        'config_name': '/data/nlp/pre_models/torch/qwen/Qwen-72B-Chat',
        'tokenizer_name': '/data/nlp/pre_models/torch/qwen/Qwen-72B-Chat',
    },

    'Qwen-72B': {
        'model_type': 'qwen',
        'model_name_or_path': '/data/nlp/pre_models/torch/qwen/Qwen-72B',
        'config_name': '/data/nlp/pre_models/torch/qwen/Qwen-72B',
        'tokenizer_name': '/data/nlp/pre_models/torch/qwen/Qwen-72B',
    },


}

# 按需修改
# TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING


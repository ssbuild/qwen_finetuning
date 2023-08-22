# -*- coding: utf-8 -*-
# @Time    : 2023/4/4 14:46

from fastapi import FastAPI, Request
import uvicorn, json, datetime
import torch

from deep_training.data_helper import ModelArguments, DataArguments
from deep_training.nlp.models.qwen.modeling_qwen import setup_model_profile, QWenConfig
from deep_training.nlp.models.lora.v2 import EffiArguments
from transformers import HfArgumentParser

from data_utils import train_info_args, NN_DataHelper,global_args
from aigc_zoo.model_zoo.qwen.llm_model import MyTransformer, QWenTokenizer

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()


@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_args= json.loads(json_post)
    prompt = json_args.pop('prompt')
    history = json_args.pop('history',None)

    gen_args = {
        "chat_format": "chatml",
        "decay_bound": 0.0,
        "decay_factor": 1.0,
        "eos_token_id": 151643,
        "factual_nucleus_sampling": False,
        "max_context_size": 1024,
        "max_generate_size": 512,
        "max_new_tokens": 512,
        "pad_token_id": 151643,
        # "stop_words_ids": [[151643]],
        "do_sample": True,
        "top_k": 0,
        "top_p": 0.8,
    }
    gen_args.update(json_args)
    generation_config = GenerationConfig(**gen_args)

    response, history = model.chat(tokenizer, prompt, history=[],generation_config=generation_config)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer


if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments, ))
    (model_args,)  = parser.parse_dict(train_info_args,allow_extra_keys=True)

    setup_model_profile()

    dataHelper = NN_DataHelper(model_args, None, data_args)
    tokenizer: QWenTokenizer
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=QWenTokenizer, config_class_name=QWenConfig)

    ckpt_dir = './best_ckpt/last'
    config = QWenConfig.from_pretrained(ckpt_dir)
    

    lora_args = EffiArguments.from_pretrained(ckpt_dir)

    assert lora_args.inference_mode == True

    # new_num_tokens = config.vocab_size
    # if config.task_specific_params is not None and config.task_specific_params.get('vocab_size', None) is not None:
    #     config.vocab_size = config.task_specific_params['vocab_size']

    pl_model = MyTransformer(config=config, model_args=model_args, lora_args=lora_args,
                             torch_dtype=torch.float16,
                             # new_num_tokens=new_num_tokens, # 扩充词
                             # load_in_8bit=global_args["load_in_8bit"],
                             # # device_map="auto",
                             # device_map={"": 0},  # 第一块卡
                             )
    # 加载lora权重
    pl_model.load_sft_weight(ckpt_dir)

    model = pl_model.get_llm_model()
    # 按需修改
    model.half().cuda()
    model = model.eval()

    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
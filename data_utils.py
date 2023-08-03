# @Time    : 2023/1/22 16:22
# @Author  : tk
# @FileName: data_utils.py
import copy
import json
import os
import typing
import numpy as np
import torch
from aigc_zoo.model_zoo.qwen.qwen_generation_utils import get_ltor_masks_and_position_ids
from deep_training.data_helper import DataHelper, ModelArguments, TrainingArguments, DataArguments
from fastdatasets.record import load_dataset as Loader, RECORD, WriterObject, gfile
from tqdm import tqdm
from transformers import HfArgumentParser
from data_processer import DataStrategy, TokenSiding, TokenTruncation
from aigc_zoo.model_zoo.qwen.llm_model import QWenTokenizer,LoraArguments,QWenConfig
from config import *
data_conf = {
   'strategy': DataStrategy.truncation, # 数据策略选项
    DataStrategy.truncation: {
        'ensure_answer_min_length': 1,
        'sup': True, # 是否监督训练
    },
    DataStrategy.siding: {
        'sliding_size': train_info_args['max_seq_length'] // 3 * 2, #prompt滑动窗口大小
        'sup': True, # 是否监督训练
    },

}


def preprocess(text):
  #text = text.replace("\n", "\\n").replace("\t", "\\t")
  return text

def postprocess(text):
  # return text.replace("\\n", "\n").replace("\\t", "\t")
  return text



class NN_DataHelper(DataHelper):
    index = 1
    def on_data_ready(self):
        self.index = -1

    # 切分词
    def on_data_process(self, data: typing.Any, mode: str):
        self.index += 1
        paragraph = data



        max_seq_length = self.max_seq_length_dict[mode]
        tokenizer: QWenTokenizer = self.tokenizer # noqa
        config: QWenConfig = self.config # noqa

        strategy = data_conf['strategy']
        if strategy == DataStrategy.truncation:
            ds = TokenTruncation.process(tokenizer,config,paragraph, max_seq_length,**data_conf[strategy])
        elif strategy == DataStrategy.siding:
            ds = TokenSiding.process(tokenizer,config,paragraph, max_seq_length, **data_conf[strategy])
        else:
            raise ValueError('Invlid strategy',strategy)

        if not ds:
            return None

        if self.index < 3:
            print(ds[0])
        return ds

    # {
    #     "id": 0, "paragraph": [
    #     # 一轮会话
    #     {
    #         "q": "从南京到上海的路线",
    #         "a": [
    #             "你好，南京到上海的路线如下：",
    #             "1. 南京到上海，可以乘坐南京地铁1号线，在南京站乘坐轨道交通1号线。",
    #             "2. 南京到浦东机场，可以搭乘上海地铁1号，在陆家嘴站乘坐地铁1线，在浦东国际机场站乘坐机场快线，前往上海浦东国际机场。",
    #             "3. 上海到南京，可以换乘上海地铁2号线，从南京站换乘地铁2线，再从南京南站换乘地铁1路，然后到达上海站"
    #         ]
    #     }
    #     # 二轮....
    # ]
    # }

    # 读取文件
    def on_get_corpus(self, files: typing.List, mode: str):
        D = []
        for file in files:
            with open(file, mode='r', encoding='utf-8', newline='\n') as f:
                lines = f.readlines()

            for line_id, line in enumerate(lines):
                jd = json.loads(line)
                if not jd:
                    continue
                paragraph = jd['paragraph']
                if line_id < 10:
                    print(paragraph)

                paragraph = [(preprocess(session['q']),
                              preprocess('\n'.join(session['a'])) if isinstance(session['a'],list) else preprocess(session['a']))
                    for session in paragraph]

                D.append(paragraph)
        return D

    def collate_fn(self,batch):
        if not hasattr(self,'sptoken'):
            self.sptoken = self.tokenizer.encode(text="")[-2:]

        o = {}
        for i, b in enumerate(batch):
            if i == 0:
                for k in b:
                    o[k] = [torch.tensor(b[k])]
            else:
                for k in b:
                    o[k].append(torch.tensor(b[k]))
        for k in o:
            o[k] = torch.stack(o[k])

        seqlens = o.pop('seqlen')
        max_len = torch.max(seqlens).tolist()
        input_ids = o['input_ids'][:, :max_len]

        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            input_ids,
            eod_token=self.config.eos_token_id,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
        )

        o['input_ids'] = input_ids.long()
        o['attention_mask'] = attention_mask.bool()
        o['position_ids'] = position_ids.long()
        o['labels'] = o['labels'][:, :max_len].long()
        return o

    def make_dataset_all(self):
        data_args = self.data_args

        # schema for arrow parquet
        schema = {
            "input_ids": "int32_list",
            "labels": "int32_list",
            "seqlen": "int32_list",
        }
        # 缓存数据集
        if data_args.do_train:
            self.make_dataset_with_args(data_args.train_file, mixed_data=False, shuffle=True,
                                              mode='train',schema=schema)
        if data_args.do_eval:
            self.make_dataset_with_args(data_args.eval_file, mode='eval',schema=schema)
        if data_args.do_test:
            self.make_dataset_with_args(data_args.test_file, mode='test',schema=schema)

if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments))
    model_args, training_args, data_args, lora_args = parser.parse_dict(train_info_args)
    lora_args = lora_args.config

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, _,_ = dataHelper.load_tokenizer_and_config(tokenizer_class_name=QWenTokenizer,config_class_name=QWenConfig)
    



    # 缓存数据集
    # 检测是否存在 output/dataset_0-train.record ，不存在则制作数据集
    dataHelper.make_dataset_all()


    # def shuffle_records(record_filenames, outfile, compression_type='GZIP'):
    #     print('shuffle_records record...')
    #     options = RECORD.TFRecordOptions(compression_type=compression_type)
    #     dataset_reader = Loader.RandomDataset(record_filenames, options=options, with_share_memory=True)
    #     data_size = len(dataset_reader)
    #     all_example = []
    #     for i in tqdm(range(data_size), desc='load records'):
    #         serialized = dataset_reader[i]
    #         all_example.append(serialized)
    #     dataset_reader.close()
    #
    #     shuffle_idx = list(range(data_size))
    #     random.shuffle(shuffle_idx)
    #     writer = WriterObject(outfile, options=options)
    #     for i in tqdm(shuffle_idx, desc='shuffle record'):
    #         example = all_example[i]
    #         writer.write(example)
    #     writer.close()
    #
    #
    # # 对每个record 再次打乱
    # for filename in dataHelper.train_files:
    #     shuffle_records(filename, filename)

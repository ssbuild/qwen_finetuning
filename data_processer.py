# @Time    : 2023/3/25 18:36
# @Author  : tk
import copy
import json
import random
from distutils.command.config import config
from enum import Enum
from typing import Tuple, List

import numpy as np
# from deep_training.zoo.model_zoo.llm.qwen_generation_utils import make_context
from transformers import PreTrainedTokenizer


class DataStrategy(Enum):
    truncation = 1
    siding = 2


def make_context(
    tokenizer: PreTrainedTokenizer,
    query: str,
    history: List[Tuple[str, str]] = None,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = tokenizer.encode(im_start, add_special_tokens=False)
        im_end_tokens = tokenizer.encode(im_end, add_special_tokens=False)
        nl_tokens = tokenizer.encode("\n", add_special_tokens=False)

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(
                role
            ) + nl_tokens + tokenizer.encode(content)

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            response_text, response_tokens_part = _tokenize_str(
                "assistant", turn_response
            )
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            prev_chat = (
                f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            )

            current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens




class TokenIdsMaker:

    @classmethod
    def final(cls, tokenizer,config, input_ids, labels, max_seq_length):
        seqlen = np.asarray(len(input_ids), dtype=np.int32)
        pad_len = max_seq_length - seqlen
        input_ids = np.asarray(input_ids, dtype=np.int32)
        labels = np.asarray(labels, dtype=np.int32)
        if pad_len:
            pad_val = tokenizer.eos_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            labels = np.pad(labels, (0, pad_len), 'constant', constant_values=(-100, -100))
        d = {
            'input_ids': input_ids,
            'labels': labels,
            'seqlen': seqlen
        }
        return d

    @classmethod
    def tunction(cls, tokenizer: PreTrainedTokenizer,config, paragraph, max_seq_length, sup=True):
        sptoken = []
        ds = []
        prefix = None
        history = []
        for sid,(role,q,a) in enumerate(paragraph):
            if role == 'system':
                prefix = q
                continue
            # 从兼容性考虑，预处理从数据源构建
            # if tools is not None:
            #     q = ToolsBuilder.build(tools,query=q)
            #
            # if role in ['observation','Observation']:
            #     q = f'Observation: {q}'

            history += [(q,a)]
            _,a_ids = make_context(tokenizer=tokenizer,query=q,history=history[:-1],
                                   system = prefix or "You are a helpful assistant." ,
                                   max_window_size = 6144,
                                   chat_format = "chatml",)
            b_ids = tokenizer.encode(a,add_special_tokens=False)

            while len(a_ids) + len(b_ids) > max_seq_length - len(sptoken) - 1:
                if len(b_ids) > len(a_ids):
                    b_ids.pop(-1)
                else:
                    a_ids.pop(0)
            b_ids += [ tokenizer.eos_token_id ]
            input_ids = a_ids + b_ids
            labels = copy.deepcopy(input_ids) if not sup else [ -100 ] * len(a_ids) + copy.deepcopy(b_ids)
            input_ids = sptoken + input_ids
            labels = sptoken + labels if not sup else [ -100 ] * len(sptoken) + labels
            assert len(input_ids) <= max_seq_length
            ds.append(cls.final(tokenizer,config, input_ids, labels, max_seq_length))
        return ds


    @classmethod
    def slidding(cls, tokenizer: PreTrainedTokenizer,config, paragraph, max_seq_length, sliding_size = None,src_max_length=None,dst_max_length=None,sup=True):
        if sliding_size is None:
            sliding_size = max_seq_length
        ds = []
        sptoken = []
        prefix = None
        history = []
        for sid, (role, q, a) in enumerate(paragraph):
            if role == 'system':
                prefix = q
                continue

            # 从兼容性考虑，预处理从数据源构建
            # if tools is not None:
            #     q = ToolsBuilder.build(tools, query=q)

            # if role in ['observation', 'Observation']:
            #     q = f'Observation: {q}'

            history += [(q, a)]
            _, a_ids = make_context(tokenizer=tokenizer, query=q, history=history[:-1],
                                    system=prefix or "You are a helpful assistant.",
                                    max_window_size=6144,
                                    chat_format="chatml", )
            b_ids = tokenizer.encode(a,add_special_tokens=False)

            if src_max_length and src_max_length > 0:
                a_ids = a_ids[ :src_max_length ]
            if dst_max_length and dst_max_length > 0:
                b_ids = b_ids[ :dst_max_length ]

            input_ids_qa = a_ids + b_ids + [tokenizer.eos_token_id]
            if sup:
                labels_all = [-100] * len(a_ids) + b_ids
            else:
                labels_all = copy.deepcopy(input_ids_qa)

            pos = 0
            while pos < len(input_ids_qa):
                input_ids = input_ids_qa[pos:pos + max_seq_length - len(sptoken)]
                labels = labels_all[pos:pos + max_seq_length - len(sptoken)]

                pos += sliding_size
                if np.all(np.asarray(labels) == -100):
                    continue
                input_ids = sptoken + input_ids
                labels = sptoken + labels if not sup else [ -100 ] * len(sptoken) + labels
                ds.append(cls.final(tokenizer, config,input_ids, labels, max_seq_length))
        return ds

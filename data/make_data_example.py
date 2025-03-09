# -*- coding: utf-8 -*-
# @Time    : 2023/2/24 12:50
import copy
import json


template =[
  {
    "id": "identity_0",
    "conversations": [
      {
        "from": "user",
        "value": "你好"
      },
      {
        "from": "assistant",
        "value": "我是Qwen-VL,一个支持视觉输入的大模型。"
      }
    ]
  },
  {
    "id": "identity_1",
    "conversations": [
      {
        "from": "user",
        "value": "Picture 1: <img>../assets/demo.jpeg</img>\n图中的狗是什么品种？"
      },
      {
        "from": "assistant",
        "value": "图中是一只拉布拉多犬。"
      },
      {
        "from": "user",
        "value": "框出图中的格子衬衫"
      },
      {
        "from": "assistant",
        "value": "<ref>格子衬衫</ref><box>(588,499),(725,789)</box>"
      }
    ]
  },
  {
    "id": "identity_2",
    "conversations": [
      {
        "from": "user",
        "value": "Picture 1: <img>../assets/Chongqing.jpeg</img>\nPicture 2: <img>../assets/Beijing.jpeg</img>\n图中都是哪"
      },
      {
        "from": "assistant",
        "value": "第一张图片是重庆的城市天际线，第二张图片是北京的天际线。"
      }
    ]
  }
]

x = []
for i,item in enumerate(template):
  d = {
    "id": i
  }
  d["paragraph"] = []
  one = {}
  for j,conversation in enumerate(item["conversations"]):
    if j % 2 == 0:
      one["q"] = conversation["value"]
    else:
      one["a"] = conversation["value"]
      d["paragraph"].append(copy.deepcopy(one))
      one.clear()

  x.append(d)





with open('./finetune_train_examples.json',mode='w',encoding='utf-8',newline='\n') as f:
    index = 0
    for i in range(100):
        for j in range(len(x)):
            index += 1
            x[j]['id'] = index
            f.write(json.dumps(x[j],ensure_ascii=False) + '\n' )

from collections import Counter
import random
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import json

with open("关系.txt",encoding='utf-8',mode='r') as f:
    content=[i.split('\t') for i in f.read().split("\n") if len(i)>0]

content_s1 = []
content_w1 = []
content_m1 = []
content_s = []
content_w = []
content_m = []

with open("relation2id.txt",encoding='utf-8',mode='r') as f:
    content_re=[i.split('\t') for i in f.read().split("\n") if len(i)>0]

dd_ict = {}
for i in content_re:
    strr = i[0].split(' ')[0]
    dd_ict[strr] = 1

for text in content:
    str = text[0].split("@")
    if len(str) != 4:
        continue
    if str[2] not in dd_ict:
        continue
    vec = []
    vecc = []
    vec.append(str[0])
    vec.append("位于")
    vec.append(str[1])
    vecc.append(vec)
    content_w1.append('(' + str[0] + ',' + "位于" + ',' + str[1] + ')')
    content_s1.append(vecc)
    content_m1.append(str[3])


num_i = 0
for text_index in range(len(content_m1)):
    ind = 0
    for index in range(len(content_m1)):
        if text_index != index and content_m1[text_index] == content_m1[index]:
            if index < text_index:
                break
            ind = 1
    if ind == 1:
        content_w.append(content_w1[text_index])
        content_s.append(content_s1[text_index])
        content_m.append(content_m1[text_index])
        for index in range(len(content_m1)):
            if text_index != index and content_m1[text_index] == content_m1[index]:
                content_w[num_i] += ',' + content_w1[index]
                content_s[num_i].append(content_s1[index][0])

        num_i += 1

for i in range(len(content_m)):
    dict = {"id": 10001, "cate": "建筑", "instruction": "已知候选的关系列表：['事件', '位于', '名称由来']，请你根据关系列表，从以下输入中抽取出可能存在的头实体(Subject)与尾实体(Object)，并给出对应的关系三元组。请按照 (Subject,Relation,Object) 的格式回答。", "input": "浅草神社位于日本东京都台东区浅草的浅草寺本堂东侧，供奉的是土师真中知、桧田浜成、桧前武成，三位对于浅草寺创立有密切关联的人，每年5月17日都会举行三社祭。现在被指定为重要文化财产。", "output": "(浅草神社,事件,三社祭),(浅草神社,位于,浅草),(台东区,位于,东京都),(浅草寺,位于,浅草),(浅草寺,名称由来,浅草)", "kg": [["浅草神社", "事件", "三社祭"], ["浅草神社", "位于", "浅草"], ["台东区", "位于", "东京都"], ["浅草寺", "位于", "浅草"], ["浅草寺", "名称由来", "浅草"]]}
    dict["input"] = content_m[i]
    dict["output"] = content_w[i]
    dict["kg"] = content_s[i]
    with open('train1.json', 'a', encoding='utf-8') as json_file:
        js = json.dumps(dict, ensure_ascii=False)
        json_file.write(js)
        json_file.write('\n')



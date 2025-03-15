import datetime
import tkinter
from  tkinter import *
import json
import random
import numpy as np
import torch

lsit = []
fp = r"wds.txt"

with open(fp, "r", encoding='UTF-8') as f:
    all_line_contents: list = f.readlines()
    for i in all_line_contents:
        if i:
            i = i.replace("\n", '')
        lsit.append(i)

vec1_s = []
vec2_s = []
sentence_s = []
relat_s = []
for sentence in lsit:
    k = 0
    s = ""
    for sen_ten in sentence:
        if sen_ten == '@':
            if k == 0:
                vec1_s.append(s)
            if k == 1:
                vec2_s.append(s)
            if k == 2:
                relat_s.append(s)
            k+=1
            s = ""
            continue
        s += sen_ten
    sentence_s.append(s)

wordtoindex = {}
for index in range(len(vec1_s)):
    if vec1_s[index] not in wordtoindex:
        wordtoindex[vec1_s[index]] = []
        wordtoindex[vec1_s[index]].append(vec2_s[index])
    else:
        wordtoindex[vec1_s[index]].append(vec2_s[index])
    if vec2_s[index] not in wordtoindex:
        wordtoindex[vec2_s[index]] = []
        wordtoindex[vec2_s[index]].append(vec1_s[index])
    else:
        wordtoindex[vec2_s[index]].append(vec1_s[index])

with open('taget_main.json', 'w') as json_file:
    js = json.dumps(wordtoindex)
    json_file.write(js)

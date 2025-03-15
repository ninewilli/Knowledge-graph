import numpy as np
import json
with open("关系.txt",encoding='utf-8',mode='r') as f:
    content=[i.split('\t') for i in f.read().split("\n") if len(i)>0]

for index in range(len(content)):
    content[index][0] = content[index][0].split('@')

fp = open("train.txt","a",encoding='utf-8')

with open("relation2id.txt",encoding='utf-8',mode='r') as f:
    content_re=[i.split('\t') for i in f.read().split("\n") if len(i)>0]

dd_ict = {}
for i in content_re:
    strr = i[0].split(' ')[0]
    dd_ict[strr] = 1

for i in range(len(content)):
    if len(content[i][0]) != 4:
        continue
    if content[i][0][2] not in dd_ict:
        continue
    fp.write(content[i][0][0])
    fp.write("	")
    fp.write(content[i][0][1])
    fp.write("	")
    fp.write(content[i][0][2])
    fp.write("	")
    fp.write(content[i][0][3])
    fp.write('\n')
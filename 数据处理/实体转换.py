import numpy as np
import json
with open("关系.txt",encoding='utf-8',mode='r') as f:
    content=[i.split('\t') for i in f.read().split("\n") if len(i)>0]

for index in range(len(content)):
    content[index][0] = content[index][0].split('@')

word_idx = []
with open("taget_main.json",encoding='utf-8') as inputData:
    for line in inputData:
        try:
            word_idx.append(json.loads(line.rstrip(';\n')))
        except ValueError:
            print ("Skipping invalid line {0}".format(repr(line)))
dict = word_idx[0]
with open("relation2id.txt",encoding='utf-8',mode='r') as f:
    content_re=[i.split('\t') for i in f.read().split("\n") if len(i)>0]

dd_ict = {}
for i in content_re:
    strr = i[0].split(' ')[0]
    dd_ict[strr] = 1
for i in range(len(content)):
    if len(content[i][0][0]) != 4:
        continue
    if content[i][0][2] not in dd_ict:
        continue
    if content[i][0][0] not in dict:
        dict[content[i][0][0]] = []
        dict[content[i][0][0]].append(content[i][0][1])
    else:
        uu = 0
        for text_i in dict[content[i][0][0]]:
            if text_i == content[i][0][1]:
                uu = 1
                break
        if uu == 0:
            dict[content[i][0][0]].append(content[i][0][1])
    if content[i][0][1] not in dict:
        dict[content[i][0][1]] = []
        dict[content[i][0][1]].append(content[i][0][0])
    else:
        uu = 0
        for text_i in dict[content[i][0][1]]:
            if text_i == content[i][0][0]:
                uu = 1
                break
        if uu == 0:
            dict[content[i][0][1]].append(content[i][0][0])

with open('taget_main.json', 'w', encoding='utf-8') as json_file:
    js = json.dumps(dict, ensure_ascii=False)
    json_file.write(js)
    json_file.write('\n')
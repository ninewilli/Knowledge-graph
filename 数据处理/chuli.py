

lsit = []
fp = r"地点关系.txt"

with open(fp, "r", encoding='UTF-8') as f:
    all_line_contents: list = f.readlines()
    for i in all_line_contents:
        if i:
            i = i.replace("\n", '')
        lsit.append(i)

ans_num = []
for sentence_num in lsit:
    s = ""
    for sentence in sentence_num:
        if sentence == '@':
            s+='	'
        else:
            s+=sentence
    ans_num.append(s)


with open("wenjian.txt","w") as f:
    for i in ans_num:
        f.write(i)
        f.write('\n')


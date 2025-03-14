# Standard library imports
import codecs
import collections
import datetime
import json
import math
import os
import random
import sys
import threading
import time
from collections import deque
from itertools import chain

# Third-party imports
import numpy as np
import pandas as pd
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AdamW, AutoModelForTokenClassification, AutoTokenizer,
                         BertConfig, BertForTokenClassification, BertModel,
                         BertTokenizer)

# GUI imports
import tkinter
from tkinter import *
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

# Local imports
import part.py as py
from part.BiLSTM_ATT import BiLSTM_ATT
from part.LSTM import LSTM_CRF, prepare_sequence
import part.bertBILSTM as bertBILSTM


window = ttk.Window()
style = ttk.Style(theme='minty')
theme_names = style.theme_names()

style.configure('my.TButton', font=('华文新魏', 10))


relation2id = {}
with codecs.open('model/relation2id.txt','r','utf-8') as input_data:
    for line in input_data.readlines():
        relation2id[line.split()[0]] = int(line.split()[1])
    input_data.close()
    #print relation2id
model_r = torch.load('model/model_epoch.pkl')
id2relation = {}
for text in relation2id:
    id2relation[relation2id[text]] = text
datas = deque()
labels = deque()
positionE1 = deque()
positionE2 = deque()
count = []
for i in range(131):
    count.append(0)
total_data=0
ad = [' ','\u3000', '\n', '。', '？', '！', '，', '；', '：', '、', '《', '》', '“', '”', '‘', '’', '［', '］', '....', '......',
          '『', '』', '（', '）', '…', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b','1','2','3','4','5','6','7','8','9','0','-']

data_re = {}
data_res = {}
with codecs.open('model/train.txt','r','utf-8') as tfc:
    for lines in tfc:
        line = lines.split()
        if count[relation2id[line[2]]] <1500:
            sentence = []
            index1 = line[3].index(line[0])
            position1 = []
            index2 = line[3].index(line[1])
            position2 = []
            for i,word in enumerate(line[3]):
                sentence.append(word)
                position1.append(i-index1)
                position2.append(i-index2)
                i+=1
            datas.append(sentence)
            labels.append(relation2id[line[2]])
            positionE1.append(position1)
            positionE2.append(position2)
        count[relation2id[line[2]]]+=1
        total_data+=1

collections.Iterable = collections.abc.Iterable
def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


all_words = flatten(datas)
sr_allwords = pd.Series(all_words)
sr_allwords = sr_allwords.value_counts()

set_words = sr_allwords.index
set_ids = range(1, len(set_words)+1)
word2id = pd.Series(set_ids, index=set_words)
id2word = pd.Series(set_words, index=set_ids)

word2id["BLANK"]=len(word2id)+1
word2id["UNKNOW"]=len(word2id)+1
id2word[len(id2word)+1]="BLANK"
id2word[len(id2word)+1]="UNKNOW"
#print "word2id",id2word

def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


all_words = flatten(datas)
sr_allwords = pd.Series(all_words)
sr_allwords = sr_allwords.value_counts()

set_words = sr_allwords.index
set_ids = range(1, len(set_words) + 1)
word2id = pd.Series(set_ids, index=set_words)
id2word = pd.Series(set_words, index=set_ids)

word2id["BLANK"] = len(word2id) + 1
word2id["UNKNOW"] = len(word2id) + 1
id2word[len(id2word) + 1] = "BLANK"
id2word[len(id2word) + 1] = "UNKNOW"

max_len = 50

def X_padding(words):
    ids = []
    for i in words:
        if i in word2id:
            ids.append(word2id[i])
        else:
            ids.append(word2id["UNKNOW"])
    if len(ids) >= max_len:
        return ids[:max_len]
    ids.extend([word2id["BLANK"]] * (max_len - len(ids)))

    return ids


def pos(num):
    if num < -40:
        return 0
    if num >= -40 and num <= 40:
        return num + 40
    if num > 40:
        return 80


def position_padding(words):
    words = [pos(i) for i in words]
    if len(words) >= max_len:
        return words[:max_len]
    words.extend([81] * (max_len - len(words)))
    return words

model = torch.load('model/model_epoch1.pkl',map_location='cpu')
dating = ""

Data = []
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

word_idx = []
relation_data = []
with open("part/testt.json",encoding='utf-8') as inputData:
    for line in inputData:
        try:
            word_idx.append(json.loads(line.rstrip(';\n')))
        except ValueError:
            print ("Skipping invalid line {0}".format(repr(line)))
word_to_ix = word_idx[0]

with open("part/taget_main.json",encoding='utf-8') as inputData:
    for line in inputData:
        try:
            Data.append(json.loads(line.rstrip(';\n')))
        except ValueError:
            print ("Skipping invalid line {0}".format(repr(line)))
data = Data[0]

with open("part/target_relation.json",encoding='utf-8') as inputData:
    for line in inputData:
        try:
            relation_data.append(json.loads(line.rstrip(';\n')))
        except ValueError:
            print ("Skipping invalid line {0}".format(repr(line)))
relation_data = relation_data[0]

def tianchong1():
    canvas = tkinter.Canvas(window, width=1500, height=760, bg='white')
    canvas.pack()
    canvas.place(relx=0.155, rely=0.15)

def tianchong():
    num_index = 0
    y_num_index = 0
    y_index_k = 0
    canvas = tkinter.Canvas(window, width=1500, height=820, bg='white')
    canvas.pack()
    canvas.place(relx=0.155, rely=0.15)
    # while (True):
    #     num_index += 0.1
    #     if num_index >= 0.8:
    #         y_num_index += 0.1
    #         y_index_k += 1
    #         num_index = 0.1
    #     if y_index_k > 6:
    #         break
    #     button_t = tkinter.Button(window, text="        ", font=('华文新魏', 15), fg='white', width=19, height=10, bd=0)
    #     button_t.place(relx=0.126 + num_index, rely=0.04 + y_num_index, anchor='ne')

def panduan():
    tianchong()
    k = Label(window,text='输入句子',font='华文新魏')
    e = tkinter.Entry(window,width=115)
    e.pack()
    k.pack()
    k.place(relx=0.25, rely=0.05)
    e.place(relx=0.25, rely=0.1)
    def insert_point():
        var = e.get()
        sentence_in = prepare_sequence(var, word_to_ix).to(device)
        sentence = model(sentence_in)[1]
        s = ""
        s_list = []
        for i in range(len(var)):
            if sentence[i] == 0 or sentence[i] == 1:
                s+=var[i]
            else:
                if s!="":
                    s_list.append(s)
                    s = ""
        NUM = 0
        for num_s in s_list:
            if len(num_s) == 1:
                continue
            if NUM%2 == 0:
                t1.insert("insert", num_s+'\n')
            else:
                t2.insert("insert", num_s+'\n')
            NUM += 1

    def insert_end():
        var = e.get()
        e.delete(0,END)
        t1.delete("1.0", "end")
        t2.delete("1.0", "end")

    b = tkinter.Button(window, text="关系抽取", width=15, height=2,font='华文新魏',command=insert_point)
    b.pack()
    b.place(relx=0.25, rely=0.15)
    b2 = tkinter.Button(window, text="重新输入", width=15, height=2,font='华文新魏',command=insert_end)
    b2.pack()
    b2.place(relx=0.25, rely=0.24)
    t1 = tkinter.Text(window, height=8)
    k1 = Label(window,text='主体',font='华文新魏')
    t1.pack()
    k1.pack()
    k1.place(relx=0.25, rely=0.33)
    t1.place(relx=0.25, rely=0.38)
    t2 = tkinter.Text(window, height=8)
    k2 = Label(window,text='客体',font='华文新魏')
    t2.pack()
    k2.pack()
    t2.place(relx=0.25, rely=0.63)
    k2.place(relx=0.25, rely=0.58)

def tianjia():
    tianchong()
    k1 = Label(window,text='输入主体',font='华文新魏')
    e1 = tkinter.Entry(window,width=55)
    e1.pack()
    k1.pack()
    k2 = Label(window,text='输入客体',font='华文新魏')
    e2 = tkinter.Entry(window,width=55)
    e2.pack()
    k2.pack()
    k3 = Label(window,text='输入关系',font='华文新魏')
    e3 = tkinter.Entry(window,width=55)
    e3.pack()
    k3.pack()
    e1.place(relx=0.35, rely=0.1)
    e2.place(relx=0.35, rely=0.2)
    e3.place(relx=0.35, rely=0.3)
    k1.place(relx=0.35, rely=0.05)
    k2.place(relx=0.35, rely=0.15)
    k3.place(relx=0.35, rely=0.25)
    def insert_related():
        test_main = e1.get()
        test_obj = e2.get()
        test_relat = e3.get()
        if test_main not in data:
            data[test_main] = []
            data[test_main].append(test_obj)
        else:
            data[test_main].append(test_obj)
        if test_main not in relation_data:
            vec_re = {}
            vec_re[test_obj] = test_relat
            relation_data[test_main] = vec_re
        else:
            relation_data[test_main][test_obj] = test_relat
        with open('taget_main.json', 'w',encoding='utf-8') as json_file:
            js = json.dumps(data, ensure_ascii=False)
            json_file.write(js)
    b = tkinter.Button(window, text="添加关系", width=15, height=2,font='华文新魏',command=insert_related)
    b.pack()
    b.place(relx=0.45, rely=0.4)

y_index = 0
num_nums = 0

def mulu():
    global y_index
    global num_nums
    y_index = 0
    num_nums = 0
    tianchong()

    def create_knowledge_graph(window, data, relation_data, selected_node):
        """
        Creates an optimized, aesthetically pleasing knowledge graph visualization

        Args:
            window: The parent window
            data: Dictionary of node data
            relation_data: Dictionary of relationship data
            selected_node: The selected central node
        """
        import tkinter as tk
        from tkinter import ttk
        import numpy as np
        from matplotlib import cm
        from math import sqrt, pi

        # Clear previous content
        for widget in window.winfo_children():
            if isinstance(widget, ttk.Frame) and widget.winfo_name() == "graph_frame":
                widget.destroy()

        # Create styled frame
        frame = ttk.Frame(window, width=1500, height=820, name="graph_frame")
        frame.place(relx=0.15, rely=0.15, relwidth=0.9, relheight=0.9)

        # Create canvas with modern gradient background
        canvas = tk.Canvas(frame, width=1500, height=820, highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        # Draw gradient background
        for i in range(820):
            gradient_ratio = i / 820
            color = '#{:02x}{:02x}{:02x}'.format(
                int(245 + 10 * gradient_ratio),
                int(247 + 8 * gradient_ratio),
                int(250 - 10 * gradient_ratio)
            )
            canvas.create_line(0, i, 1500, i, fill=color)

        # Modern color scheme
        colors = {
            'primary': '#4A90E2',
            'secondary': '#7ED321',
            'accent': '#FF6B6B',
            'text': '#2C3E50',
            'background': '#F8F9FA',
            'relation': '#A389D4',
            'shadow': '#000000'
        }

        # Improved node styling parameters
        NODE_CONFIG = {
            'central_radius': 48,
            'node_radius': 36,
            'relation_radius': 28,
            'font': ('Segoe UI', 14),
            'line_width': 3
        }

        # Create dynamic layout positions
        center_x, center_y = 750, 410
        node_positions = {}

        # Draw central node with modern effects
        def create_central_node():
            # Glow effect
            for i in range(1, 6):
                glow_size = NODE_CONFIG['central_radius'] + i * 6
                canvas.create_oval(
                    center_x - glow_size, center_y - glow_size,
                    center_x + glow_size, center_y + glow_size,
                    outline=f'#FF6B6B',
                    width=0,
                    stipple='gray50'
                )

            # Main node
            canvas.create_oval(
                center_x - NODE_CONFIG['central_radius'],
                center_y - NODE_CONFIG['central_radius'],
                center_x + NODE_CONFIG['central_radius'],
                center_y + NODE_CONFIG['central_radius'],
                fill=colors['accent'],
                outline=''
            )

            # Inner highlight
            canvas.create_oval(
                center_x - NODE_CONFIG['central_radius'] + 8,
                center_y - NODE_CONFIG['central_radius'] + 8,
                center_x + NODE_CONFIG['central_radius'] - 8,
                center_y + NODE_CONFIG['central_radius'] - 8,
                outline='#000000',
                width=2
            )

            # Node text
            canvas.create_text(
                center_x, center_y,
                text=selected_node,
                fill='white',
                font=('Segoe UI', 16, 'bold'),
                tags=('central_node',)
            )

        create_central_node()

        # Generate category colors using plasma colormap
        node_categories = {node.split(' ')[0] if ' ' in node else node for node in data[selected_node]}
        category_colors = {}
        plasma = cm.get_cmap('plasma', len(node_categories))

        for i, category in enumerate(node_categories):
            rgb = plasma(i / len(node_categories))[:3]
            category_colors[category] = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            )

        # Optimized layout using Fibonacci sphere algorithm
        node_count = len(data[selected_node])
        golden_angle = pi * (3 - sqrt(5))

        def calculate_position(index):
            angle = index * golden_angle
            radius = 240 + 120 * (index % 2)  # Dynamic radius based on node index
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            return x, y

        # Create connections with bezier curves
        for i, node_label in enumerate(data[selected_node]):
            node_x, node_y = calculate_position(i)
            node_positions[node_label] = (node_x, node_y)

            # Get category color
            category = node_label.split(' ')[0] if ' ' in node_label else node_label
            line_color = category_colors.get(category, colors['secondary'])

            # Create smooth bezier curve
            control_offset = 80  # Increased curvature for better visual flow
            control_points = [
                center_x + (node_x - center_x) * 0.3,
                center_y + (node_y - center_y) * 0.3,
                center_x + (node_x - center_x) * 0.7,
                center_y + (node_y - center_y) * 0.7
            ]

            # Connection line with depth effect
            canvas.create_line(
                center_x, center_y,
                *control_points,
                node_x, node_y,
                smooth=True,
                fill=line_color,
                width=NODE_CONFIG['line_width'],
                capstyle=tk.ROUND,
                tags=('connection', node_label)
            )

            # Directional arrow with modern design
            arrow_size = 10
            arrow_pos = (
                center_x + (node_x - center_x) * 0.78,
                center_y + (node_y - center_y) * 0.78
            )
            canvas.create_polygon(
                arrow_pos[0] - arrow_size / 1.5, arrow_pos[1] - arrow_size,
                arrow_pos[0] + arrow_size / 1.5, arrow_pos[1],
                arrow_pos[0] - arrow_size / 1.5, arrow_pos[1] + arrow_size,
                fill=line_color,
                outline='',
                smooth=True
            )

        # Draw all nodes with modern styling
        for node_label, (node_x, node_y) in node_positions.items():
            category = node_label.split(' ')[0] if ' ' in node_label else node_label
            node_color = category_colors.get(category, colors['secondary'])

            # Node shadow
            canvas.create_oval(
                node_x - NODE_CONFIG['node_radius'] + 4,
                node_y - NODE_CONFIG['node_radius'] + 4,
                node_x + NODE_CONFIG['node_radius'] + 4,
                node_y + NODE_CONFIG['node_radius'] + 4,
                fill=colors['shadow'],
                outline=''
            )

            # Main node body
            canvas.create_oval(
                node_x - NODE_CONFIG['node_radius'],
                node_y - NODE_CONFIG['node_radius'],
                node_x + NODE_CONFIG['node_radius'],
                node_y + NODE_CONFIG['node_radius'],
                fill='white',
                outline=node_color,
                width=2
            )

            # Inner glow effect
            canvas.create_oval(
                node_x - NODE_CONFIG['node_radius'] + 4,
                node_y - NODE_CONFIG['node_radius'] + 4,
                node_x + NODE_CONFIG['node_radius'] - 4,
                node_y + NODE_CONFIG['node_radius'] - 4,
                outline='#000000',
                width=1
            )

            # Node text with automatic scaling
            text = node_label if len(node_label) < 14 else node_label[:12] + '...'
            text_id = canvas.create_text(
                node_x, node_y,
                text=text,
                fill=colors['text'],
                font=NODE_CONFIG['font'],
                width=NODE_CONFIG['node_radius'] * 2 - 10
            )

            # Adjust font size if needed
            bbox = canvas.bbox(text_id)
            if bbox[2] - bbox[0] > NODE_CONFIG['node_radius'] * 1.8:
                canvas.itemconfig(text_id, font=('Segoe UI', 12))

        # Modern legend design
        def create_legend():
            legend_x, legend_y = 50, 50
            legend_width = 280

            # Legend container with shadow
            canvas.create_rectangle(
                legend_x + 4, legend_y + 4,
                legend_x + legend_width + 4, legend_y + 110 + 4,
                fill=colors['shadow'],
                outline=''
            )
            canvas.create_rectangle(
                legend_x, legend_y,
                legend_x + legend_width, legend_y + 110,
                fill='white',
                outline='#E0E0E0',
                width=1
            )

            # Legend content
            canvas.create_text(
                legend_x + 15, legend_y + 15,
                text="知识图谱图例",
                anchor='w',
                fill=colors['text'],
                font=('Segoe UI', 12, 'bold')
            )

            # Legend items
            items = [
                ('central', colors['accent'], '中心节点'),
                ('related', colors['secondary'], '关联节点'),
                ('relation', colors['relation'], '关系类型')
            ]

            for i, (item_type, color, label) in enumerate(items):
                y_pos = legend_y + 40 + i * 25
                if item_type == 'central':
                    canvas.create_oval(
                        legend_x + 15, y_pos - 6,
                        legend_x + 27, y_pos + 6,
                        fill=color,
                        outline=''
                    )
                else:
                    canvas.create_rectangle(
                        legend_x + 15, y_pos - 8,
                        legend_x + 27, y_pos + 8,
                        fill='white',
                        outline=color,
                        width=2
                    )
                canvas.create_text(
                    legend_x + 40, y_pos,
                    text=label,
                    anchor='w',
                    fill=colors['text'],
                    font=('Segoe UI', 11)
                )

        create_legend()

        # Interactive hover effects
        def on_hover(event):
            canvas.delete('hover_effect')
            x, y = event.x, event.y

            for node, (node_x, node_y) in node_positions.items():
                if sqrt((x - node_x) ** 2 + (y - node_y) ** 2) <= NODE_CONFIG['node_radius']:
                    # Highlight node
                    canvas.create_oval(
                        node_x - NODE_CONFIG['node_radius'] - 4,
                        node_y - NODE_CONFIG['node_radius'] - 4,
                        node_x + NODE_CONFIG['node_radius'] + 4,
                        node_y + NODE_CONFIG['node_radius'] + 4,
                        outline=colors['primary'],
                        width=2,
                        tags='hover_effect'
                    )

                    # Highlight connections
                    canvas.itemconfig(node_label, width=NODE_CONFIG['line_width'] + 2)

                    # Create modern tooltip
                    tooltip_text = data.get(node, "No additional information")
                    bbox = canvas.bbox(node_label)
                    tt_x = node_x + 20
                    tt_y = node_y - 20

                    # Tooltip container
                    canvas.create_rectangle(
                        tt_x - 10, tt_y - 30,
                        tt_x + len(tooltip_text) * 9 + 10, tt_y + 10,
                        fill='white',
                        outline=colors['primary'],
                        width=1,
                        tags='hover_effect'
                    )
                    canvas.create_text(
                        tt_x, tt_y - 10,
                        text=node,
                        anchor='w',
                        fill=colors['text'],
                        font=('Segoe UI', 11, 'bold'),
                        tags='hover_effect'
                    )
                    break

        canvas.bind("<Motion>", on_hover)
        canvas.bind("<Leave>", lambda e: canvas.delete('hover_effect'))

        # Add statistics overlay
        canvas.create_rectangle(
            1400 - 160, 50 - 20,
            1400 + 10, 50 + 40,
            fill='white',
            outline='#E0E0E0',
            width=1
        )
        canvas.create_text(
            1400 - 20, 50,
            text=f"总节点数\n{len(data[selected_node]) + 1}",
            fill=colors['text'],
            font=('Segoe UI', 12),
            anchor='e'
        )

    def moveit(s):
        # Call the existing function (assuming it's defined elsewhere)
        tianchong1()

        # Create the knowledge graph
        create_knowledge_graph(window, data, relation_data, s)

    def the_next():
        tianchong1()
        global y_index
        global num_nums
        y_index_k = 0
        msg1 = Message(window, text='目录   ', font=('华文新魏',20), bd=0, bg='red', width=90)
        msg1.place(relx=0.15, rely=0.1)
        num_index = 0
        y_num_index = 0
        t_num = len(data)
        i = 0
        q = num_nums
        num_index_dd = -1
        for data_s in data:
            num_index_dd += 1
            i += 1
            if i < q:
                continue
            num_index += 0.1
            if num_index >= 0.8:
                y_num_index += 0.1
                y_index += 1
                y_index_k += 1
                num_index = 0.1
            if y_index_k > 6:
                break
            num_nums += 1
            button_t = ttk.Button(window, text=data_s, bootstyle="success-link", width=10,
                                      command=lambda num_dd = num_index_dd:moveit(list(data.keys())[num_dd]))
            button_t.place(relx=0.1 + num_index, rely=0.25 + y_num_index, anchor='ne')
        # if num_nums % 8 != 0:
        #     while (True):
        #         num_index += 0.1
        #         if num_index >= 0.8:
        #             y_num_index += 0.1
        #             y_index += 1
        #             num_index = 0.1
        #         if num_nums % 8 == 0:
        #             break
        #         num_nums += 1
        #         button_t = tkinter.Button(window, text="        ", font=('华文新魏', 15), fg='blue', width=10, height=1,
        #                                   bd=0)
        #         button_t.place(relx=0.1 + num_index, rely=0.25 + y_num_index, anchor='ne')
        # num_index = 0
        # while (True):
        #     i += 1
        #     if i < q:
        #         continue
        #     num_index += 0.1
        #     if num_index >= 0.8:
        #         y_num_index += 0.1
        #         y_index += 1
        #         y_index_k += 1
        #         num_index = 0.1
        #     if y_index_k > 6:
        #         break
        #     num_nums += 1
        #     button_t = tkinter.Button(window, text="        ", font=('华文新魏', 15), fg='blue', width=10, height=1,
        #                               bd=0)
        #     button_t.place(relx=0.1 + num_index, rely=0.25 + y_num_index, anchor='ne')

    def the_before():
        tianchong1()
        global y_index
        global num_nums
        num_nums -= 120
        y_index_k = 0
        msg1 = Message(window, text='目录   ', font=('华文新魏',20), bd=0, bg='red', width=90)
        msg1.place(relx=0.15, rely=0.1)
        num_index = 0
        y_num_index = 0
        i = 0
        q = num_nums
        num_index_dd = -1
        for data_s in data:
            num_index_dd += 1
            i += 1
            if i < q:
                continue
            num_index += 0.1
            if num_index >= 0.8:
                y_num_index += 0.1
                y_index += 1
                y_index_k += 1
                num_index = 0.1
            if y_index_k > 6:
                break
            num_nums += 1

            button_t = ttk.Button(window, text=data_s, bootstyle="success-link", width=10,
                                      command=lambda num_dd = num_index_dd:moveit(list(data.keys())[num_dd]))
            button_t.place(relx=0.1 + num_index, rely=0.25 + y_num_index, anchor='ne')
        # if num_nums % 8 != 0:
        #     while (True):
        #         num_index += 0.1
        #         if num_index >= 0.8:
        #             y_num_index += 0.1
        #             y_index += 1
        #             num_index = 0.1
        #         if num_nums % 8 == 0:
        #             break
        #         num_nums += 1
        #         button_t = tkinter.Button(window, text="        ", font=('华文新魏', 15), fg='blue', width=10, height=1,
        #                                   bd=0)
        #         button_t.place(relx=0.1 + num_index, rely=0.25 + y_num_index, anchor='ne')
        # num_index = 0
        # while (True):
        #     i += 1
        #     if i < q:
        #         continue
        #     num_index += 0.1
        #     if num_index >= 0.8:
        #         y_num_index += 0.1
        #         y_index += 1
        #         y_index_k += 1
        #         num_index = 0.1
        #     if y_index_k > 6:
        #         break
        #     num_nums += 1
        #     button_t = tkinter.Button(window, text="        ", font=('华文新魏', 15), fg='blue', width=10, height=1,
        #                               bd=0)
        #     button_t.place(relx=0.1 + num_index, rely=0.25 + y_num_index, anchor='ne')

    msg1 = Message(window, text='目录   ', font=('华文新魏',20), bd=0, bg='red', width=90)
    msg1.place(relx=0.15, rely=0.1)
    num_index = 0
    y_num_index = 0
    num_index_dd = -1
    for data_s in data:
        num_index_dd += 1
        num_index += 0.1
        if num_index >= 0.8:
            y_num_index += 0.1
            y_index += 1
            num_index = 0.1
        if y_index > 6:
            break
        num_nums += 1
        button_t = ttk.Button(window, text=data_s, bootstyle="success-link", width=10,
                                  command=lambda num_dd = num_index_dd:moveit(list(data.keys())[num_dd]))
        button_t.place(relx=0.1 + num_index, rely=0.25 + y_num_index, anchor='ne')

    button_t = tkinter.Button(window, text="上一页", font=('华文新魏', 15), fg='blue', width=10, height=1,
                              command=the_before)
    button_t.place(relx=0.42, rely=0.9, anchor='ne')
    button_t = tkinter.Button(window, text="下一页", font=('华文新魏', 15), fg='blue', width=10, height=1,
                              command=the_next)
    button_t.place(relx=0.62, rely=0.9, anchor='ne')

def tianjia_d():
    global data_load
    data_load = {}
    def clear_s(ss):
        strs = ""
        for dd in ss:
            u = 0
            for qq in ad:
                if dd == qq:
                    u = 1
                    break
            if u == 0:
                strs += dd
        return strs
    tianchong()
    k1 = Label(window, text='输入文件的绝对路径', font='华文新魏')
    e1 = tkinter.Entry(window, width=55)
    e1.pack()
    k1.pack()
    e1.place(relx=0.35, rely=0.1)
    k1.place(relx=0.35, rely=0.05)
    def insert_end():
        totle_relation = []
        def relation_insert(test_main,test_obj,test_relat):
            ss = test_relat
            position1 = []
            position2 = []

            index1 = ss.index(test_main)
            index2 = ss.index(test_obj)

            for i, word in enumerate(ss):
                position1.append(i - index1)
                position2.append(i - index2)
                i += 1

            sentence = []
            pos1 = []
            pos2 = []

            sentence.append(X_padding(ss))
            pos1.append(position_padding(position1))
            pos2.append(position_padding(position2))

            sentence = torch.tensor(sentence)
            pos1 = torch.tensor(pos1)
            pos2 = torch.tensor(pos2)

            sentence = Variable(sentence)
            pos1 = Variable(pos1)
            pos2 = Variable(pos2)
            y = model_r(sentence, pos1, pos2)
            y = np.argmax(y.data.numpy(), axis=1)
            ans = id2relation[y[0]]
            return ans

        num_right = 0
        var = e1.get()
        e1.delete(0,END)
        trans = os.path.splitext(var)
        name1,name2 = trans
        if os.path.exists("test.txt"):
            os.remove("test.txt")
        if name2 == '.pdf':
            py.pdf_docx(var)
            py.word(name1+".docx")
            num_right = 1
        elif name2 == '.wps':
            py.wps_train(var)
            num_right = 1
        elif name2 == '.docx':
            py.word(var)
            num_right = 1
        elif name2 == '.ofd':
            py.ofd_train(var)
            num_right = 1
        elif name2 == '.txt':
            py.txt_train(var)
            num_right = 1
        else:
            tkinter.messagebox.showinfo(title="HI", message="请输入正确的表示")

        lsit = []
        if num_right == 1:
            fp = r"test.txt"
            with open(fp, "r", encoding='UTF-8') as f:
                all_line_contents: list = f.readlines()
                for i in all_line_contents:
                    if i:
                        i = i.replace("\n", '')
                    if i != "":
                        lsit.append(i)
        qwd = 0
        with open('model/dev.json', 'w', encoding='utf-8') as json_file:
            js = json.dumps([], ensure_ascii=False)
        for texti in lsit:
            dict1 = {}
            if len(texti) > 50:
                texti = texti[0:50]
            dict1["text"] = texti
            dict1["label"] = {}
            qwd += 1
            with open('model/dev.json', 'a', encoding='utf-8') as json_file:
                js = json.dumps(dict1, ensure_ascii=False)
                json_file.write(js)
                json_file.write('\n')
        while(qwd%128 < 127):
            dict1 = {}
            dict1["text"] = " "
            dict1["label"] = {}
            qwd += 1
            with open('model/dev.json', 'a', encoding='utf-8') as json_file:
                js = json.dumps(dict1, ensure_ascii=False)
                json_file.write(js)
                json_file.write('\n')
        dict1 = {}
        dict1["text"] = " "
        dict1["label"] = {}
        qwd += 1
        with open('model/dev.json', 'a', encoding='utf-8') as json_file:
            js = json.dumps(dict1, ensure_ascii=False)
            json_file.write(js)
            json_file.write('\n')
        data_index = []
        data_rela = []
        vecc = bertBILSTM.evaluate()
        for i in range(min(len(vecc),len(lsit))):
            s = ""
            vec_re = []
            for j in range(min(min(len(lsit[i]),50),len(vecc[i]))):
                if vecc[i][j] != 20:
                    s += lsit[i][j]
                elif s != "":
                    data_index.append(s)
                    vec_re.append(s)
                    s = ""
            data_rela.append(vec_re)
        for kw1 in data_rela:
            if len(kw1) >= 2:
                if kw1[1] not in data:
                    data[kw1[1]] = []
                data[kw1[1]].append(data[kw1[0]])
                if kw1[0] not in data:
                    data[kw1[0]] = []
                data[kw1[0]].append(data[kw1[1]])
        data_index = []
        data_rela = []
        totle_app = []
        totle_str = []
        for i in lsit:
            s = ""
            for q in i:
                if q not in word_to_ix:
                    continue
                s += q
            totle_str.append(s)
            totle_app.append(model(prepare_sequence(s, word_to_ix).to(device))[1])
        for i in range(len(totle_app)):
            s = ""
            vec_re = []
            for j in range(len(totle_app[i])):
                if totle_app[i][j] != 2:
                    s += totle_str[i][j]
                elif s != "":
                    data_index.append(s)
                    vec_re.append(s)
                    s = ""
            data_rela.append(vec_re)
        for i in range(len(data_rela)):
            if len(data_rela) < 3:
                continue
            for j in range(len(data_rela[i]) - 3):
                ans1 = relation_insert(data_rela[i][j],data_rela[i][j+1],totle_str[i])
                ans2 = relation_insert(data_rela[i][j], data_rela[i][j + 2], totle_str[i])
                data_rela[i][j] = clear_s(data_rela[i][j])
                data_rela[i][j+1] = clear_s(data_rela[i][j+1])
                data_rela[i][j+2] = clear_s(data_rela[i][j+2])

                if data_rela[i][j] not in relation_data:
                    vec = {}
                    vec[data_rela[i][j+1]] = ans1
                    relation_data[data_rela[i][j]] = vec
                else:
                    relation_data[data_rela[i][j]][data_rela[i][j+1]] = ans1
                relation_data[data_rela[i][j]][data_rela[i][j+2]] = ans2
        with open('target_relation.json', 'w',encoding='utf-8') as json_file:
            js = json.dumps(relation_data, ensure_ascii=False)
            json_file.write(js)
        for i in data_index:
            if len(i) < 2:
                continue
            dd_d = 0
            for j in data_index:
                if len(j) < 2:
                    continue
                dd_d = 1
            if dd_d == 0:
                continue
            if i not in data:
                data[i] = []
            if i not in data_load:
                data_load[i] = []
            qw = 0
            dw = 0
            for j in data_index:
                if len(j) < 2:
                    continue
                if dw > 2:
                    break
                if qw == 1:
                    dw += 1
                    data[i].append(j)
                    data_load[i].append(j)
                    if j not in data_load:
                        data_load[j] = []
                        data_load[j].append(i)
                    else:
                        uk = 0
                        for kkk in data_load[j]:
                            if kkk == i:
                                uk = 1
                        if uk == 0:
                            data_load[j].append(i)
                    if j not in data:
                        data[j] = []
                        data[j].append(i)
                    else:
                        uk = 0
                        for kkk in data[j]:
                            if kkk == i:
                                uk = 1
                        if uk == 0:
                            data[j].append(i)
                if i == j:
                    qw = 1
        for k in data:
            vec = {}
            for j in range(len(data[k])):
                if data[k][j] not in vec:
                    vec[data[k][j]] = 1
            finda = []
            for k_w in vec:
                finda.append(k_w)
            data[k] = finda
        with open('taget_main.json', 'w',encoding='utf-8') as json_file:
            js = json.dumps(data, ensure_ascii=False)
            json_file.write(js)

        num_right = 0

    b = tkinter.Button(window, text="关系抽取", width=15, height=2,font='华文新魏',command=insert_end)
    b.pack()
    b.place(relx=0.35, rely=0.15)

def relation_rely():
    tianchong()
    k = Label(window, text='输入句子', font='华文新魏')
    k1 = Label(window, text='输入主体', font='华文新魏')
    k2 = Label(window, text='输入客体', font='华文新魏')
    e = tkinter.Entry(window, width=115)
    e1 = tkinter.Entry(window, width=35)
    e2 = tkinter.Entry(window, width=35)
    e.pack()
    k.pack()
    k1.place(relx=0.25, rely=0.05)
    k2.place(relx=0.25, rely=0.15)
    k.place(relx=0.25, rely=0.25)
    e.place(relx=0.25, rely=0.3)
    e1.place(relx=0.25, rely=0.1)
    e2.place(relx=0.25, rely=0.2)
    t2 = tkinter.Text(window, height=1)
    k2 = Label(window, text='关系', font='华文新魏')
    t2.pack()
    k2.pack()
    t2.place(relx=0.25, rely=0.59)
    k2.place(relx=0.25, rely=0.54)
    def insert_s():
        test_main = e.get()
        test_obj = e2.get()
        test_relat = e1.get()
        ss = test_main
        position1 = []
        position2 = []

        index1 = ss.index(test_relat)
        index2 = ss.index(test_obj)

        for i, word in enumerate(ss):
            position1.append(i - index1)
            position2.append(i - index2)
            i += 1

        sentence = []
        pos1 = []
        pos2 = []

        sentence.append(X_padding(ss))
        pos1.append(position_padding(position1))
        pos2.append(position_padding(position2))

        sentence = torch.tensor(sentence)
        pos1 = torch.tensor(pos1)
        pos2 = torch.tensor(pos2)

        sentence = Variable(sentence)
        pos1 = Variable(pos1)
        pos2 = Variable(pos2)
        y = model_r(sentence, pos1, pos2)
        y = np.argmax(y.data.numpy(), axis=1)
        t2.delete("1.0", "end")
        t2.insert("insert", id2relation[y[0]]+'\n')
    def insert_end():
        var = e.get()
        e.delete(0,END)
        e1.delete(0,END)
        e2.delete(0,END)

    b2 = tkinter.Button(window, text="重新输入", width=15, height=2,font='华文新魏',command = insert_end)
    b2.pack()
    b2.place(relx=0.45, rely=0.4)
    b = tkinter.Button(window, text="判断关系", width=15, height=2, font='华文新魏',command = insert_s)
    b.pack()
    b.place(relx=0.25, rely=0.4)

def search():
    tianchong()
    k = Label(window, text='输入单词', font='华文新魏')
    e = tkinter.Entry(window, width=35)
    e.pack()
    k.pack()
    k.place(relx=0.25, rely=0.05)
    e.place(relx=0.25, rely=0.1)
    def insert_s_w():
        global data
        main_dat = e.get()
        data_ds = data
        vec_num_word = []
        for i in data:
            index = 0
            for k in i:
                if k == main_dat[index]:
                    index += 1
                if index == len(main_dat)-1:
                    vec_num_word.append(i)
                    break
        if main_dat not in data and len(vec_num_word) == 0:
            tkinter.messagebox.showinfo(title="HI", message="未找到相应的词语")
        else:
            vec2vec = {}
            if main_dat in data:
                for i in data[main_dat]:
                    if i in data:
                        vec2vec[i] = data[i]
            for i in vec_num_word:
                vec2vec[i] = data[i]
            data = vec2vec
            mulu()
            data = data_ds



    b = tkinter.Button(window, text="搜索", width=15, height=2, font='华文新魏',command= insert_s_w)
    b.pack()
    b.place(relx=0.25, rely=0.15)


def connect():
    global data
    dictt = data
    for i in list(data.keys()):
        st = ""
        for j in i:
            q = 0
            for k in ad:
                if k == j:
                    q = 1
                    break
            if q == 0:
                st += j
        if len(st) > 1:
            dictt[st] = dictt.pop(i)
            vec = []
            for k in range(len(dictt[st])):
                strr = ""
                for dd in dictt[st][k]:
                    u = 0
                    for qq in ad:
                        if dd == qq:
                            u = 1
                            break
                    if u == 0:
                        strr += dd
                if len(strr) > 1:
                    vec.append(strr)
            if len(vec) > 0:
                dictt[st] = vec
            else:
                del dictt[st]
        else:
            del dictt[i]
    for i in list(data.keys()):
        for j in list(data.keys()):
            if i not in dictt or j not in dictt:
                continue
            if i!=j:
                ans_i = i
                ans_j = j
                if len(ans_i) > len(ans_j):
                    sd = ans_i
                    ans_i = ans_j
                    ans_j = sd
                inde_x = 0
                for k in range(len(ans_j)):
                    if inde_x >= len(ans_i):
                        break
                    if ans_j[k] == ans_i[inde_x]:
                        inde_x+=1
                if inde_x/len(ans_j) > 4/5:
                    if i > j:
                        del dictt[i]
                    else:
                        del dictt[j]
    data = dictt
    with open('taget_main.json', 'w', encoding='utf-8') as json_file:
        js = json.dumps(data, ensure_ascii=False)
        json_file.write(js)

def swap_s():
    global data
    global data_res
    dadad = data
    data = data_res
    data_res = dadad

def swapp():
    global data_load
    global data
    if len(data_load) == 0:
        tkinter.messagebox.showinfo(title="HI", message="未找到相应的词语")
        return
    data_sw = []
    data_sw = data_load
    data_load = data
    data = data_sw
    mulu()
    data_sw = data_load
    data_load = data
    data = data_sw

for i in data:
    for j in data[i]:
        if j in relation_data and i in relation_data[j]:
            if i not in data_res:
                data_res[i] = []
            else:
                uu = 0
                for k in data_res[i]:
                    if k == j:
                        uu = 1
                        break
                if uu == 1:
                    continue
            data_res[i].append(j)
        if i in relation_data and j in relation_data[i]:
            if j not in data_res:
                data_res[j] = []
            else:
                uu = 0
                for k in data_res[j]:
                    if k == i:
                        uu = 1
                        break
                if uu == 1:
                    continue
            data_res[j].append(i)

data_load = {}

origin = (10, 120)
unit_length_x = 20
unit_length_y = 40
curve_precision = 0.01
fun_text = 'y=sin(x)'

coordinate_x = origin[1]
coordinate_y = origin[0]

def calc_x(t):
    num_x = t * unit_length_x
    num_x += coordinate_y
    return num_x


def calc_y(t):
    num_y = math.sin(t) * unit_length_y
    num_y = -num_y
    num_y += coordinate_x
    return num_y

def quick():
    global data_load
    data_load = {}

    def clear_s(ss):
        strs = ""
        for dd in ss:
            u = 0
            for qq in ad:
                if dd == qq:
                    u = 1
                    break
            if u == 0:
                strs += dd
        return strs

    tianchong()
    k1 = Label(window, text='输入文件的绝对路径', font='华文新魏')
    e1 = tkinter.Entry(window, width=55)
    e1.pack()
    k1.pack()
    e1.place(relx=0.35, rely=0.1)
    k1.place(relx=0.35, rely=0.05)

    def insert_end():
        totle_relation = []

        def relation_insert(test_main, test_obj, test_relat):
            ss = test_relat
            position1 = []
            position2 = []

            index1 = ss.index(test_main)
            index2 = ss.index(test_obj)

            for i, word in enumerate(ss):
                position1.append(i - index1)
                position2.append(i - index2)
                i += 1

            sentence = []
            pos1 = []
            pos2 = []

            sentence.append(X_padding(ss))
            pos1.append(position_padding(position1))
            pos2.append(position_padding(position2))

            sentence = torch.tensor(sentence)
            pos1 = torch.tensor(pos1)
            pos2 = torch.tensor(pos2)

            sentence = Variable(sentence)
            pos1 = Variable(pos1)
            pos2 = Variable(pos2)
            y = model_r(sentence, pos1, pos2)
            y = np.argmax(y.data.numpy(), axis=1)
            ans = id2relation[y[0]]
            return ans

        num_right = 0
        var = e1.get()
        e1.delete(0, END)
        trans = os.path.splitext(var)
        name1, name2 = trans
        if os.path.exists("test.txt"):
            os.remove("test.txt")
        if name2 == '.pdf':
            py.pdf_docx(var)
            py.word(name1 + ".docx")
            num_right = 1
        elif name2 == '.wps':
            py.wps_train(var)
            num_right = 1
        elif name2 == '.docx':
            py.word(var)
            num_right = 1
        elif name2 == '.ofd':
            py.ofd_train(var)
            num_right = 1
        elif name2 == '.txt':
            py.txt_train(var)
            num_right = 1
        else:
            tkinter.messagebox.showinfo(title="HI", message="请输入正确的表示")

        lsit = []
        if num_right == 1:
            fp = r"test.txt"
            with open(fp, "r", encoding='UTF-8') as f:
                all_line_contents: list = f.readlines()
                for i in all_line_contents:
                    if i:
                        i = i.replace("\n", '')
                    if i != "":
                        lsit.append(i)
        qwd = 0
        with open('model/dev.json', 'w', encoding='utf-8') as json_file:
            js = json.dumps([], ensure_ascii=False)
        for texti in lsit:
            dict1 = {}
            if len(texti) > 50:
                texti = texti[0:50]
            dict1["text"] = texti
            dict1["label"] = {}
            qwd += 1
            with open('model/dev.json', 'a', encoding='utf-8') as json_file:
                js = json.dumps(dict1, ensure_ascii=False)
                json_file.write(js)
                json_file.write('\n')
        while (qwd % 128 < 127):
            dict1 = {}
            dict1["text"] = " "
            dict1["label"] = {}
            qwd += 1
            with open('model/dev.json', 'a', encoding='utf-8') as json_file:
                js = json.dumps(dict1, ensure_ascii=False)
                json_file.write(js)
                json_file.write('\n')
        dict1 = {}
        dict1["text"] = " "
        dict1["label"] = {}
        qwd += 1
        with open('model/dev.json', 'a', encoding='utf-8') as json_file:
            js = json.dumps(dict1, ensure_ascii=False)
            json_file.write(js)
            json_file.write('\n')
        data_index = []
        data_rela = []
        totle_app = []
        totle_str = []
        for i in lsit:
            s = ""
            for q in i:
                if q not in word_to_ix:
                    continue
                s += q
            totle_str.append(s)
            totle_app.append(model(prepare_sequence(s, word_to_ix).to(device))[1])
        for i in range(len(totle_app)):
            s = ""
            vec_re = []
            for j in range(len(totle_app[i])):
                if totle_app[i][j] != 2:
                    s += totle_str[i][j]
                elif s != "":
                    data_index.append(s)
                    vec_re.append(s)
                    s = ""
            data_rela.append(vec_re)
        for i in range(len(data_rela)):
            if len(data_rela) < 3:
                continue
            for j in range(len(data_rela[i]) - 3):
                ans1 = relation_insert(data_rela[i][j], data_rela[i][j + 1], totle_str[i])
                ans2 = relation_insert(data_rela[i][j], data_rela[i][j + 2], totle_str[i])
                data_rela[i][j] = clear_s(data_rela[i][j])
                data_rela[i][j + 1] = clear_s(data_rela[i][j + 1])
                data_rela[i][j + 2] = clear_s(data_rela[i][j + 2])

                if data_rela[i][j] not in relation_data:
                    vec = {}
                    vec[data_rela[i][j + 1]] = ans1
                    relation_data[data_rela[i][j]] = vec
                else:
                    relation_data[data_rela[i][j]][data_rela[i][j + 1]] = ans1
                relation_data[data_rela[i][j]][data_rela[i][j + 2]] = ans2
        with open('target_relation.json', 'w', encoding='utf-8') as json_file:
            js = json.dumps(relation_data, ensure_ascii=False)
            json_file.write(js)
        for i in data_index:
            if len(i) < 2:
                continue
            dd_d = 0
            for j in data_index:
                if len(j) < 2:
                    continue
                dd_d = 1
            if dd_d == 0:
                continue
            if i not in data:
                data[i] = []
            if i not in data_load:
                data_load[i] = []
            qw = 0
            dw = 0
            for j in data_index:
                if len(j) < 2:
                    continue
                if dw > 2:
                    break
                if qw == 1:
                    dw += 1
                    data[i].append(j)
                    data_load[i].append(j)
                    if j not in data_load:
                        data_load[j] = []
                        data_load[j].append(i)
                    else:
                        uk = 0
                        for kkk in data_load[j]:
                            if kkk == i:
                                uk = 1
                        if uk == 0:
                            data_load[j].append(i)
                    if j not in data:
                        data[j] = []
                        data[j].append(i)
                    else:
                        uk = 0
                        for kkk in data[j]:
                            if kkk == i:
                                uk = 1
                        if uk == 0:
                            data[j].append(i)
                if i == j:
                    qw = 1
        for k in data:
            vec = {}
            for j in range(len(data[k])):
                if data[k][j] not in vec:
                    vec[data[k][j]] = 1
            finda = []
            for k_w in vec:
                finda.append(k_w)
            data[k] = finda
        with open('taget_main.json', 'w', encoding='utf-8') as json_file:
            js = json.dumps(data, ensure_ascii=False)
            json_file.write(js)

        num_right = 0

    b = tkinter.Button(window, text="关系抽取", width=15, height=2, font='华文新魏', command=insert_end)
    b.pack()
    b.place(relx=0.35, rely=0.15)

def begin():
    # 初始化窗口配置
    window.title('知识图谱管理系统 - 信创版')
    window.geometry('1900x1100')
    window.configure(bg='#f8f9fa')

    # ================= 样式配置 =================
    STYLE_CONFIG = {
        'colors': {
            'primary': '#2c3e50',  # 主色调
            'secondary': '#3498db',  # 辅助色
            'accent': '#e74c3c',  # 强调色
            'background': '#ecf0f1',  # 背景色
            'text': '#2c3e50'  # 文字色
        },
        'fonts': {
            'header': ('华文新魏', 24, 'bold'),
            'button': ('微软雅黑', 12),
            'label': ('Segoe UI', 10)
        },
        'spacing': {
            'padx': 8,
            'pady': 6
        }
    }

    # 初始化ttk样式
    style = ttk.Style()

    # 自定义样式配置
    style.configure('Header.TFrame', background=STYLE_CONFIG['colors']['primary'])
    style.configure('Sidebar.TFrame', background='#ffffff', relief=ttk.RAISED, borderwidth=1)
    style.configure('Primary.TButton',
                    font=STYLE_CONFIG['fonts']['button'],
                    foreground='white',
                    background=STYLE_CONFIG['colors']['primary'],
                    padding=10,
                    borderwidth=0)
    style.map('Primary.TButton',
              background=[('active', STYLE_CONFIG['colors']['secondary'])])
    style.configure('Accent.TButton',
                    font=STYLE_CONFIG['fonts']['button'],
                    foreground='white',
                    background=STYLE_CONFIG['colors']['accent'],
                    padding=10)
    style.configure('ThemeSelector.TCombobox', font=STYLE_CONFIG['fonts']['label'])

    # ================= 布局结构 =================
    # 顶部标题栏
    header_frame = ttk.Frame(window, style='Header.TFrame')
    header_frame.pack(fill=ttk.X, side=ttk.TOP)

    ttk.Label(
        header_frame,
        text='基于信创的知识图谱系统',
        font=STYLE_CONFIG['fonts']['header'],
        foreground='white',
        background=STYLE_CONFIG['colors']['primary']
    ).pack(pady=15)

    # 主内容容器
    main_container = ttk.Frame(window)
    main_container.pack(fill=ttk.BOTH, expand=True)

    # 侧边导航栏
    sidebar_frame = ttk.Frame(main_container, style='Sidebar.TFrame', width=220)
    sidebar_frame.pack(side=ttk.LEFT, fill=ttk.Y, padx=(0, 10), pady=10)

    # 功能按钮区域
    button_container = ttk.Frame(sidebar_frame)
    button_container.pack(padx=10, pady=20, fill=ttk.X)

    # ================= 功能按钮 =================
    BUTTONS = [
        ('🏠 总体目录', mulu),
        ('➕ 添加新关系', tianjia),
        ('⚡ 直接添加', tianjia_d),
        ('🔗 关系查看', relation_rely),
        ('🔍 搜索词语', search),
        ('📦 整理实体', connect),
        ('🔄 数据转换', swap_s),
        ('⏮ 上一次提取', swapp),
        ('🚀 快速提取', quick),
        ('📝 识别句子', panduan),
        ('⏏ 退出系统', lambda: window.destroy())
    ]

    for text, command in BUTTONS:
        btn = ttk.Button(
            button_container,
            text=text,
            style='Primary.TButton' if '退出' not in text else 'Accent.TButton',
            command=command
        )
        btn.pack(fill=ttk.X, pady=3)

    # ================= 主题选择器 =================
    theme_frame = ttk.Frame(sidebar_frame, style='Sidebar.TFrame')
    theme_frame.pack(side=ttk.BOTTOM, fill=ttk.X, padx=10, pady=20)

    ttk.Label(theme_frame,
              text="界面主题：",
              font=STYLE_CONFIG['fonts']['label'],
              background='white').pack(side=ttk.LEFT)

    theme_cbo = ttk.Combobox(
        theme_frame,
        values=['flatly', 'litera', 'minty', 'pulse', 'sandstone'],
        style='ThemeSelector.TCombobox',
        state='readonly',
        width=12
    )
    theme_cbo.set('flatly')
    theme_cbo.pack(side=ttk.RIGHT)
    theme_cbo.bind('<<ComboboxSelected>>', lambda e: style.theme_use(theme_cbo.get()))

    # ================= 内容区域 =================
    content_frame = ttk.Frame(main_container, style='TFrame')
    content_frame.pack(fill=ttk.BOTH, expand=True, padx=10, pady=10)

    # 初始化默认视图
    mulu()

# 其他功能函数保持原样...
window.title('登录界面')
window.geometry('600x600')
bg = Label(window, text='登录界面', font=('华文新魏',45))
k = Label(window, text='输入', font='华文新魏')
k1 = Label(window, text='输入', font='华文新魏')
k2 = Label(window, text='输入', font='华文新魏')
k3 = Label(window, text='输入', font='华文新魏')
k4 = Label(window, text='输入', font='华文新魏')
e = tkinter.Entry(window, width=35)
e1 = tkinter.Entry(window, width=35)
e2 = tkinter.Entry(window, width=35)
e3 = tkinter.Entry(window, width=35)
e4 = tkinter.Entry(window, width=35)
e.pack()
k.pack()
bg.place(relx=0.25,rely=0.05)
k1.place(relx=0.25, rely=0.2)
k2.place(relx=0.25, rely=0.3)
k.place(relx=0.25, rely=0.4)
k3.place(relx=0.25, rely=0.5)
k4.place(relx=0.25, rely=0.6)
e.place(relx=0.25, rely=0.45)
e1.place(relx=0.25, rely=0.25)
e2.place(relx=0.25, rely=0.35)
e3.place(relx=0.25, rely=0.55)
e4.place(relx=0.25, rely=0.65)

b = tkinter.Button(window, text="开始",font='华文新魏', width=15, height=2,command=begin)
b.pack()
b.place(relx=0.20, rely=0.75)
b1 = tkinter.Button(window, text="退出",font='华文新魏', width=15, height=2,command=begin)
b1.pack()
b1.place(relx=0.55, rely=0.75)


window.mainloop()
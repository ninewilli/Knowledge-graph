import os
from transformers import BertTokenizer
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import time
import datetime
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn import metrics
from itertools import chain
from transformers import BertForTokenClassification, AdamW, BertConfig
import os
import json
import pickle
from transformers import BertModel, AdamW, BertTokenizer

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import bertBILSTM
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(bertBILSTM.evaluate())
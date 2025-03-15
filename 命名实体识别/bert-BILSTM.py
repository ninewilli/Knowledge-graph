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
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_label_map(data_path=''):
    label_map_path = '/kaggle/input/mingming/label_map.json'
    if os.path.exists(label_map_path):
        with open(label_map_path, 'r', encoding='utf-8') as fp:
            label_map = json.load(fp)
    else:
        json_data = []
        with open(data_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                json_data.append(json.loads(line))


    n_classes = []
    for data in json_data:
        for label in data['label'].keys():
            if label not in n_classes:
                n_classes.append(label)
    n_classes.sort()

    label_map = {}
    for n_class in n_classes:
        label_map['B-' + n_class] = len(label_map)
        label_map['I-' + n_class] = len(label_map)
    label_map['O'] = len(label_map)
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    label_map[START_TAG] = len(label_map)
    label_map[STOP_TAG] = len(label_map)

    with open(label_map_path, 'w', encoding='utf-8') as fp:
        json.dump(label_map, fp, indent=4)

    label_map_inv = {v: k for k, v in label_map.items()}
    return label_map, label_map_inv


def get_vocab(data_path=''):
    vocab_path = '/kaggle/input/mingming/vocab.pkl'
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as fp:
            vocab = pickle.load(fp)
    else:
        json_data = []
        with open(data_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                json_data.append(json.loads(line))
        vocab = {'PAD': 0, 'UNK': 1}
        for data in json_data:
            for word in data['text']:
                if word not in vocab:
                    vocab[word] = len(vocab)
        with open(vocab_path, 'wb') as fp:
            pickle.dump(vocab, fp)


    vocab_inv = {v: k for k, v in vocab.items()}
    return vocab, vocab_inv



def data_process(path):
    json_data = []
    with open(path, 'r', encoding='utf-8') as fp:
        for line in fp:
            json_data.append(json.loads(line))
    data = []
    for i in range(len(json_data)):
        label = ['O'] * len(json_data[i]['text'])
        for n in json_data[i]['label']:
            for key in json_data[i]['label'][n]:
                for n_list in range(len(json_data[i]['label'][n][key])):
                    start = json_data[i]['label'][n][key][n_list][0]
                    end = json_data[i]['label'][n][key][n_list][1]
                    label[start] = 'B-' + n
                    label[start + 1: end + 1] = ['I-' + n] * (end - start)


    texts = []
    for t in json_data[i]['text']:
        texts.append(t)

    data.append([texts, label])
    return data



class Mydataset(Dataset):
    def __init__(self, file_path, vocab, label_map):
        self.file_path = file_path
        self.data = data_process(self.file_path)
        self.label_map, self.label_map_inv = label_map
        self.vocab, self.vocab_inv = vocab
        self.examples = []
        for text, label in self.data:
            t = [self.vocab.get(t, self.vocab['UNK']) for t in text]
            l = [self.label_map[l] for l in label]
            q = [q for q in text]
            self.examples.append([t, l, q])
        def __getitem__(self, item):
            return self.examples[item]


        def __len__(self):
            return len(self.data)



        def collect_fn(self, batch):
            text = [t for t, l, q in batch]
            label = [l for t, l, q in batch]
            sentence = [q for t, l, q in batch]
            seq_len = [len(i) for i in text]
            max_len = max(seq_len)


            text = [t + [self.vocab['PAD']] * (max_len - len(t)) for t in text]
            label = [l + [self.label_map['O']] * (max_len - len(l)) for l in label]

            text = torch.tensor(text, dtype=torch.long)
            label = torch.tensor(label, dtype=torch.long)
            seq_len = torch.tensor(seq_len, dtype=torch.long)
            return text, label, seq_len, sentence


def log_sum_exp(vec):
    max_score, _ = torch.max(vec, dim=-1)
    max_score_broadcast = max_score.unsqueeze(-1).repeat_interleave(vec.shape[-1], dim=-1)
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=-1))



class BiLSTM_CRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab, label_map, device='cpu'):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(vocab)
        self.tagset_size = len(label_map)
        self.device = device
        self.state = 'train'

        self.bert = BertModel.from_pretrained('bert-base-chinese')
        config = self.bert.config
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        self.word_embeds = nn.Embedding(self.vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=2, bidirectional=True, batch_first=True)


        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size, bias=True)
        self.crf = CRF(label_map, device)
        self.dropout = nn.Dropout(p=0.5, inplace=True)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.max_length = 50
        self.linear = nn.Linear(768, 128)


def _get_lstm_features(self, sentence, seq_len):
    embeds = sentence
    self.dropout(embeds)


    packed = torch.nn.utils.rnn.pack_padded_sequence(embeds, seq_len.cpu(), batch_first=True, enforce_sorted=False)
    lstm_out, _ = self.lstm(packed)
    seq_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

    seqence_output = self.layer_norm(seq_unpacked)
    lstm_feats = self.hidden2tag(seqence_output)
    return lstm_feats


def __get_lstm_features(self, sentence, seq_len):
    max_len = sentence.shape[1]
    mask = [[1] * seq_len[i] + [0] * (max_len - seq_len[i]) for i in range(sentence.shape[0])]


    embeds = self.word_embeds(sentence)
    self.dropout(embeds)
    mask = torch.tensor(mask, dtype=torch.float32, device=self.device)
    input = embeds * mask.unsqueeze(2)
    lstm_out, _ = self.lstm(input)
    seqence_output = self.layer_norm(lstm_out)
    lstm_feats = self.hidden2tag(seqence_output)
    return lstm_feats


def forward(self, sentence, seq_len, text, tags=''):
    sen = []
    for sen_s in text:
        x = self.tokenizer.encode_plus(sen_s, return_token_type_ids=True, return_attention_mask=True,
                                       return_tensors='pt',
                                       padding='max_length', max_length=self.max_length).to(device)
        bert_output = \
        self.bert(input_ids=x.input_ids, attention_mask=x.attention_mask, token_type_ids=x.token_type_ids)[0]
        bert_output = self.linear(bert_output)
        bert_output = torch.squeeze(bert_output, dim=0)
        bert_output = bert_output.tolist()
        lens = len(bert_output)
        if lens > 50:
            del bert_output[50:lens]
        sen.append(bert_output)
    sen = torch.Tensor(sen).to(device)
    sentence = sen
    feats = self._get_lstm_features(sentence, seq_len)
    if self.state == 'train':
        loss = self.crf.neg_log_likelihood(feats, tags, seq_len)
        return loss
    elif self.state == 'eval':
        all_tag = []
        for i, feat in enumerate(feats):
            all_tag.append(self.crf._viterbi_decode(feat[:seq_len[i]])[1])
        return all_tag
    else:
        return self.crf._viterbi_decode(feats[0])[1]




class CRF(nn.Module):
    def __init__(self, label_map, device='cpu'):
        super(CRF, self).__init__()
        self.label_map = label_map
        self.label_map_inv = {v: k for k, v in label_map.items()}
        self.tagset_size = len(self.label_map)
        self.device = device


        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"
        self.transitions.data[self.label_map[self.START_TAG], :] = -10000
        self.transitions.data[:, self.label_map[self.STOP_TAG]] = -10000


def _forward_alg(self, feats, seq_len):
    init_alphas = torch.full((self.tagset_size,), -10000.)
    init_alphas[self.label_map[self.START_TAG]] = 0.


    forward_var = torch.zeros(feats.shape[0], feats.shape[1] + 1, feats.shape[2], dtype=torch.float32,
                              device=self.device)
    forward_var[:, 0, :] = init_alphas

    transitions = self.transitions.unsqueeze(0).repeat(feats.shape[0], 1, 1)
    for seq_i in range(feats.shape[1]):
        emit_score = feats[:, seq_i, :]
        tag_var = (
                forward_var[:, seq_i, :].unsqueeze(1).repeat(1, feats.shape[2], 1)  # (batch_size, tagset_size, tagset_size)
                + transitions
                + emit_score.unsqueeze(2).repeat(1, 1, feats.shape[2])
        )
        cloned = forward_var.clone()
        cloned[:, seq_i + 1, :] = log_sum_exp(tag_var)
        forward_var = cloned

    forward_var = forward_var[range(feats.shape[0]), seq_len, :]
    terminal_var = forward_var + self.transitions[self.label_map[self.STOP_TAG]].unsqueeze(0).repeat(feats.shape[0], 1)
    alpha = log_sum_exp(terminal_var)
    return alpha


def _score_sentence(self, feats, tags, seq_len):
    score = torch.zeros(feats.shape[0], device=self.device)
    start = torch.tensor([self.label_map[self.START_TAG]], device=self.device).unsqueeze(0).repeat(feats.shape[0], 1)
    tags = torch.cat([start, tags], dim=1)
    for batch_i in range(feats.shape[0]):
        score[batch_i] = torch.sum(
            self.transitions[tags[batch_i, 1:seq_len[batch_i] + 1], tags[batch_i, :seq_len[batch_i]]]) \
                         + torch.sum(feats[batch_i, range(seq_len[batch_i]), tags[batch_i][1:seq_len[batch_i] + 1]])
        score[batch_i] += self.transitions[self.label_map[self.STOP_TAG], tags[batch_i][seq_len[batch_i]]]
    return score



def _viterbi_decode(self, feats):
    backpointers = []
    init_vvars = torch.full((1, self.tagset_size), -10000., device=self.device)
    init_vvars[0][self.label_map[self.START_TAG]] = 0
    forward_var = init_vvars
    for feat in feats:
        forward_var = forward_var.repeat(feat.shape[0], 1)
        next_tag_var = forward_var + self.transitions
        bptrs_t = torch.max(next_tag_var, 1)[1].tolist()
        viterbivars_t = next_tag_var[range(forward_var.shape[0]), bptrs_t]
        forward_var = (viterbivars_t + feat).view(1, -1)
        backpointers.append(bptrs_t)


    terminal_var = forward_var + self.transitions[self.label_map[self.STOP_TAG]]
    best_tag_id = torch.max(terminal_var, 1)[1].item()
    path_score = terminal_var[0][best_tag_id]
    best_path = [best_tag_id]
    for bptrs_t in reversed(backpointers):
        best_tag_id = bptrs_t[best_tag_id]
        best_path.append(best_tag_id)
    start = best_path.pop()
    assert start == self.label_map[self.START_TAG]  # Sanity check
    best_path.reverse()
    return path_score, best_path


def neg_log_likelihood(self, feats, tags, seq_len):
    forward_score = self._forward_alg(feats, seq_len)
    gold_score = self._score_sentence(feats, tags, seq_len)
    return torch.mean(forward_score - gold_score)


torch.manual_seed(3407)

embedding_size = 128
hidden_dim = 768
epochs = 50
batch_size = 128
device = "cuda:0" if torch.cuda.is_available() else "cpu"

vocab = get_vocab('train.json')
label_map = get_label_map('train.json')
train_dataset = Mydataset('train.json', vocab, label_map)
valid_dataset = Mydataset('dev.json', vocab, label_map)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True,
                              collate_fn=train_dataset.collect_fn)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, pin_memory=False, shuffle=False,
                              collate_fn=valid_dataset.collect_fn)
model = BiLSTM_CRF(embedding_size, hidden_dim, train_dataset.vocab, train_dataset.label_map, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


def train():
    total_start = time.time()
    best_score = 0
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        model.state = 'train'
        for step, (text, label, seq_len, sentence) in enumerate(train_dataloader, start=1):
            start = time.time()
            text = text.to(device)
            label = label.to(device)
            seq_len = seq_len.to(device)
            loss = model(text, seq_len, sentence, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        score = evaluate()
        if score > best_score:
            print(f'score increase:{best_score} -> {score}')
            best_score = score
            torch.save(model.state_dict(), './model.bin')
        print(f'current best score: {best_score}')


def evaluate():
    # model.load_state_dict(torch.load('./model1.bin'))
    all_label = []
    all_pred = []
    model.eval()
    model.state = 'eval'
    with torch.no_grad():
        for text, label, seq_len, sentence in tqdm(valid_dataloader, desc='eval: '):
            text = text.to(device)
            seq_len = seq_len.to(device)
            batch_tag = model(text, seq_len, sentence, label)
            all_label.extend(
                [[train_dataset.label_map_inv[t] for t in l[:seq_len[i]].tolist()] for i, l in enumerate(label)])
            all_pred.extend([[train_dataset.label_map_inv[t] for t in l] for l in batch_tag])


    all_label = list(chain.from_iterable(all_label))
    all_pred = list(chain.from_iterable(all_pred))
    sort_labels = [k for k in train_dataset.label_map.keys()]
    f1 = metrics.f1_score(all_label, all_pred, average='macro', labels=sort_labels[:-3])

    print(metrics.classification_report(
        all_label, all_pred, labels=sort_labels[:-3], digits=3
    ))
    return f1

train()
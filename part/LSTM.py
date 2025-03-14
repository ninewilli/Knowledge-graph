import torch
import torch.nn as nn
import json
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
Data = []

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

with open("part/train1.json",encoding='utf-8') as inputData:
    for line in inputData:
        try:
            Data.append(json.loads(line.rstrip(';\n')))
        except ValueError:
            print ("Skipping invalid line {0}".format(repr(line)))

Test_data = []
with open("part/valid1.json",encoding='utf-8') as inputData:
    for line in inputData:
        try:
            Test_data.append(json.loads(line.rstrip(';\n')))
        except ValueError:
            print ("Skipping invalid line {0}".format(repr(line)))

train_shape = []
train_relations = []
train_input = []
train_output = []
train_kg = []
for train_x in Data:
    train_shape.append(train_x['cate'])
    train_relations.append(train_x['instruction'])
    train_input.append(train_x['input'])
    train_output.append(train_x['output'])
    train_kg.append(train_x['kg'])

test_shape = []
test_relations = []
test_input = []
for test_x in Test_data:
    test_shape.append(test_x['cate'])
    test_relations.append(test_x['instruction'])
    test_input.append(test_x['input'])

train_relations_s = []
for i in range(len(train_relations)):
    num_index = 0
    vec_index = ""
    for j in range(len(train_relations[i])):
        if train_relations[i][j] == ']':
            num_index = 0
        if num_index == 1:
            vec_index+=train_relations[i][j]
        if train_relations[i][j] == '[':
            num_index = 1
    train_relations_s.append(vec_index)

test_relations_s = []
for i in range(len(test_relations)):
    num_index = 0
    vec_index = ""
    for j in range(len(test_relations[i])):
        if test_relations[i][j] == ']':
            num_index = 0
        if num_index == 1:
            vec_index+=test_relations[i][j]
        if test_relations[i][j] == '[':
            num_index = 1
    test_relations_s.append(vec_index)

#for train_text in train_input:
    #train_input_s.append(ps.lcut(train_text))

dict = {}
relation_index = 0
train_relations_df = []
for i in range(len(train_relations_s)):
    relation_s = ""
    num_index = 0
    vec_index = []
    for j in range(len(train_relations_s[i])):
        if train_relations_s[i][j] == "'":
            num_index = 1-num_index
            if relation_s != "":
                vec_index.append(relation_s)
                if relation_s not in dict:
                    dict[relation_s] = relation_index
                    relation_index+=1
            relation_s = ""
        elif num_index == 1:
            relation_s += train_relations_s[i][j]
    train_relations_df.append(vec_index)

dict_shape = {}
relation_index = 0
test_relations_df = []
for i in range(len(test_relations_s)):
    relation_s = ""
    num_index = 0
    vec_index = []
    for j in range(len(test_relations_s[i])):
        if test_relations_s[i][j] == "'":
            num_index = 1-num_index
            if relation_s != "":
                vec_index.append(relation_s)
                if relation_s not in dict:
                    dict[relation_s] = relation_index
                    relation_index+=1
            relation_s = ""
        elif num_index == 1:
            relation_s += test_relations_s[i][j]
    test_relations_df.append(vec_index)

START_TAG = "<START>"
STOP_TAG = "<STOP>"
target_main = []
index = 5
tag_to_ix = {"B-SUBJ": 0, "I-SUBJ": 1,"B-OBJ": 2 ,"I-OBJ": 3 , "O": 4 ,START_TAG: 5, STOP_TAG: 6}
for i in range(len(train_kg)):
    vec_index = []
    for k in range(len(train_input[i])):
        vec_index.append('O')
    for train_data in train_kg[i]:
        B_index = "B-SUBJ"
        I_index = "I-SUBJ"
        index_0 = 0
        index_2 = 0
        for k in range(len(train_input[i])):
            if index_0>=len(train_data[0]):
                break
            if train_input[i][k] == train_data[0][index_0]:
                if index_0 == 0:
                    vec_index[k] = B_index
                else:
                    vec_index[k] = I_index
                index_0+=1
            else:
                index_0 = 0
        B_index = "B-OBJ"
        I_index = "I-OBJ"
        for k in range(len(train_input[i])):
            if index_2>=len(train_data[2]):
                break
            if train_input[i][k] == train_data[2][index_2]:
                if index_2 == 0:
                    vec_index[k] = B_index
                else:
                    vec_index[k] = I_index
                index_2+=1
            else:
                index_2 = 0
    target_main.append(vec_index)


training_data = []
for i in range(len(target_main)):
    training_data.append((train_input[i],target_main[i]))

START_TAG = "<START>"
STOP_TAG = "<STOP>"
def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
class LSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(LSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size).to(device))
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2).to(device),

                torch.randn(2, 1, self.hidden_dim // 2).to(device))

    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

    def _viterbi_decode(self, feats):
        backpointers = []
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000.).to(device)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        forward_var = init_alphas
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1).to(device)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)

        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).to(device), tags.to(device)])
        for i, feat in enumerate(feats):
            score = (score.to(device) + self.transitions[tags[i + 1].to(device), tags[i].to(device)].to(device) + feat[tags[i + 1]].to(device)).to(device)
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score
EMBEDDING_DIM = index
HIDDEN_DIM = 4

def prepare_sequence(seq, to_ix):
    idxs = []
    for w in seq:
        if w not in to_ix:
            idxs.append(0)
        else:
            idxs.append(to_ix[w])
    return torch.tensor(idxs, dtype=torch.long)
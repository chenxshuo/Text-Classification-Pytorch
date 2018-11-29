import sys
import os
import time
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from models.LSTM import LSTMClassifier
from models.selfAttention import SelfAttention


TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset()
learning_rate = 2e-5
batch_size = 32
output_size = 2
hidden_size = 256
embedding_length = 300
model = SelfAttention(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
model.load_state_dict(torch.load("model_param.pkl"))
model.cuda()
test_sen1 = sys.argv[1]
test_sen1 = TEXT.preprocess(test_sen1)
test_sen1 = [[TEXT.vocab.stoi[x] for x in test_sen1]]
test_sen = np.asarray(test_sen1)
test_sen = torch.LongTensor(test_sen)
test_tensor = Variable(test_sen, volatile=True)
test_tensor = test_tensor.cuda()
model.eval()
output = model(test_tensor,1)
out = F.softmax(output, 1)
if (torch.argmax(out[0]) == 1):
    print ("Type: Finance")
else:
    print ("Type: Medical")

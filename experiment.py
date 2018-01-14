import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from gen_examples import gen_bad_examples, gen_good_examples
from gen_languages import gen_bad_examples1, gen_good_examples1, gen_examples2, gen_bad_examples3, gen_good_examples3
import random
import time
import sys

class LSTMModel(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        """
        LSTM Model ctor
        :param embedding_dim: Emb dim
        :param hidden_dim: Hidden dim
        :param vocab_size: Vocab size
        :param tagset_size: Target size
        """
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        """
        cleans hidden state
        :return:
        """
        return (torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        """
        LSTM fowarding
        :param sentence: input
        :return: tag scores
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores =  F.tanh(tag_space[-1])
        return tag_scores

def prepare_sequence(sentence, word2idx):
    """
    prepares sequence to index and long tensor variable
    :param sentence: input
    :param word2idx: word to index dictionary
    :return: input as variable
    """
    index_list = [word2idx[w] for w in sentence]
    tensor = torch.LongTensor(index_list)
    return torch.autograd.Variable(tensor)

def train(model, loss_function, optimizer, full_data_list, epoch_number, flag):
    """
    train
    :param model: model
    :param loss_function: loss function (binary)
    :param optimizer: optimizer
    :param full_data_list: sentence and tags
    :param epoch_number: epoch num
    :param flag: if "train" or "test"
    :return:
    """
    startTime = time.time()
    for epoch in range(epoch_number):
        acc_list = []
        loss_list = []
        for sentence, tags in full_data_list:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            if flag == "train":
                model.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Variables of word indices.
            sentence_in = prepare_sequence(sentence, Char2IdxDict)
            targets =  Variable(torch.FloatTensor(tags))

            # Step 3. Run our forward pass.
            tag_scores  = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            loss_list.append(loss.data[0])
            if flag == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc_list.append(1 if round(tag_scores.data[0]) == targets.data[0] else 0)

        print float(sum(acc_list))/len(full_data_list), float(sum(loss_list))/len(full_data_list), time.time() - startTime

if __name__ == "__main__":
    EPOCH = 20
    type = sys.argv[1]

    if type == "part1":
        Char2IdxDict = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "a": 10, "b": 11, "c": 12, "d": 0}
        Tag2IdxDict = {'0': 0, "1": 1}
        pos_examples = gen_good_examples()
        neg_examples = gen_bad_examples()
        pos_dev = gen_good_examples(examples_num=100)
        neg_dev = gen_bad_examples(examples_num=100)

    elif type == "palindromes":
        Char2IdxDict = {"0": 0, "1": 1}
        Tag2IdxDict = {'0': 0, "1": 1}
        pos_examples = gen_good_examples1()
        neg_examples = gen_bad_examples1()
        pos_dev = gen_good_examples1(examples_num=100)
        neg_dev = gen_bad_examples1(examples_num=100)

    elif type == "mid_zero":
        Char2IdxDict = {"0": 0, "1": 1}
        Tag2IdxDict = {'0': 0, "1": 1}
        pos_examples, neg_examples = gen_examples2()
        pos_dev, neg_dev = gen_examples2(examples_num=100)

    elif type == "same_edge":
        Char2IdxDict = {"0": 0, "1": 1}
        Tag2IdxDict = {'0': 0, "1": 1}
        pos_examples = gen_good_examples3()
        neg_examples = gen_bad_examples3()
        pos_dev = gen_good_examples3(examples_num=100)
        neg_dev = gen_bad_examples3(examples_num=100)

    else:
        print "input error"

    train_data_list = []
    for line in pos_examples:
        train_data_list.append((line, np.array([1])))
    for line in neg_examples:
        train_data_list.append((line, np.array([0])))
    random.shuffle(train_data_list)

    test_data_list = []
    for line in pos_dev:
        test_data_list.append((line, np.array([1])))
    for line in neg_dev:
        test_data_list.append((line, np.array([0])))
    random.shuffle(test_data_list)

    EMBEDDING_DIM = 6
    HIDDEN_DIM = 30

    model = LSTMModel(EMBEDDING_DIM, HIDDEN_DIM, len(Char2IdxDict),1)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print "train"
    train(model, loss_function, optimizer, train_data_list, EPOCH, "train")
    print "test"
    train(model, loss_function, optimizer, test_data_list, 1, "test")




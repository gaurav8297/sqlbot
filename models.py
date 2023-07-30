import torch.nn as nn
import math
from torch.autograd import Variable


class LSTMModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(LSTMModel, self).__init__()
        self.encoder = nn.Linear(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.nhid = nhid
        self.ninp = ninp
        self.nlayers = nlayers

        self.init_weights()

    def init_weights(self):
        initrange = 1
        encoder = self.encoder
        self.encoder.bias.data.fill_(0)
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        encoder.weight.data.normal_(0, math.sqrt(2. / (encoder.in_features + encoder.out_features)))

        decoder = self.decoder
        self.decoder.bias.data.fill_(0)
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        decoder.weight.data.normal_(0, math.sqrt(2. / (decoder.in_features + decoder.out_features)))

        self.rnn.weight_ih_l0.data.normal_(0, math.sqrt(2. / (self.ninp + self.nhid)))
        self.rnn.weight_ih_l0.data.normal_(0, math.sqrt(2. / (self.ninp + self.nhid)))
        self.rnn.bias_ih_l0.data.fill_(0)
        self.rnn.bias_hh_l0.data.fill_(0)

        self.rnn.weight_ih_l1.data.normal_(0, math.sqrt(2. / (self.ninp + self.nhid)))
        self.rnn.weight_ih_l1.data.normal_(0, math.sqrt(2. / (self.ninp + self.nhid)))
        self.rnn.bias_ih_l1.data.fill_(0)
        self.rnn.bias_hh_l1.data.fill_(0)

    def forward(self, input, hidden):
        bptt = input.size(0)
        bsz = input.size(1)

        input = input.reshape(bptt * bsz, -1)
        emb = self.encoder(input)
        emb = emb.view(bptt, bsz, -1)

        output, hidden = self.rnn(emb, hidden)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))


class KernelRegressionModel:
    def __init__(self):
        self.data = None


class LinearModel:
    def __init__(self):
        self.params = None
        self.ma_params = None

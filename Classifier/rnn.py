import torch
import torch.nn as nn
from torch.autograd import Variable

hidden_size_ = 15 # hyperparameter
num_layers = 4    # hyperparameter
p_dropout = 0.5  # hyperparameter

class RNN_model(nn.Module):
    hidden_size = hidden_size_
    def __init__(self, input_size):
        super(RNN_model, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        
        self.rnn = nn.LSTM(input_size, hidden_size_, num_layers, dropout = p_dropout)
        
    def forward(self, embeddings, lengths, hidden=None):
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, lengths.tolist(), batch_first=True)
        output, hidden = self.rnn(packed, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        idx = (lengths-1).view(-1,1).expand(output.size(0), output.size(2)).unsqueeze(1)
        
        #print(output)
        decoded = output.gather(1, Variable(idx)).squeeze()
        
        return decoded
        '''
        output, hidden = self.rnn(embeddings, hidden)
        idx = (lengths-1).view(-1,1).expand(output.size(0), output.size(2)).unsqueeze(1)
        decoded = output.gather(1, Variable(idx)).squeeze()
        return decoded
        '''
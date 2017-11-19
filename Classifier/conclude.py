from torch.autograd import Variable
import torch
import torch.utils.data
import torch.tensor
import torch.nn as nn
import torch.nn.functional as nn_func
import torch.optim as optim
import numpy as np
import math

class conclude(nn.Module):
    def __init__(self):
        self.number_of_classifiers = 3
        self.number_of_conclusions = 1
        super(conclude,self).__init__()
        self.fc1 = nn.Linear(self.number_of_classifiers,self.number_of_conclusions)

    def forward(self,flow):
        output_minftoinf = nn_func.relu(self.fc1(flow))
        output_0to1 = torch.nn.functional.sigmoid(output_minftoinf)
        output_tuple = torch.cat([1- output_0to1, output_0to1],1)
        return output_tuple

    def bind(self, rnn_input, cnn_input, nb_input):
        return torch.cat([rnn_input,cnn_input,nb_input],1)
        

"""conclude_ex = conclude()
batch_size_input = 10
t1 = Variable(0.25*torch.randn(batch_size_input,1))
t2 = Variable(0.25*torch.randn(batch_size_input,1))
t3 = Variable(0.25*torch.randn(batch_size_input,1))
print(t1.data.shape)
print(t2.data.shape)
print(t3.data.shape)
print(conclude_ex.infer(t1,t2,t3))
"""
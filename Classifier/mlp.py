import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class mlp(nn.Module):
    def __init__(self, FR_attribute_size, prediction_context_size):
        self.FR_attribute_size = FR_attribute_size
        self.output_size = prediction_context_size
        self.input_channels = self.FR_attribute_size + self.output_size
        self.hidden_layer_fc1 = 10
        self.output_channels = 1
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(self.input_channels, self.hidden_layer_fc1)
        self.fc2 = nn.Linear(self.hidden_layer_fc1, self.output_channels)

    def forward(self, flow):
        flow = F.relu(self.fc1(flow))
        flow = self.fc2(flow)
        return flow

    def getdata(self, FRlist, predict_variable):
        FR_attribute_list = []
        for FR in FRlist:
            FR_attribute_list.append(FR.get_attribute())

        FR_attribute_variable = Variable(torch.from_numpy(np.asarray(FR_attribute_list)).float()
                                     , requires_grad=True)
        out = torch.cat([FR_attribute_variable, predict_variable], 1)
        return out

"""
frlist = []
for i in range(10):
    frlist.append(frdummy.FRdummy())

predict_variable = Variable(torch.rand(10,60), requires_grad = True)
mlp_model = mlp(5,60) #model initialization
mlp_input = mlp.getdata(mlp_model, frlist, predict_variable)
    # input: FRlist, (prediction from context (torch variable))
mlp_output = mlp_model(mlp_input) # output : batch size * 1 torch variable
print(mlp_input.data.size())
print(mlp_output)
"""

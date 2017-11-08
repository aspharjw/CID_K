def cnn (formattedReviewList_input, cnn_model_input) :
    inference_cnn_numpy_array_list = cnn_model_input.infer(formattedReviewList_input)
    return inference_cnn_numpy_array_list

def Create_cnn_model (variable_input):
    wordvector_size_arg = variable_input.data.shape()[1]
    return cnn_model(wordvector_size_arg)

from torch.autograd import Variable
import torch
import torch.utils.data
import torch.tensor
import torch.nn as nn
import torch.nn.functional as nn_func
import torch.optim as optim
import numpy as np
import math
class ConvNet (nn.Module):
        def __init__(self,wordvector_size_input):
            input_channels = wordvector_size_input
            n_grams = 3 # must be odd number
            self.hidden_channel_conv1 = 50
            self.hidden_channel_conv2 = 10
            self.hidden_layer_fc1 = 5
            self.number_of_classes = 1
            self.output_vector_size = 60

            hidden_channel_conv1 = self.hidden_channel_conv1
            hidden_channel_conv2 = self.hidden_channel_conv2
            hidden_layer_fc1 = self.hidden_layer_fc1
            number_of_classes = self.number_of_classes
            output_vector_size = self.output_vector_size

            super(ConvNet,self).__init__()
            self.conv1 = nn.Conv1d(input_channels,hidden_channel_conv1,n_grams,padding=((n_grams-1)//2 ))
            self.batch1 = nn.BatchNorm1d(hidden_channel_conv1)
            self.conv2 = nn.Conv1d(hidden_channel_conv1,hidden_channel_conv2,n_grams,padding=((n_grams-1)//2))
            self.batch2 = nn.BatchNorm1d(hidden_channel_conv2)

            self.fc1 = nn.Linear(hidden_channel_conv2 , hidden_layer_fc1)
            self.fc2 = nn.Linear(hidden_layer_fc1, number_of_classes)

        def forward(self,flow):
            flow = flow.transpose(1,2) # nbatches * height * nchannels -> nbatches * nchannels * height
            mini_batch_size_here = flow.data.shape[0]
            number_of_words_here = flow.data.shape[2]
            flow = nn_func.relu(self.batch1(self.conv1(flow)))
            flow = nn_func.relu(self.batch2(self.conv2(flow)))
            flow = flow.transpose(1, 2).contiguous().view(-1, self.hidden_channel_conv2) # Does contiguous preserve graph relations between variables?
            flow = nn_func.relu(self.fc1(flow))
            flow = nn_func.relu(self.fc2(flow))
            flow = flow.view(mini_batch_size_here,number_of_words_here)
            variable_to_fixed_length_matrix = Variable(self.variable_to_fixed_length_matrix(number_of_words_here,self.output_vector_size))
            flow = torch.mm(flow ,variable_to_fixed_length_matrix)
            return flow

        def num_flat_features(self, x):
            size = x.size()[1:]  # all dimensions except the batch dimension
            num_features = 1
            for s in size:
                num_features *= s
            return num_features

        def variable_to_fixed_length_matrix(self,row,column):
            output_np = np.zeros((row,column))
            for i in range(column):
                index = (i+1) * row/(column )
                index_floor = math.floor(index)
                for j in range(0,index_floor):
                    output_np[j][i] = 1
                if (index != index_floor):
                    output_np[index_floor][i] = index - index_floor
            for k in range(row):
                index = 0
                flag = True
                for l in range(column):
                    if ((output_np[k][l] > 0) and flag):
                        index = l
                        flag = False
                if(output_np[k][index] == 1) :
                    for l in range(index+1,column):
                        output_np[k][l] = 0
                elif(output_np[k][index] < 1):
                    if(index+1 < column):
                        output_np[k][index+1] = 1- output_np[k][index]
                        if(index+2 < column):
                            for l in range (index+2, column):
                                output_np[k][l] = 0

            return torch.from_numpy(output_np).float()




class cnn_model :
    def __init__(self,wordvector_size_input):
        self.net = ConvNet(wordvector_size_input) #wordvector size 100
        self.word_vector_size =wordvector_size_input
    def train_net(self,formattedReviewList_input):
        self.criterion = nn.L1Loss() #ifnotdefined
        learning_rate = 0.001
        momentum_opt = 0.9
        self.optimizer = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=momentum_opt) #ifnotdefined
        number_of_loops_over_dataset = 2
        n_samples_per_mini_batch = 5
        print_per_n_minibatches = 10
        shuffle_training_dataset = False
        wordvector_size = self.word_vector_size
        max_words_for_string = self.max_words_for_string


        #making dataset, mini-batch
        for i, formattedReview in enumerate(formattedReviewList_input,0):
            #zero padding
            context_zeropadded = np.zeros([max_words_for_string, wordvector_size])
            if (formattedReview.context.shape[0] > 0):
                context_zeropadded[:formattedReview.context.shape[0],
                :formattedReview.context.shape[1]] = formattedReview.context
            # nSamples * nChannels * words form, make a tensor of it
            input = torch.unsqueeze(torch.transpose(torch.from_numpy(context_zeropadded).float(), 0, 1), 0)
            label = torch.FloatTensor([int(formattedReview.label)])
            if(i != 0):
                dataset_tensor = torch.cat((dataset_tensor,input), 0)
                targetset_tensor = torch.cat((targetset_tensor,label),0)
            else:
                dataset_tensor = input
                targetset_tensor = label

        print(dataset_tensor.shape)
        print(targetset_tensor.shape)
        dataset = torch.utils.data.TensorDataset(dataset_tensor,targetset_tensor)
        trainloader = torch.utils.data.DataLoader(dataset, batch_size = n_samples_per_mini_batch,shuffle =  shuffle_training_dataset, num_workers = 1,drop_last= True)

        for epoch in range(number_of_loops_over_dataset):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, train_data in enumerate(trainloader, 0):
                '''
                # get the inputs
                context_zeropadded = np.zeros([max_words_for_string, wordvector_size])
                if (formattedReview.context.shape[0] > 0):
                    context_zeropadded[:formattedReview.context.shape[0],
                    :formattedReview.context.shape[1]] = formattedReview.context
                input = torch.unsqueeze(torch.transpose(torch.from_numpy(context_zeropadded).float(), 0, 1), 0)
                label = torch.FloatTensor([int(formattedReview.label)])
                '''
                #get the inputs
                input, label = train_data
                # wrap them in Variable
                inputs, labels = Variable(input), Variable(label)

                # zero the parameter gradients
                self.optimizer.zero_grad()


                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.data[0]
                if i % print_per_n_minibatches == (print_per_n_minibatches-1):  # print every n mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / print_per_n_minibatches))
                    running_loss = 0.0

        print('Finished Training')

    def infer(self, minibatch_variable_input):
        '''
        minibatch_size = 20
        minibatch_tensor_list = self.mini_batch_from_formattedReview(minibatch_size,formattedReviewList_input)
        output_list = []

        for i, minibatch_tensor in enumerate(minibatch_tensor_list,0):
            output =  self.net(Variable(minibatch_tensor))
            output_numpy_array = output.data.numpy().reshape((minibatch_size))
            output_list.append(output_numpy_array)
        print(output_list)
        '''
        output = self.net(minibatch_variable_input)
        #print(output)
        return output
        #return output_list

    def mini_batch_from_formattedReview(self,mini_batch_size, formattedReviewList_input):
        mini_batch_list = []
        wordvector_size = self.word_vector_size
        max_words_for_string = self.max_words_for_string
        for i,formattedReview in enumerate(formattedReviewList_input,0):
            #zero padding
            context_zeropadded = np.zeros([max_words_for_string, wordvector_size])
            if(formattedReview.context.shape[0] > 0):
                context_zeropadded[:formattedReview.context.shape[0],:formattedReview.context.shape[1]] = formattedReview.context
            input = torch.unsqueeze(torch.transpose(torch.from_numpy(context_zeropadded).float(),0,1),0)
            label = torch.FloatTensor(int(formattedReview.label))
            #making a minibatch list

            if((i % mini_batch_size == mini_batch_size-1) or i >= len(formattedReviewList_input)):
                minibatch_tensor = torch.cat((minibatch_tensor, input), 0)
                mini_batch_list.append(minibatch_tensor)
            elif (i %mini_batch_size != 0):
                minibatch_tensor = torch.cat((minibatch_tensor, input), 0)
            else:
                minibatch_tensor = input
        return mini_batch_list

cnn_1 = cnn_model(100)
cnn_1.infer(Variable(torch.rand(1,36,100)))
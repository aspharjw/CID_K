import torch
import torch.nn as nn
from torch.autograd import Variable   
from torch import optim
import numpy as np
import sys
import os
import time

sys.path.append("../Preprocessor")
import format_module

import rnn
import naivebayesian
import cnn
import conclude
import mlp
import convert_to_excel

import pickle

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

class classifierModule(nn.Module):
    def __init__(self, input_size, batch_size, path, refresh = False):
        super(classifierModule, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        
        self.rnn_model = rnn.RNN_model(input_size)
        self.rnn_out_size = rnn.RNN_model.hidden_size
        self.rnn_mlp = mlp.mlp(format_module.FormattedReview.attribute_num, self.rnn_out_size)
        
        self.nb_model = naivebayesian.NaiveBayesianDB()
        self.nb_out_size = 1
        self.nb_mlp = mlp.mlp(format_module.FormattedReview.attribute_num, self.nb_out_size)
        
        self.cnn_model = cnn.ConvNet(input_size)
        self.cnn_out_size = cnn.ConvNet.output_vector_size
        self.cnn_mlp = mlp.mlp(format_module.FormattedReview.attribute_num, self.cnn_out_size)
        
        self.conclude = conclude.conclude()
        
        self.nb_path = "./models/nb_db.pkl"
        if os.path.exists(self.nb_path):
            self.nb_model = load_object(self.nb_path)
        elif os.path.exists("./Classifier/models/nb_db.pkl"):
            self.nb_path = "./Classifier/models/nb_db.pkl"
            self.nb_model = load_object(self.nb_path)
        else:
            try:
                FRlist = load_object("../Preprocessor/pkl/save_formatted_review_test.pkl")
            except FileNotFoundError:
                FRlist = load_object("./Preprocessor/pkl/save_formatted_review_test.pkl")
            self.nb_model.add_FRlist(FRlist) #initialize nb database
            
            if os.path.exists(self.nb_path):
                save_object(self.nb_model, self.nb_path)
            elif os.path.exists("./Classifier/models/"):
                self.nb_path = "./Classifier/models/nb_db.pkl"
                save_object(self.nb_model, self.nb_path)
                
            self.nb_model = load_object(self.nb_path)
            
        
        self.path = path
        
        if os.path.exists(path) and not refresh:
            self.load_state_dict(torch.load(path))
            
        else:
            self.save_state_dict()
        
    def save_state_dict(self):
        torch.save(self.state_dict(), self.path)
        save_object(self.nb_model, self.nb_path)
        
        
    def encoder(self, formattedList):
        length = len(formattedList)
        contextList = [formattedList[i].context for i in range(length)]

        lengths = torch.LongTensor([len(contextList[i]) for i in range(length)])
        max_len = torch.max(lengths)
        
        data = np.zeros((length, max_len, self.input_size))

        for i in range(length):
            context = contextList[i]
            if not (context.size == 0):
                data[i, :context.shape[0],:] = context
            else:
                lengths[i] = 1
            
        return self.sort_batch(torch.FloatTensor(data), formattedList, lengths)
        
    def sort_batch(self, context, formatted, seq_len):
        batch_size = context.size(0)
        sorted_seq_len, sorted_idx = seq_len.sort(0, descending = True)
        
        sorted_context = context[sorted_idx]
        sorted_formatted = [formatted[i] for i in sorted_idx]
        
        return Variable(sorted_context), sorted_formatted, sorted_seq_len
    
    def resize_input(self, input):
        list_ = list()
        for i in range(0, len(input), self.batch_size):
            list_.append(input[i:i+self.batch_size])
        return list_
        
    def forward(self, formatted_list, hidden=None, mode = "Default"):
        context, formatted, lengths = self.encoder(formatted_list)
        
        if mode == "rnn":
            rnn_out = self.rnn_model(context, lengths)
            rnn_mlp_out = self.rnn_mlp(self.rnn_mlp.getdata(formatted, rnn_out))
            output_0to1 = torch.nn.functional.sigmoid(rnn_mlp_out)
            return torch.cat([1- output_0to1, output_0to1],1)
            
        elif mode == "cnn":
            cnn_out = self.cnn_model(context)
            cnn_mlp_out = self.cnn_mlp(self.cnn_mlp.getdata(formatted, cnn_out))
            output_0to1 = torch.nn.functional.sigmoid(cnn_mlp_out)
            return torch.cat([1- output_0to1, output_0to1],1)
            
        elif mode == "nb":
            nb_out = self.nb_model.naive_bayes_FRlist(formatted)
            nb_mlp_out = self.nb_mlp(self.nb_mlp.getdata(formatted, nb_out))
            output_0to1 = torch.nn.functional.sigmoid(nb_mlp_out)
            return torch.cat([1- output_0to1, output_0to1],1)
            
        else:
            rnn_out = self.rnn_model(context, lengths)
            cnn_out = self.cnn_model(context)
            nb_out = self.nb_model.naive_bayes_FRlist(formatted)
            
            rnn_mlp_out = self.rnn_mlp(self.rnn_mlp.getdata(formatted, rnn_out))
            cnn_mlp_out = self.cnn_mlp(self.cnn_mlp.getdata(formatted, cnn_out))
            nb_mlp_out = self.nb_mlp(self.nb_mlp.getdata(formatted, nb_out))
            
            return self.conclude(self.conclude.bind(rnn_mlp_out, cnn_mlp_out, nb_mlp_out))
        
        '''
        print("rnn_out : ", rnn_out.size())
        print("cnn_out : ", cnn_out.size())
        print("nb_out : ", nb_out.size(), "\n")
        '''
        
    def print_contribution(self):
        (weight, bias) = self.conclude.get_contribution()
        print("----------------- Current model contribution ----------------")
        print("-- rnn : ", weight.data[0][0])
        print("-- cnn : ", weight.data[0][1])
        print("-- nb : ", weight.data[0][2])
        print("-- bias : ", bias.data[0])
        print("-------------------------------------------------------------\n")
        
        
learning_rate = 0.0013
input_size = 100  # word2vec k size
batch_size = 800
n_epochs = 1000

try:
    reviewDB = format_module.ReviewDB("../Preprocessor/pkl/train_1to1")
    model = classifierModule(input_size, batch_size, "./models/final_39.mdl")
except FileNotFoundError:
    reviewDB = format_module.ReviewDB("./Preprocessor/pkl/train_1to1")
    model = classifierModule(input_size, batch_size, "./Classifier/models/final_39.mdl")
    
format_module.FormattedReview.setDB(reviewDB)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# target이 0일 때, p가 1-s보다 작으면 +1
# target이 1일 때, p가 1-s보다 크면 +1
# -> (1-s-p)*(t-1/2) <= 0 일 때 +1
def get_accuracy(outputs, targets, sensitivity):
    result = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    t = targets.data.type(torch.FloatTensor)-0.5
    x = (1-sensitivity-outputs.data[:, 1])*t
    i = 0
    for y in x:
        if y < 0:
            result+=1
            if t[i] > 0: tp += 1
            else: tn += 1
        else:
            if t[i] > 0: fn += 1
            else: fp += 1
        i += 1
    return np.array([result, tp, tn, fp, fn])
    
def get_targets(input, model, out = [1, 0]):
    _, batch, _ = model.encoder(input)
    targets = list()
    for formatted in batch:
        if formatted.label:
            targets.append(out[0])
        else:
            targets.append(out[1])
    
    return Variable(torch.LongTensor(targets), requires_grad = False)

def get_prediction(outputs, sensitivity):
    return np.ceil(outputs.data[:, 1]+sensitivity-1+0.000000001)

def train_net(train_list, validation_list, sensitivity = 0.5, run_mode = "default"):
    batch_list = model.resize_input(train_list)
    
    for epoch in range(n_epochs):
        time_1 = time.time()
        
        tacc = np.array([0, 0, 0, 0, 0])
        vacc = np.array([0, 0, 0, 0, 0])

        for bl in batch_list:
            outputs = model(bl, mode = run_mode)
            #targets_ = get_targets(bl, model, [1, -1])
            targets = get_targets(bl, model)

            
            optimizer.zero_grad()    
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            tacc += get_accuracy(outputs, targets, sensitivity)

        tacc = tacc / len(train_list)
        
        time_2 = time.time()
        
        v_outputs = model(validation_list, mode = run_mode)
        v_targets = get_targets(validation_list, model)
        #v_targets_ = get_targets(validation_list, model, [1, -1])
        vacc = get_accuracy(v_outputs, v_targets, sensitivity) / len(validation_list)
        v_loss = criterion(v_outputs, v_targets)
        
        f1_train = 0.0
        f1_val = 0.0
        if tacc[1] != 0: f1_train = 2/( 1 + ((1-tacc[2])/tacc[1]) )
        if vacc[1] != 0: f1_val = 2/( 1 + ((1-vacc[2])/vacc[1]) )
        
        print("epoch{:>3}: {} s" .format(epoch, time_2-time_1) )
        print("--------- train_acc {:.6f} | ham_acc {:.6f} | spam_acc {:.6f} | f1_score {:.6f} | loss.data {:.6f}"
             .format(tacc[0], tacc[2]/(tacc[2]+tacc[3]), tacc[1]/(tacc[1]+tacc[4]), f1_train, loss.data[0]) )
        print("---- validation_acc {:.6f} | ham_acc {:.6f} | spam_acc {:.6f} | f1_score {:.6f} | loss.data {:.6f}"
             .format(vacc[0], vacc[2]/(vacc[2]+vacc[3]), vacc[1]/(vacc[1]+vacc[4]), f1_val, v_loss.data[0]) )
        
        #if epoch > 5 and np.mean(np.array(tacc_list[-6:-1])) < np.mean(np.array(vacc_list[-6:-1])):
        #    print("Seems like m1 starts to overfit, aborting training")
        #    break
        model.save_state_dict()
            
    print("Finished Training")
    
def inference(test_list, sensitivity = 0.5, run_mode = "Default"):
    outputs = model(test_list, mode = run_mode)
    targets = get_targets(test_list, model)
    prediction = get_prediction(outputs, sensitivity)
    
    result = []
    _, formatted, _ = model.encoder(test_list)
    
    i = 0
    for f in formatted:
        result.append( (reviewDB.get_review_node(f.review_id).data, outputs.data[i, 1], prediction[i]) )
        i += 1
    
    acc = get_accuracy(outputs, targets, sensitivity) / len(test_list)
    loss = criterion(outputs, targets)
        
    f1_val = 0.0
    if acc[1] != 0: f1_val = 2/( 1 + ((1-acc[2])/acc[1]) )

    print("\naccuracy {:.6f} | ham_acc {:.6f} | spam_acc {:.6f} | f1_score {:.6f} | loss.data {:.6f}"
             .format(acc[0], acc[2]/(acc[2]+acc[3]), acc[1]/(acc[1]+acc[4]), f1_val, loss.data[0]) )
    
    convert_to_excel.convert_to_excel(result, "./result.xlsx")

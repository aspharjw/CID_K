import pickle
import torch
from torch.autograd import Variable

class SpamCount(object):
    def __init__(self):
        self.spam_count = 0
        self.ham_count = 0

    def add_spam(self):
        self.spam_count += 1

    def add_ham(self):
        self.ham_count += 1

    def get_spam(self):
        return self.spam_count

    def get_ham(self):
        return self.ham_count

    def get_total(self):
        return self.spam_count + self.ham_count

    def __str__(self):
        return "spam/ham : " + str(self.spam_count) + "/" + str(self.ham_count)


class NaiveBayesianDB(object):
    def __init__(self):
        self.db = {}
        self.total_spam = 0
        self.total_ham = 0

    def add_postag(self, postag, spam_ham):
        if spam_ham == 'ham':
            self.total_ham += 1
            if postag in self.db:
                self.db[postag].add_ham()
            else:
                self.db[postag] = SpamCount()
                self.db[postag].add_ham()
        elif spam_ham == 'spam':
            self.total_spam += 1
            if postag in self.db:
                self.db[postag].add_spam()
            else:
                self.db[postag] = SpamCount()
                self.db[postag].add_spam()
        else:
            raise Exception('wrong input # 1')

    def add_FRlist(self, FRlist):
        for FR in FRlist:
            if FR.label == True:
                for postag in FR.context_bayes:
                    self.add_postag(postag, 'spam')
            elif FR.label == False:
                for postag in FR.context_bayes:
                    self.add_postag(postag, 'ham')

    def p_spam(self):
        return self.total_spam / (self.total_spam + self.total_ham)

    def p_ham(self):
        return self.total_ham / (self.total_spam + self.total_ham)

    # neglect since it applies to both spam / ham
    # def p_postag(self, postag):
    #    return self.db[postag].get_total / (self.total_spam + self.total_ham)

    # case of p(not postag) is never considered
    def p_postag_l_spam(self, postag, laplace):
        return (self.db[postag].get_spam() + 1) / (self.total_spam + laplace)

    def p_postag_l_ham(self, postag, laplace):
        return (self.db[postag].get_ham() + 1) / (self.total_ham + laplace)

    # laplace estimation is set as default
    def naive_bayes(self, postaglist):
        if self.total_ham == 0 & self.total_spam == 0 :
            raise Exception('no training data')
        elif self.total_ham == 0:
            return 1
        elif self.total_spam == 0:
            return 0

        laplace = len(postaglist)
        # ratio_spam p = a/b , where a = p_spam, b = p_ham
        ratio_spam = self.p_spam()
        ratio_spam /= self.p_ham()

        for postag in postaglist:
            if postag in self.db:
                ratio_spam *= self.p_postag_l_spam(postag, laplace)
                ratio_spam /= self.p_postag_l_ham(postag, laplace)

        # p_spam normalize = p_spam/(p_spam+p_ham) = 1/(1+1/p)
        p_spam = 1 / (1 + 1 / ratio_spam)
        return p_spam
        # print("p_spam : " + str(p_spam))

    def naive_bayes_FRlist(self, FRlist):
        it = 0
        v = Variable(torch.Tensor(len(FRlist), 1).zero_(), requires_grad=False)
        for FR in FRlist:
            v[it, 0] = self.naive_bayes(FR.context_bayes)
            it += 1

        print(v)
        return v

    def __str__(self):
        return_string = ""
        return_string = return_string + "total postag: " + str(
            self.total_spam + self.total_ham) + "\n" + "total spam:   " + str(
            self.total_spam) + "\n" + "total ham:    " + str(self.total_ham) + "\n"
        return_string = return_string + "total diff postag: " + str(len(self.db)) + "\n"
        it = 0
        for key in self.db:
            return_string = return_string + str(it) + ": " + key.__str__() + " " + self.db[key].__str__() + "\n"
            it += 1

        return return_string

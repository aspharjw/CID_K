import gensim
import numpy as np

word2vec_model_path = './models/namuwiki_testmodel'
word2vec_model = gensim.models.Word2Vec.load(word2vec_model_path)


class PreprocessedReview(object):
    def __init__(self, word2vec_model_input, BookingReview_input):

        self.company_postag = BookingReview_input.company # string list

        self.company_word2vec = []
        wordlist_iterator = iter(BookingReview_input.company)
        for word in wordlist_iterator:
            self.company_word2vec.append(word2vec_model_input.wv[word])  # wv is dictionary and word is string.
        self.company_word2vec = np.array(self.company_word2vec)

        self.context_postag = BookingReview_input.context # string list

        self.context_word2vec = []
        wordlist_iterator = iter(BookingReview_input.context)
        for word in wordlist_iterator:
            self.context_word2vec.append(word2vec_model_input.wv[word]) # wv is dictionary and word is string.
        self.context_word2vec = np.array(self.context_word2vec)

        self.id = BookingReview_input.id

        self.rate = BookingReview_input.rate

        self.post_time = BookingReview_input.post_time

        self.spam_ham = BookingReview_input.spam_ham

        self.review_id = BookingReview_input.review_id

        self.db_node = BookingReview_input.db_node


class BookingReview(object):

    def __init__(self, company, id, rate, context, post_time, spam_ham,review_id,db_node):

        self.company = company

        self.id = id

        self.rate = rate

        self.context = context

        self.post_time = post_time

        self.spam_ham = spam_ham
        self.review_id = review_id
        self.db_node = db_node

def BookingReview_list_to_PreprocessedReview_list(word2vec_model_input,BookingReview_list):
    output = []
    list_iterator = iter(BookingReview_list)
    for item_BookingReview in list_iterator:
        output.append(PreprocessedReview(word2vec_model_input,item_BookingReview))
    return output
"""
test = [BookingReview(['여담/Noun','으로/Josa'],'paradox',10,['세계/Noun', '수의/Noun', '미궁/Noun' ,'시리즈/Noun'],10,True,'Zeus','faker')]
#test2 = PreprocessedReview(word2vec_model,test)
test3 = BookingReview_list_to_PreprocessedReview_list(word2vec_model,test)
print(test3[0].company_word2vec)
"""
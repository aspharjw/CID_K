import gensim
import numpy as np

word2vec_model_path = './models/namuwiki_testmodel'
word2vec_model = gensim.models.Word2Vec.load(word2vec_model_path)


class PreprocessReview(object):
    def __init__(self, company_postag, company_word2vec, context_postag, context_word2vec,
                 id, rate, context, post_time, label, review_id): # argument 'context' why does it exist?
        self.company_postag = company_postag
        self.company_word2vec = company_word2vec
        self.context_postag = context_postag
        self.context_word2vec = context_word2vec

        self.id = id
        self.rate = rate
        self.post_time = post_time
        self.label = label
        self.review_id = review_id

        self.db_node = None

    def __str__(self):
        return ("PreprocessReview object {6} :\n"
                "  Company name _postagged = {0}\n"
                "  ID = {1}\n"
                "  Rating = {2}\n"
                "  Context _postagged = {3}\n"
                "  Post time = {4}\n"
                "  Spam/Ham = {5}\n"
                "  Company name _vectorized = {7}\n"
                "  Context _vectorized = {8}\n"
                .format(self.company_postag, self.id, self.rate,
                    self.context_postag, self.post_time, self.label,
                    self.review_id,self.company_word2vec,self.context_word2vec))
    """
class PreprocessReview(object):
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

"""
class BookingReview(object):
    def __init__(self, company, id, rate, context, post_time, label,
                 review_id):
        self.company = company
        self.id = id
        self.rate = rate
        self.context = context
        self.post_time = post_time
        self.label = label
        self.review_id = review_id

        self.db_node = None

    def __str__(self):
        return ("BookingReview object {6}:\n"
                "  Company name = {0}\n"
                "  ID = {1}\n"
                "  Rating = {2}\n"
                "  Current Context = {3}\n"
                "  Post time = {4}\n"
                "  Spam/Ham = {5}\n"
                .format(self.company, self.id, self.rate,
                        self.context, self.post_time, self.label,
                        self.review_id))

    def __lt__(self, cmp):
        if (self.id > cmp.id):
            return False
        elif (self.id < cmp.id):
            return True
        elif (self.post_time > cmp.post_time):
            return False
        else:
            return True

def BookingReview_to_PreprocessReview(word2vec_model_input, BookingReview_input):

    # Company word2vec
    company_word2vec_output = []
    wordlist_iterator = iter(BookingReview_input.company)
    for word in wordlist_iterator:
        company_word2vec_output.append(word2vec_model_input.wv[word])  # wv is dictionary and word is string.
    company_word2vec_output = np.array(company_word2vec_output)

    # Context word2vec
    context_word2vec_output = []
    wordlist_iterator = iter(BookingReview_input.context)
    for word in wordlist_iterator:
        context_word2vec_output.append(word2vec_model_input.wv[word])  # wv is dictionary and word is string.
    context_word2vec_output = np.array(context_word2vec_output)

    return PreprocessReview(BookingReview_input.company,company_word2vec_output,BookingReview_input.context,context_word2vec_output,
                            BookingReview_input.id,BookingReview_input.rate,BookingReview_input.context,
                            BookingReview_input.post_time,BookingReview_input.label,BookingReview_input.review_id)

def BookingReview_list_to_PreprocessReview_list(word2vec_model_input,BookingReview_list):
    output = []
    list_iterator = iter(BookingReview_list)
    for item_BookingReview in list_iterator:
        output.append(BookingReview_to_PreprocessReview(word2vec_model_input,item_BookingReview))
    return output

test = [BookingReview(['여담/Noun','으로/Josa'],'paradox',10,['세계/Noun', '수의/Noun', '미궁/Noun' ,'시리즈/Noun'],10,True,'Zeus')]
#test2 = PreprocessedReview(word2vec_model,test)
test3 = BookingReview_list_to_PreprocessReview_list(word2vec_model,test)
print(test3[0])

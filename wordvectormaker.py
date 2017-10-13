import gensim
import numpy as np

class PreprocessReview(object):
    def __init__(self, company_postag, company_word2vec, context_postag, context_word2vec,
                 id, rate, post_time, label, review_id): # argument 'context' why does it exist?
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
                    self.review_id,self.company_word2vec,self.context_word2vec)) # [0][0] for observation.. later need revision

def word2vec_to_PreprocessReview(word2vec_model_input, PreprocessReview_input):

    # Company word2vec
    company_word2vec_output = []
    wordlist_iterator = iter(PreprocessReview_input.company_postag)
    for word in wordlist_iterator:
        company_word2vec_output.append(word2vec_model_input.wv[word])  # wv is dictionary and word is string.
    company_word2vec_output = np.array(company_word2vec_output)

    # Context word2vec
    context_word2vec_output = []
    wordlist_iterator = iter(PreprocessReview_input.context_postag)
    for word in wordlist_iterator:
        context_word2vec_output.append(word2vec_model_input.wv[word])  # wv is dictionary and word is string.
    context_word2vec_output = np.array(context_word2vec_output)

    return PreprocessReview(PreprocessReview_input.company_postag,
                            company_word2vec_output,
                            PreprocessReview_input.context_postag,
                            context_word2vec_output,
                            PreprocessReview_input.id,
                            PreprocessReview_input.rate,
                            PreprocessReview_input.post_time,
                            PreprocessReview_input.label,
                            PreprocessReview_input.review_id)

def word2vec_to_PreprocessReview_list(word2vec_model_input,PreprocessReview_input_list):
    output = []
    list_iterator = iter(PreprocessReview_input_list)
    for item_PreprocessReview_input in list_iterator:
        output.append(word2vec_to_PreprocessReview(word2vec_model_input,item_PreprocessReview_input))
    return output


# TEST CODE

word2vec_model_path = './models/namuwiki_testmodel'
word2vec_model = gensim.models.Word2Vec.load(word2vec_model_path)
test = [PreprocessReview(['여담/Noun','으로/Josa'],None,['세계/Noun', '수의/Noun', '미궁/Noun' ,'시리즈/Noun'],None,'Hello',10,100,True,678927),PreprocessReview(['밀레니엄/Noun', '의/Josa'],None,['던전/Noun', '크/Verb', '롤러/Noun'],None,'world',100,1730,False,67227)]
#test2 = PreprocessedReview(word2vec_model,test)
test3 = word2vec_to_PreprocessReview_list(word2vec_model,test)
print(test3[0])
print(test3[1])
print(test3)

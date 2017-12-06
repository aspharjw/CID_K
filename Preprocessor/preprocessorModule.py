import format_module
import bookingreview
import preprocessreview
import wordvectormaker
import postag_module
import gensim
import xl_to_br_module
#sys.setrecursionlimit(20000)

import _pickle
import gc

def save_object_split(obj, filename, ext, split_size = 1000):
    index = 0
    for i in range(0, len(obj), split_size):
        with open(filename + "_p" + str(index) + "." + ext, 'wb') as output:
            _pickle.dump(obj[i:i+split_size], output)
            index += 1
            mem()
            gc.collect()
                
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        _pickle.dump(obj, output)

def load_object(filename):
    with open(filename, 'rb') as input:
        return _pickle.load(input)
    
def load_object_split(filename, ext, last_index):
    for index in range(last_index+1):
        with open(filename + "_p" + str(index) + "." + ext, 'rb') as input:
            if(index == 0):
                out = _pickle.load(input)
            else:
                out.extend(_pickle.load(input))
    
    return out

def postagModule(xl_name, reviewDB, mode = "komoran"):
    BRList = xl_to_br_module.xl_to_BookingReview(xl_name)
    
    print("adding review to DB...")
    reviewDB.add_review_list(BRList)
    
    if(mode == "twitter"):
        PRList = postag_module.twitter(BRList)
    else:
        PRList = postag_module.komoran(BRList)
        mode = "komoran"
    '''save_object(PRList_twitter, "save_PR_twitter.pkl")
    save_object(PRList_komoran, "save_PR_komoran.pkl")'''
    
    return PRList

def embeddingModule(PRList):
    try:
        word2vec_model = gensim.models.Word2Vec.load('./models/namuwiki_testmodel_Komoran.model')
    except FileNotFoundError:
        word2vec_model = gensim.models.Word2Vec.load('./Preprocessor/models/namuwiki_testmodel_Komoran.model')
    print("processing word embedding...")
    return wordvectormaker.word2vec_to_PreprocessReview_list(word2vec_model, PRList)

def formattingModule(PRList, reviewDB, version):
    print("processing formatted review...")
    formatted_list = [format_module.FormattedReview(review) for review in PRList]
    '''
    print("saving formatted review...")
    try:
        save_object(formatted_list, "./pkl/save_formatted_review_" + version +".pkl")
    except FileNotFoundError:
        save_object(formatted_list, "./Preprocessor/pkl/save_formatted_review_" + version +".pkl")
    '''
    return formatted_list

def preprocessModule(xl_name, reviewDB, version):
    PRList = postagModule(xl_name, reviewDB)
    embeddingModule(PRList)
    formatted_list = formattingModule(PRList, reviewDB, version)
    
    return formatted_list
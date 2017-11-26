import numpy as np
import re
from konlpy.tag import _komoran as Komoran
#from postag import Postag
from stemming import stemming_and_normalizing_word

postagger_jw = Komoran()

def word_part_of_word_slash_tag (word):
    word_slash_tag = re.findall(r"[\w']+|[/]",  word)
    word = word_slash_tag[0]
    return word

def postag_and_reformat_wordlist (word_list):
    postag_tuple_list = []
    for word in word_list:
        postag_tuple_list = postag_tuple_list + postagger_jw.postag_komoran(word)# Trust that this postagger correctly outputs list of tuples of (text, tag)

    output = []
    for tup in postag_tuple_list:
        adder = "{}/{}".format(tup[0], tup[1])
        output.append(adder)
    return output

def filter_words_with_vocab(word2vec_model_input,wordlist_iterator):
    word_list_retry = list(filter(lambda x: x in word2vec_model_input.wv.vocab, wordlist_iterator))
    retry_wordvector_list = [word2vec_model_input.wv[retry_word] for retry_word in word_list_retry]
    return (retry_wordvector_list,word_list_retry)

def wordlist_to_wordvec_list (word2vec_model_input, wordlist_iterator):
    word2vec_output = []
    for word in wordlist_iterator:
        try:
            wordvector = word2vec_model_input.wv[word]
            word2vec_output.append(wordvector)  # wv is dictionary and word is string.
        except KeyError:
            word_part = word_part_of_word_slash_tag(word)
            #cohesiontokenized_word_part = postagger_jw.tokenizer(word_part) #tokenizing
            stemmed_word_list = stemming_and_normalizing_word(word_part) #stemming/normalizing
            postagged_tokens_list_of_word = postag_and_reformat_wordlist(stemmed_word_list) #postagging into vocablist
            (wordvectorlist_retry,wordlist_retry) = filter_words_with_vocab(word2vec_model_input,postagged_tokens_list_of_word)
            word2vec_output = word2vec_output + wordvectorlist_retry

            # notin vocab printer
            print()
            #print("tokenized : ", end='')
            #print(cohesiontokenized_word_part)
            print("stemmed :  ", end='')
            print(stemmed_word_list)
            print("postagged : " , end = '')
            print(postagged_tokens_list_of_word)
            print(word + " not in vocabulary so replaced to " , end='')
            print((wordlist_retry))

    word2vec_output = np.array(word2vec_output)
    return word2vec_output

def word2vec_to_PreprocessReview(word2vec_model_input, PreprocessReview_input):

    # Company word2vec
    wordlist_iterator = iter(PreprocessReview_input.company_postag)
    company_word2vec_output = wordlist_to_wordvec_list(word2vec_model_input,wordlist_iterator)

    # Context word2vec
    wordlist_iterator = iter(PreprocessReview_input.context_postag)
    context_word2vec_output = wordlist_to_wordvec_list(word2vec_model_input,wordlist_iterator)

    PreprocessReview_input.company_word2vec = company_word2vec_output
    PreprocessReview_input.context_word2vec = context_word2vec_output
    return PreprocessReview_input

def word2vec_to_PreprocessReview_list(word2vec_model_input,PreprocessReview_input_list):
    output = []
    list_iterator = iter(PreprocessReview_input_list)
    for item_PreprocessReview_input in list_iterator:
        output.append(word2vec_to_PreprocessReview(word2vec_model_input,item_PreprocessReview_input))
    return output


# TEST CODE
'''
word2vec_model_path = './models/namuwiki_testmodel_Komoran.model'
word2vec_model = gensim.models.Word2Vec.load(word2vec_model_path)
test = [PreprocessReview.PreprocessReview(['일더하기일은귀요미','으로/JC'],None,['이더하기이는구이요미', '삼더하기삼도구이요미NNG', '사더하기사도귀요미' ,'오더하기오도귀요미'],None,'Hello',10,100,True,678927,None),
        PreprocessReview.PreprocessReview(['밀레니엄/NNG', 'ㄹㅇ루다가개노답인거인정?어인정~'],None,['육더하기육도귀요미', '칠더하기칠도귀요미', '길가다가은행밟았어ㅠㅠ너무시러'],None,'world',100,1730,False,67227,None)]
test3 = word2vec_to_PreprocessReview_list(word2vec_model,test)
print(test3[0].context_word2vec.shape)
print(test3[1].context_word2vec.shape)
'''


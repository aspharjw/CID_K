GOAL :
a list of PreprocessReview() # without company_word2vec , context_word2vec
into
a list of PreprocessReview() # with company_word2vec, context_word2vec

How to achieve Goal :
import wordvectormaker.py
word2vec_model = gensim.models.Word2Vec.load('word2vec_model_path')
word2vec_to_PreprocessReview_list(word2vec_model_input,BookingReview_list)

What you need :
gensim
numpy
word2vec model made before
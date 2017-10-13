import gensim, logging, os

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname),encoding='utf-8'):
                yield line.split()

def word2vec_model_maker(corpus_path, model_save_path):
    sentences = MySentences(corpus_path) # a memory-friendly iterator
    model = gensim.models.Word2Vec(sentences)
    model.save(model_save_path)

# Test code
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
corpus_path_param = './corpus'
model_save_path_param = './models/namuwiki_testmodel'
word2vec_model_maker(corpus_path_param, model_save_path_param)
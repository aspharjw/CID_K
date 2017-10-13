import gensim, logging, os
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

corpus_path = './corpus'
model_save_path = './models/namuwiki_testmodel'

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname),encoding='utf-8'):
                yield line.split()

sentences = MySentences(corpus_path) # a memory-friendly iterator
model = gensim.models.Word2Vec(sentences)
model.save(model_save_path)
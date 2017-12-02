from konlpy.tag import *
import sys
import re
sys.path.append('../Preprocessor/soy/')
from soy.nlp.tokenizer import RegexTokenizer, LTokenizer, MaxScoreTokenizer
import pickle

tests = [
    'ㅋㅋㅋㅋㅋㅋㅋ맛맛맛맛있는사사탕탕abbcc',
    'ㅋㅋ신림점',
    '좋겟어여',
    '알바진짜싸가지드럽게없네욬ㅋ주인은착하다뭐다하는데',
    '술좀사줘'
]

class Postag:
    def __init__(self):
        normalize_reg_pattern = '[^ a-zA-Z0-9.!?ㄱ-ㅣ가-힣\n]+'
        self.normalize_reg = re.compile(normalize_reg_pattern)
        self.regex_tokenizer = RegexTokenizer()
        self.duplicate_reg_pattern = '(([^.])\\2+)'
        f = open('../Preprocessor/pkl/krwordrank_data.pkl', 'rb')
        scores = pickle.load(f)
        f.close()
        self.maxscore_tokenizer = MaxScoreTokenizer(scores=scores)
        self.komoran = Komoran()
        print("=> Postag initiated")

# string -> postag list
    def tokenizer (self, text):
        # 불필요한 문자 제거
        text = self.normalize_reg.sub('', text)

        # 언어의 종류가 바뀌는 부분을 띄워준다
        text = ' '.join(self.regex_tokenizer.tokenize(text))

        # 같은 문자의 반복을 띄운다
        duplicate_list = re.findall(self.duplicate_reg_pattern, text)
        for word, char in duplicate_list:
            replace_word = word.replace(char, char+' ').rstrip()
            text = text.replace(word, replace_word)

        # MaxScoreTokenizer
        text = ' '.join(self.maxscore_tokenizer.tokenize(text))
        return text

    def postag_komoran (self, text):
        return self.komoran.pos(text)

'''
postag = Postag()
for test in tests:
    test = postag.tokenizer(test)
    print(postag.postag_komoran(test))
'''

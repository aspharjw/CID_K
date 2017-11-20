from konlpy.tag import *
import sys
import re
sys.path.append('./soy/')
from soy.nlp.tokenizer import RegexTokenizer, LTokenizer, MaxScoreTokenizer

'''
tests = [
    'ㅋㅋㅋㅋㅋㅋㅋ맛맛맛맛있는사사탕탕abbcc',
    'ㅋㅋ신림점',
    '좋겟어여',
    '알바진짜싸가지드럽게없네욬ㅋ주인은착하다뭐다하는데',
    '색다른곳을찾다가퓨전한정식집이나오길래봤더니괜찮아보이더라구요.그래서남친이랑다녀왔는데음식이너무멋지게(?)나오고인테리어도굉장히예쁜곳이라서너무괜찮았어요.연예인들이많이가는집이라고하더니그럴만하더라구요.음식최고예요~'
]
'''

# string -> postag list
def postag_komoran (text):
    #불필요한 문자 제거
    reg_pattern = '[^ a-zA-Z0-9.!?ㄱ-ㅣ가-힣\n]+'
    kreng = re.compile(reg_pattern)
    text = kreng.sub('', text)

    #언어의 종류가 바뀌는 부분을 띄워준다
    regtoken = RegexTokenizer()
    text = ' '.join(regtoken.tokenize(text))

    #같은 문자의 반복을 띄운다
    reg_pattern = '(([^.])\\2+)'
    repeat_list = re.findall(reg_pattern, text)
    for word, char in repeat_list:
        replace_word = word.replace(char, char+' ').rstrip()
        text = text.replace(word, replace_word)

    #konlpy 실행
    komoran = Komoran()
    return komoran.pos(text)

'''
komoran = Komoran()
for test in tests:
    print("전")
    print(komoran.pos(test))
    print("후")
    print(postag_komoran(test))
'''

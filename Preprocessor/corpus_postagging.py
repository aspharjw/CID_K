# import modules & set up logging
#-*-coding utf8-*-
import re
from konlpy.tag import Twitter

tagger = Twitter()
inputtxtpath = './text/namuwiki_articles.txt'
outputtxtpath = './corpus/namuwiki_articles_postagged.txt'
extraction_format_re = '[^ ㄱ-ㅣ가-힣\n]+'

def flat(content):
    return ["{}/{}".format(word, tag) for word, tag in tagger.pos(content)]

f1 = open(inputtxtpath, 'r',encoding='utf-8')
f2 = open(outputtxtpath, 'w',encoding='utf-8')

kreng = re.compile(extraction_format_re)

while True:
    line = f1.readline()
    if not line: break
    line_kreng = kreng.sub('',line)
    f2.write(' '.join(flat(line_kreng)) + '\n')

f1.close()
f2.close()
# import modules & set up logging
#-*-coding utf8-*-
import re
from konlpy.tag import Twitter


def flat(content,tagger):
    return ["{}/{}".format(word, tag) for word, tag in tagger.pos(content)]

def corpus_postagger(inputtxtpath, outputtxtpath, extraction_format_re,tagger):
    f1 = open(inputtxtpath, 'r',encoding='utf-8')
    f2 = open(outputtxtpath, 'w',encoding='utf-8')
    kreng = re.compile(extraction_format_re)
    while True:
        line = f1.readline()
        if not line: break
        line_kreng = kreng.sub('',line)
        f2.write(' '.join(flat(line_kreng,tagger)) + '\n')
    f1.close()
    f2.close()

# Test code

tagger_param = Twitter()
inputtxtpath_param = './text/test3.txt'
outputtxtpath_param = './corpus/out3.txt'
extraction_format_re_param = '[^ a-zA-Z0-9.!?ㄱ-ㅣ가-힣\n]+'
corpus_postagger(inputtxtpath_param,outputtxtpath_param,extraction_format_re_param,tagger_param)

import glob
import json
import sys
import pickle
sys.path.append('./soy/')
from Preprocessor.soy.soy.nlp.hangle import normalize
from Preprocessor.soy.soy.nlp.extractors import KR_WordRank
from xlrd import open_workbook
from Preprocessor.soy.soy.nlp.hangle import normalize

file_dir = 'Commonreviews_snuproject.xlsx'

def xl_to_context_list(file_dir):
    wb = open_workbook(file_dir)
    for sheet in wb.sheets():
        number_of_rows = sheet.nrows
        number_of_columns = sheet.ncols

        textList = []

        for row in range(1, number_of_rows):
            textList.append(normalize(str(sheet.cell(row,3).value),
                                      english=True, number=True))

    return textList

textlist = xl_to_context_list(file_dir)

'''
f = open('./text_list_corpus.pkl', 'wb')
pickle.dump(xl_context_list(file_dir), f)
f.close()

f = open('./text_list_corpus.pkl', 'rb')
textlist = pickle.load(f)
f.close()
'''

min_count = 5
max_length = 10
kr_wordrank = KR_WordRank(min_count, max_length)

beta = 0.85
max_iter = 10
verbose = True
keywords, rank, graph = kr_wordrank.extract(textlist, beta, max_iter, verbose)

f = open('./krwordrank_data.pkl', 'wb')
pickle.dump(keywords, f)
f.close()

#for word, r in sorted(keywords.items(), key=lambda x:x[1], reverse=True) :
#    print('%8s:\t%.4f' % (word, r))

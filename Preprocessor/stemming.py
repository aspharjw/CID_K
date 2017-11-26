from konlpy.tag import _twitter as twitter
twitter_postagger = twitter.Twitter()

def stemming_and_normalizing_word (word_in):
    normed_stemmed_word_list_withtag = twitter_postagger.pos(word_in,norm=True,stem=True)
    output = []
    for tup in normed_stemmed_word_list_withtag:
        output.append(tup[0])
    return output
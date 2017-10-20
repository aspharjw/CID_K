class PreprocessReview(object):
    def __init__(self, company_postag, company_word2vec, context_postag, context_word2vec,
                 id, rate, post_time, label, review_id, db_node):
        self.company_postag = company_postag
        self.company_word2vec = company_word2vec
        self.context_postag = context_postag
        self.context_word2vec = context_word2vec
        self.id = id
        self.rate = rate
        self.post_time = post_time
        self.label = label
        self.review_id = review_id
        self.db_node = db_node

    def __str__(self):
        return ("PreprocessReview object:\n"
                "  ID = {0}\n"
                "  Rating = {1}\n"
                "  Post time = {2}\n"
                "  Label = {3}\n"
                "  Review id = {4}\n"
                "  DB node = {5}\n"
                "  Context postag = {6}\n"
                "  Company postag = {7}\n"
                "  Context Word2Vec = {8}\n"
                "  Company Word2Vec = {9}\n"
                .format(self.id, self.rate, self.post_time, self.label,
                        self.review_id, self.db_node, self.context_postag,
                        self.company_postag, self.context_word2vec,
                        self.company_word2vec))




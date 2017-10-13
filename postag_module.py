from konlpy.tag import *
from preprocessreview import PreprocessReview

def komoran(BRList):
    komoran = Komoran()

    PRList = []

    for BR in BRList:
        company_postag = komoran.pos(BR.company)
        flat_company_postag = []
        for tup in company_postag:
            adder = "{}/{}".format(tup[0], tup[1])
            flat_company_postag.append(adder)

        context_postag = komoran.pos(BR.context)
        flat_context_postag = []
        for tup in context_postag:
            adder = "{}/{}".format(tup[0], tup[1])
            flat_context_postag.append(adder)

        PR = PreprocessReview(flat_company_postag, None, flat_context_postag, None,
                                BR.id, BR.rate, BR.post_time, BR.label,
                                BR.review_id, BR.db_node)
        PRList.append(PR)

    return PRList

def twitter(BRList):
    twitter = Twitter()

    PRList = []

    for BR in BRList:
        company_postag = twitter.pos(BR.company)
        flat_company_postag = []
        for tup in company_postag:
            adder = "{}/{}".format(tup[0], tup[1])
            flat_company_postag.append(adder)

        context_postag = twitter.pos(BR.context)
        flat_context_postag = []
        for tup in context_postag:
            adder = "{}/{}".format(tup[0], tup[1])
            flat_context_postag.append(adder)

        PR = PreprocessReview(flat_company_postag, None, flat_context_postag, None,
                                BR.id, BR.rate, BR.post_time, BR.label,
                                BR.review_id, BR.db_node)
        PRList.append(PR)

    return PRList

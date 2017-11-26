from preprocessreview import PreprocessReview
from postag import Postag

def komoran(BRList):
    postag = Postag()

    PRList = []

    count = 0
    for BR in BRList:
        if(count % 10000 == 0):
            print("postagging [komoran] " + str(count) + " ith unit...")
        count += 1
        company_postag = postag.postag_komoran(postag.tokenizer(BR.company))
        flat_company_postag = []
        for tup in company_postag:
            adder = "{}/{}".format(tup[0], tup[1])
            flat_company_postag.append(adder)

        context_postag = postag.postag_komoran(postag.tokenizer(BR.context))
        flat_context_postag = []
        for tup in context_postag:
            adder = "{}/{}".format(tup[0], tup[1])
            flat_context_postag.append(adder)

        PR = PreprocessReview(flat_company_postag, None, flat_context_postag, None,
                                BR.id, BR.rate, BR.post_time, BR.label,
                                BR.review_id)
        PRList.append(PR)

    return PRList
'''
def twitter(BRList):
    twitter = Twitter()

    PRList = []

    count = 0
    for BR in BRList:
        if(count % 10000 == 0):
            print("postagging [twitter] " + str(count) + " ith unit...")
        count += 1
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
'''

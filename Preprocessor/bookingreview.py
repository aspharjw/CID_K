class BookingReview(object):
    def __init__(self, company, id, rate, context, post_time, label,
                 review_id):
        self.company = company
        self.id = id
        self.rate = rate
        self.context = context
        self.post_time = post_time
        self.label = label
        self.review_id = review_id
        self.db_node = None

    def __str__(self):
        return ("BookingReview object:\n"
                "  Company name = {0}\n"
                "  ID = {1}\n"
                "  Rating = {2}\n"
                "  Context = {3}\n"
                "  Post time = {4}\n"
                "  Label = {5}\n"
                "  Review id = {6}\n"
                .format(self.company, self.id, self.rate,
                        self.context, self.post_time, self.label,
                        self.review_id))

    def __lt__(self, cmp):
        if(self.id > cmp.id):
            return False
        elif(self.id < cmp.id):
            return True
        elif(self.post_time > cmp.post_time):
            return False
        else:
            return True

# CID_K
===================
2017 SNU Creative Integrated Design 1 <br />
Team K <br />
spam filtering project <br />

Preprocess
----------------
## 1. Interface

### Booking_Review 
```
class BookingReview
- company : string
- id : string
- rate : int
- context : string
- post_time : float
- spam_ham : boolean
- review_id : int
- db_node : ReviewNode
```
-> output of module XlToBookingModule, input for module PostagModule

### PreProcessReview
```
class PreProcessReview
- company_postag : (postaged output)
- company_word2vec : n*k array
- context_postag : (postaged output)
- context_word2vec : m*k array
- id : string
- rate : int
- post_time : float
- spam_ham : boolean
- review_id : int
- db_node : ReviewNode
```

-> output of module PostagModule, input for module Word2VecModule <br />
-> output of module Word2VecModule, input for module FormattingModule

### FormattedReview
```
class FormattedReview
- context : n*k numpy array
- comp_similarity : float
- rate : float
- reiteration_context : float
- reiteration_repeat : float
- post_time : float
- post_vip : float
- id : float
- label : bool
- review_id : int
```

-> output of module FormattingModule

### ReviewDB
```
class ReviewDB(object):
    def __init__(self):
        self.review_dict = {}
        self.review_tree = ReviewTree()
    
    def add_review_list(self, bookingReview_list):
        ...
                
    def add_review(self, bookingReview):
        ...
        
    ...
```

-> call add_review in module XlToBookingModule.

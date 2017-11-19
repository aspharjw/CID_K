import numpy as np
from scipy import spatial
import bookingreview
import preprocessreview


class ReviewNode:
    def __init__(self, val, parent):
        self.val = val
        self.leftChild = None
        self.rightChild = None
        self.parent = parent
        
        val.db_node = self
    
    def get(self):
        return self.val
    
    def set(self, val):
        self.val = val
        
    def max_value(self):
        current = self
        
        while(current is not None):
            if current.rightChild is None:
                break
            current = current.rightChild
            
        return current
    
    def previous_node(self, num = 1):
        if(num == 0):
            return self
        
        if self.leftChild is not None:
            return self.leftChild.max_value().previous_node(num-1)
        
        n = self
        p = self.parent
        while(p is not None):
            if n != p.leftChild:
                break
            n = p
            p = p.parent
            
        if p is None:
            return None
        
        return p.previous_node(num-1)

    def __str__(self):     
        return "ReviewNode - "+self.val.__str__() + '\n'
    
        
class ReviewTree:
    def __init__(self):
        self.root = None

    def set_root(self, val):
        self.root = ReviewNode(val, None)

    def insert(self, val):
        if(self.root is None):
            self.set_root(val)
        else:
            self.insert_node(self.root, val)

    def insert_node(self, currentNode, val):
        current = currentNode
        
        while(True):
            if(val < current.val):
                if(current.leftChild != None):
                    current = current.leftChild
                else:
                    current.leftChild = ReviewNode(val, currentNode)
                    break
            else:
                if(current.rightChild != None):
                    current = current.rightChild
                else:
                    current.rightChild = ReviewNode(val, currentNode)
                    break
                
                
    def __str__(self):
        if self.root is None:
            return "empty"
        
        return self.to_str(self.root)
    
    def to_str(self, node):
        string = ""
        if node.leftChild is not None:
            string = string + self.to_str(node.leftChild)
        
        string = string + node.__str__() + '\n'
        
        if node.rightChild is not None:
            string = string + self.to_str(node.rightcompanyChild)
        
        return string



class ReviewDB(object):
    def __init__(self):
        self.review_dict = {}
        self.review_tree = ReviewTree()
        self.id_dict = {}
    
    def add_review_list(self, bookingReview_list):
        for review in bookingReview_list:
            if not (review.review_id in self.review_dict):
                self.review_dict[review.review_id] = review
                self.review_tree.insert(review)
                
                #if not (review.id in self.id_dict):
                #    self.id_dict[review.id] = len(self.id_dict)
                
    def add_review(self, bookingReview):
        if not (bookingReview.review_id in self.review_dict):
            self.review_dict[bookingReview.review_id] = bookingReview
            self.review_tree.insert(bookingReview)      
        
    def get_review(self, review_id):
        if (review_id in self.review_dict):
            return self.review_dict[review_id]
        
    def get_id_spamRecord (self, id):
        if (id in self.id_dict):
            return (self.id_dict[id][0]/self.id_dict[id][1])
        else:
            return 0.0
    
    def add_spam_result (self, id, result):
        accumulate = 0
        if result:
            accumulate = 1
            
        if (id in self.id_dict):
            self.id_dict[id][0] += accumulate
            self.id_dict[id][1] += 1
        else:
            self.id_dict[id] = (accumulate, 1)
                
    def size(self):
        return len(self.review_dict)
    
    def __str__(self):
        return self.review_tree.__str__()


class FormattedReview(object):
    reviewDB = None
    attribute_num = 7
    def __init__(self, preprocessReview):
        
        self.bookingReview = preprocessReview.db_node.val
        
        self.context = preprocessReview.context_word2vec
        self.context_bayes = preprocessReview.context_postag
        self.calc_comp_similarity(preprocessReview)
        self.rate = preprocessReview.rate / 10
        self.reiteration_context = self.calc_reiteration_context(self.bookingReview)
        self.reiteration_repeat = self.calc_reiteration_repeat(self.bookingReview)
        self.post_time = preprocessReview.post_time % 1
        self.post_vip = (int(preprocessReview.post_time) % 7) / 7
        
        self.id = self.reviewDB.get_id_spamRecord(preprocessReview.id)
        
        self.label = preprocessReview.label
        self.review_id = preprocessReview.review_id
    
    def calc_comp_similarity(self, preprocessReview):
        max_sim = -1;
        for company_vec in preprocessReview.company_word2vec:
            for context_vec in preprocessReview.context_word2vec:
                cos_sim = 1 - spatial.distance.cosine(company_vec, context_vec)
                max_sim = max_sim if (max_sim > cos_sim) else cos_sim
        
        self.comp_similarity = max_sim
    
    def calc_reiteration_context(self, bookingReview, num = 1):
        if num > 10:         # reiteration_context 최대 수치는 1
            return 0
        
        prev_review = bookingReview.db_node.previous_node(num)
        if prev_review is None:
            return 0

        prev_review = prev_review.val
        
        if(prev_review.id == bookingReview.id     #리뷰어 동일
               and prev_review.context == bookingReview.context   #텍스트 내용 동일
               and bookingReview.post_time - prev_review.post_time < 30):   #한달 이내 작성
            return 0.1 + self.calc_reiteration_context(bookingReview, num+1)
        
        else:
            return 0
        
        
    def calc_reiteration_repeat(self, bookingReview, num = 1):     
        prev_review = bookingReview.db_node.previous_node(num)
        if prev_review is None:
            return 0

        prev_review = prev_review.val
        
        if(prev_review.company == bookingReview.company      #업체명 동일
               and prev_review.id == bookingReview.id):     #리뷰어 동일
            
            time_diff = bookingReview.post_time - prev_review.post_time
            
            if(time_diff < 1): #하루 이내 작성
                val = 0.1+self.calc_reiteration_repeat(bookingReview, num+1)
            
            elif(time_diff < 365): #1년 이내 작성
                val = 0.1*time_diff/365+self.calc_reiteration_repeat(bookingReview, num+1)
            
            else:
                val = 0.1
                
            return val if val<1.0 else 1.0
        
        else:
            return 0

    def get_attribute(self):
        return np.array([self.comp_similarity, self.rate, self.reiteration_context,
                         self.reiteration_repeat, self.post_time, self.post_vip, self.id])
        
    @classmethod    
    def setDB(self, reviewDB):
        self.reviewDB = reviewDB
    
    def __str__(self):
        return ("FormattedReview object {0}:\n"
                "  context = \n{1}\n"
                "  context_bayes = \n{9}\n"
                "  comp_similarity = {2}\n"
                "  rate = {3}\n"
                "  reiteration_context = {4}\n"
                "  reiteration_repeat = {5}\n"
                "  post_time = {6}\n"
                "  post_vip = {7}\n"
                "  label = {8}\n"
                .format(self.review_id, self.context, self.comp_similarity,
                        self.rate, self.reiteration_context, self.reiteration_repeat,
                        self.post_time, self.post_vip, self.label, self.context_bayes))

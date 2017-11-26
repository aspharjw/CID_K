import numpy as np
from scipy import spatial
import bookingreview
import preprocessreview
from pydblite import Base


class ReviewNode:
    def __init__(self, data, prev=None, next=None):
        self.data = data
        self.prev = prev
        self.next = next
        
    def get(self):
        return self.data
    
    def set(self, val):
        self.data = val
  
    def previous_node(self, num = 1):
        if(num == 0):
            return self
        
        return self.prev.previous_node(num-1)

    def __str__(self):     
        return "ReviewNode - "+self.data.__str__() + '\n'
    
        
class LinkedList:
    def __init__(self):
        self.root = None
        self.size = 0

    def set_root(self, val):
        self.root = ReviewNode(val)
        self.root.prev = self.root
        self.root.next = self.root
        self.size = 1

    def insert_after(self, val, prev):
        if (prev is None) or self.size == 0:
            self.set_root(val)
           
        else:
            inserted = ReviewNode(val, prev, prev.next)
            prev.next.prev = inserted
            prev.next = inserted
            self.size += 1
            
    def insert_tail(self, val):
        if self.size == 0:
            self.set_root(val)
        else:
            self.insert_after(val, self.root.prev)

    def to_list(self, get_node = False):
        if self.root is None:
            return []
        
        if get_node: ret = [self.root]
        else: ret = [self.root.data]
            
        temp = self.root.next
        for i in range(1, self.size):
            if get_node: ret.append(temp)
            else: ret.append(temp.data)
            temp = temp.next
        
        return ret

    @classmethod
    def list_to_linked(cls, arg):
        ret = LinkedList()
        for val in arg:
            ret.insert_tail(val)
            
        return ret
                
                
    def __str__(self):
        if self.root is None:
            return "empty"
        
        string = self.root.__str__()+ '\n'
        temp = self.root.next
        while (temp is not self.root):
            string = string + temp.__str__() + '\n'
            temp = temp.next
        
        return string


class ReviewDB(object):
    def __init__(self, name):
        self.review_db = Base(name + "_DB.pdl")
        self.review_dict = {}
        
        if self.review_db.exists():
            self.review_db.open()
            self.set_review_dict(self.review_db[0]['review_list'])
            
        else:
            self.review_db.create('review_list', 'id_dict')
            self.review_db.insert([], {})
            self.save_db()
    
    def set_review_dict(self, review_list):
        self.review_dict = {}
        linked_list = LinkedList.list_to_linked(review_list)
        for review_node in linked_list.to_list(get_node = True):
            self.review_dict[review_node.data.review_id] = review_node
    
    def add_review_list(self, bookingReview_list):
        temp_list = list()
        for review in bookingReview_list:
            if not (review.review_id in self.review_dict):
                temp_list.append(review)
             
        temp_list.sort()
        prev_list = self.review_db[0]['review_list']
            
        from heapq import merge
        merged_list = list(merge(temp_list, prev_list))
        self.review_db[0]['review_list'] = merged_list
                    
        self.set_review_dict(merged_list)
       
            
        self.review_db.commit()
        
    '''
    def add_review(self, bookingReview):
        if not (bookingReview.review_id in self.review_db[0]['review_dict']):
            self.review_db[0]['review_dict'][bookingReview.review_id] = bookingReview
            self.review_db[0]['tree'].insert(bookingReview)      
            
        self.save_db()
    '''
        
    def get_review_node(self, review_id):
        if (review_id in self.review_dict):
            return self.review_dict[review_id]
        
    def get_id_spamRecord (self, id):
        id_dict = self.review_db[0]['id_dict']
        if (id in id_dict):
            return (id_dict[id][0]/id_dict[id][1])
        else:
            return 0.0
    
    def add_spam_result (self, id, result):
        id_dict = self.review_db[0]['id_dict']
        accumulate = 0
        if result:
            accumulate = 1
            
        if (id in id_dict):
            id_dict[id][0] += accumulate
            id_dict[id][1] += 1
        else:
            id_dict[id] = [accumulate, 1]
            
    def save_db(self):
        self.review_db.commit()
                
    def size(self):
        return len(self.review_db[0]['review_list'])
    
    def __str__(self):
        return "reviewDB __str__ : unimplemented"


class FormattedReview(object):
    reviewDB = None
    attribute_num = 7
    def __init__(self, preprocessReview):
        
        self.label = preprocessReview.label
        self.review_id = preprocessReview.review_id
        
        self.bookingReview = FormattedReview.reviewDB.get_review_node(self.review_id).data
        
        self.context = preprocessReview.context_word2vec
        self.context_bayes = preprocessReview.context_postag
        self.calc_comp_similarity(preprocessReview)
        self.rate = preprocessReview.rate / 10
        self.reiteration_context = self.calc_reiteration_context()
        self.reiteration_repeat = self.calc_reiteration_repeat()
        self.post_time = preprocessReview.post_time % 1
        self.post_vip = (int(preprocessReview.post_time) % 7) / 7
        
        self.id = self.reviewDB.get_id_spamRecord(preprocessReview.id)
    
    def calc_comp_similarity(self, preprocessReview):
        max_sim = -1;
        for company_vec in preprocessReview.company_word2vec:
            for context_vec in preprocessReview.context_word2vec:
                cos_sim = 1 - spatial.distance.cosine(company_vec, context_vec)
                max_sim = max_sim if (max_sim > cos_sim) else cos_sim
        
        self.comp_similarity = max_sim
    
    def calc_reiteration_context(self, num = 1):
        if num > 10:         # reiteration_context 최대 수치는 1
            return 0
        
        prev_review_node = FormattedReview.reviewDB.get_review_node(self.review_id).previous_node(num)
        if prev_review_node is None:
            return 0

        prev_review = prev_review_node.data
        
        if(prev_review.id == self.bookingReview.id     #리뷰어 동일
               and prev_review.context == self.bookingReview.context   #텍스트 내용 동일
               and self.bookingReview.post_time - prev_review.post_time < 30):   #한달 이내 작성
            return 0.1 + self.calc_reiteration_context(num+1)
        
        else:
            return 0
        
        
    def calc_reiteration_repeat(self, num = 1):     
        prev_review_node = FormattedReview.reviewDB.get_review_node(self.review_id).previous_node(num)
        if prev_review_node is None:
            return 0

        prev_review = prev_review_node.data
        
        if(prev_review.company == self.bookingReview.company      #업체명 동일
               and prev_review.id == self.bookingReview.id):     #리뷰어 동일
            
            time_diff = self.bookingReview.post_time - prev_review.post_time
            
            if(time_diff < 1): #하루 이내 작성
                val = 0.1+self.calc_reiteration_repeat(num+1)
            
            elif(time_diff < 365): #1년 이내 작성
                val = 0.1*time_diff/365+self.calc_reiteration_repeat(num+1)
            
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

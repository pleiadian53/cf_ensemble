import collections
import re
import sys
import time
# import utils_sys as utils
import heapq
from operator import itemgetter


class Graph(object):
    def __init__(adjlist=None, vertices=None, V=None): 
        self.adj = {}
        self.V = V
        self.path = {}  # keep track of all reachable vertics starting from a given vertex v

        if adjlist is not None: 
            self.V = len(adjlist)
            for h, vx in adjlist.items(): 
                self.adj[h] = vx

        elif vertices is not None: 
            assert hasattr(vertices, '__iter__')
            self.V = len(vertices)
            for v in vertices: 
                self.adj[v] = []
        else: 
            assert isinstance(V, int)
            self.V = V
            for i in range(V): 
                self.adj[i] = []
    def DFS(x): 
        pass 
    def DFStrace(): 
        pass

def least_common(array, to_find=None):
    # import heapq 
    # from operator import itemgetter
    counter = collections.Counter(array)
    if to_find is None:
        return sorted(counter.items(), key=itemgetter(1), reverse=False)
    return heapq.nsmallest(to_find, counter.items(), key=itemgetter(1))

def tokenize(string):
    """Convert string to lowercase and split into words (ignoring
    punctuation), returning list of words.
    """
    # '\w+' does not work well for codes with special chars such as '.' as part of the 'word'
    return re.findall(r'([-0-9a-zA-Z_:.]+)', string.lower())  


def find_ngrams(input_list, n=3):
    """
    Example
    -------
    input_list = ['all', 'this', 'happened', 'more', 'or', 'less']

    """
    return zip(*[input_list[i:] for i in range(n)])

def count_given_ngrams(seqx, ngrams, partial_order=True):
    """
    Count numbers of occurrences of ngrams in input sequence (seqx, a list of a list of ngrams)

    Related
    -------
    count_given_ngrams2()

    Output
    ------
    A dictionary: n-gram -> count 
    """    

    # usu. the input ngrams have the same length 
    ngram_tb = {1: [], }
    for ngram in ngrams: # ngram is in tuple form 
        if isinstance(ngram, tuple): 
            length = len(ngram)
            if not ngram_tb.has_key(length): ngram_tb[length] = []
            ngram_tb[length].append(ngram)
        else: # assume to be unigrams 
            ngram_tb[1].append(ngram)
            
    ng_min, ng_max = min(ngram_tb.keys()), max(ngram_tb.keys())
    if partial_order:

        # evaluate all possible n-grams 
        counts = count_ngrams2(seqx, min_length=ng_min, max_length=ng_max, partial_order=True)

        counts_prime = {}
        for n, ngx in ngram_tb.items(): # n in n-gram
            if counts.has_key(n): 
                for ngram in ngx: # query each desired ngram 
                    # if n == 1: print '> unigram: %s' % ngram
                    # sorted('x') == sorted(('x', )) == ['x'] =>  f ngram is a unigram, can do ('e', ) or 'e'
                    counts_prime[ngram] = counts[n][tuple(sorted(ngram))] 
            else: 
                for ngram in ngx: 
                    counts_prime[ngram] = 0 
    else: 
        counts = count_ngrams2(seqx, min_length=ng_min, max_length=ng_max, partial_order=False)
        
        counts_prime = {}
        for n, ngx in ngram_tb.items(): # n in n-gram
            if counts.has_key(n): 
                for ngram in ngx: # query each desired ngram 
                    counts_prime[ngram] = counts[n][tuple(ngram)]
            else: 
                for ngram in ngx: 
                    counts_prime[ngram] = 0 

    return counts_prime  # n-gram -> count

def count_given_ngrams2(seqx, ngrams, partial_order=True):
    """
    Count numbers of occurrences of ngrams in input sequence (seqx, a list of a list of ngrams)

    Output
    ------
    A dictionary: n (as in ngram) -> ngram -> count 
    """
    # the input ngrams may or may not have the same length 
    ngram_tb = {1: [], }
    for ngram in ngrams: # ngram is in tuple form 
        if isinstance(ngram, tuple): 
            length = len(ngram)
            if not ngram_tb.has_key(length): ngram_tb[length] = []
            ngram_tb[length].append(ngram)
        else: # assume to be unigrams 
            assert isinstance(ngram, str)
            ngram_tb[1].append(ngram)
            
    # print('verify> ngram_tb:\n%s\n' % ngram_tb) # utils.sample_hashtable(ngram_tb, n_sample=10))

    ng_min, ng_max = min(ngram_tb.keys()), max(ngram_tb.keys())
    if partial_order:

        # evaluate all possible n-grams 
        counts = count_ngrams2(seqx, min_length=ng_min, max_length=ng_max, partial_order=True)

        counts_prime = {}
        for n, ngx in ngram_tb.items(): # n in n-gram
            if not counts_prime.has_key(n): counts_prime[n] = {} 
            if counts.has_key(n):
                for ngram in ngx: # query each desired ngram 
                    # if n == 1: print '> unigram: %s' % ngram
                    # sorted('x') == sorted(('x', )) == ['x'] =>  f ngram is a unigram, can do ('e', ) or 'e'
                    counts_prime[n][ngram] = counts[n][tuple(sorted(ngram))] 
            else: 
                for ngram in ngx: 
                    counts_prime[n][ngram] = 0 
    else: 
        counts = count_ngrams2(seqx, min_length=ng_min, max_length=ng_max, partial_order=False)
        
        counts_prime = {}
        for n, ngx in ngram_tb.items(): # n in n-gram
            if not counts_prime.has_key(n): counts_prime[n] = {} 
            if counts.has_key(n): 
                for ngram in ngx: # query each desired ngram 
                    # assert isinstance(ngram, tuple), "Ngram is not a tuple: %s" % str(ngram)
                    counts_prime[n][ngram] = counts[n][tuple(ngram)]
            else: 
                for ngram in ngx: 
                    counts_prime[n][ngram] = 0 

    return counts_prime  # n (as n-gram) -> counts (ngram -> count)

def count_ngrams2(lines, min_length=2, max_length=4, **kargs): 
    def eval_sequence_dtype(): 
        if not lines: 
            return False # no-op
        if isinstance(lines[0], str): # ['a b c d', 'e f', ]
            return False
        elif hasattr(lines[0], '__iter__'): # [['a', 'b'], ['c', 'd', 'e'], ]
            return True
        return False

    is_partial_order = kargs.get('partial_order', True)
    lengths = range(min_length, max_length + 1)    
    
    # is_tokenized = eval_sequence_dtype()
    seqx = []
    for line in lines: 
        if isinstance(line, str): # not tokenized  
            seqx.append([word for word in tokenize(line)])
        else: 
            seqx.append(line)
    
    # print('count_ngrams2> debug | seqx: %s' % seqx[:5]) # list of (list of codes)
    if not is_partial_order:  # i.e. total order 
        # ordering is important

        # this includes ngrams that CROSS line boundaries 
        # return count_ngrams(seqx, min_length=min_length, max_length=max_length) # n -> counter (of n-grams)

        # this counts ngrams in each line independently 
        counts = count_ngrams_per_seq(seqx, min_length=min_length, max_length=max_length) # n -> counter (of n-grams)
        return {length: counts[length] for length in lengths}

    # print('> seqx:\n%s\n' % seqx)
    # print('status> ordering NOT important ...')
    
    counts = {}
    for length in lengths: 
        counts[length] = collections.Counter()
        # ngrams = find_ngrams(seqx, n=length)  # list of n-grams in tuples
        if length == 1: 
            for seq in seqx: 
                counts[length].update([(ugram, ) for ugram in seq])
        else: 
            for seq in seqx:  # use sorted n-gram to standardize its entry since ordering is not important here
                counts[length].update( tuple(sorted(ngram)) for ngram in find_ngrams(seq, n=length) ) 

    return counts

def count_ngrams_per_line(**kargs):
    return count_ngrams_per_seq(**kargs)
def count_ngrams_per_seq(lines, min_length=1, max_length=4): # non boundary crossing  
    def update(ngrams):
        # print('> line = %s' % single_doc)
        for n, counts in ngrams.items(): 
            # print('  ++ ngrams_total: %s' % ngrams_total)
            # print('      +++ ngrams new: %s' % counts)
            ngrams_total[n].update(counts)
            # print('      +++ ngrams_total new: %s' % ngrams_total)

    lengths = range(min_length, max_length + 1)
    ngrams_total = {length: collections.Counter() for length in lengths}

    doc_boundary_crossing = False
    if not doc_boundary_crossing: # don't count n-grams that straddles two documents
        for line in lines: 
            nT = len(line)
            # print(' + line=%s, nT=%d' % (line, nT))
            single_doc = [line]

            # if the line length, nT, is smaller than max_length, will miscount
            ngrams = count_ngrams(single_doc, min_length=1, max_length=min(max_length, nT))
            update(ngrams) # update total counts
    else: 
        raise NotImplementedError

    return ngrams_total

def count_ngrams(lines, min_length=1, max_length=4): 
    """
    Iterate through given lines iterator (file object or list of
    lines) and return n-gram frequencies. The return value is a dict
    mapping the length of the n-gram to a collections.Counter
    object of n-gram tuple and number of times that n-gram occurred.
    Returned dict includes n-grams of length min_length to max_length.

    Use this only when (strict) ordering is important; otherwise, use count_ngrams2()

    Input
    -----
    lines: [['x', 'y', 'z'], ['y', 'x', 'z', 'u'], ... ]
    """
    def add_queue():
        # Helper function to add n-grams at start of current queue to dict
        current = tuple(queue)
        for length in lengths:
            if len(current) >= length:  # count n-grams up to length those in queue
                ngrams[length][current[:length]] += 1  # ngrams[length] => counter
    def eval_sequence_dtype(): 
        if not lines: 
            return False # no-op
        if isinstance(lines[0], str): 
            return False
        elif hasattr(lines[0], '__iter__'): 
            return True
        return False

    lengths = range(min_length, max_length + 1)
    ngrams = {length: collections.Counter() for length in lengths}
    queue = collections.deque(maxlen=max_length)

    # tokenized or not? 
    is_tokenized = eval_sequence_dtype()
    # print('> tokenized? %s' % is_tokenized)

    # Loop through all lines and words and add n-grams to dict
    if is_tokenized: 
        # print('input> lines: %s' % lines)
        for line in lines:
            for word in line:
                queue.append(word)
                if len(queue) >= max_length:
                    add_queue()  # this does the counting
            # print('+ line: %s\n+ngrams: %s' % (line, ngrams))
    else: 
        for line in lines:
            for word in tokenize(line):
                queue.append(word)
                if len(queue) >= max_length:
                    add_queue()

    # Make sure we get the n-grams at the tail end of the queue
    while len(queue) > min_length:
        queue.popleft()
        add_queue()
        # print('+ line: %s\n+ngrams: %s' % (line, ngrams))

    return ngrams

def check_boundary(lines, ngram_counts):
    # def isInDoc(ngstr): 
    #     for line in lines: 
    #         linestr = sep.join(str(e) for e in line)
    #         if linestr.find(ngstr) >= 0: 
    #             return True 
    #     return False

    # sep = ' ' 
    # for n, counts in ngram_counts: 
    #     counts_prime = []  # only keep those that do not cross line boundaries
    #     crossed = set()
    #     for ngr, cnt in counts: 

    #         # convert to string 
    #         ngstr = sep.join([str(e) for e in ngr])
    #         if isInDoc(ngstr): 
    #             counts_prime[]
    raise NotImplementedError
    # return ngram_counts  # new ngram counts


def print_most_frequent(ngrams, num=10):
    """Print num most common n-grams of each length in n-grams dict."""
    for n in sorted(ngrams):
        print('----- {} most common {}-grams -----'.format(num, n))
        for gram, count in ngrams[n].most_common(num):
            print('{0}: {1}'.format(' '.join(gram), count))
        print('')

def t_priority_queue(): 
    import platform
    import heapq

    try:
        import Queue as Q  # ver. < 3.0
    except ImportError:
        print("> import queue | python version %d" % platform.python_version())
        import queue as Q

    q = Q.PriorityQueue()
    q.put((10,'ten'))
    q.put((1,'one'))
    q.put((5,'five'))
    while not q.empty():
        print ('%s, ' % q.get())

    print('info> try heapq module ...')
    
    heap = []
    heapq.heappush(heap, (-1.5, 'negative one'))
    heapq.heappush(heap, (1, 'one'))
    heapq.heappush(heap, (10, 'ten'))
    heapq.heappush(heap, (5.7,'five'))
    heapq.heappush(heap, (100.6, 'hundred'))

    for x in heap:
        print ('%s, ' % x)
    print()

    heapq.heappop(heap)

    for x in heap:
        print('{0},'.format(x)) # print x,   
    print()

    # the smallest
    print('info> smallest: %s' % str(heap[0]))

    smallestx = heapq.nsmallest(2, heap)  # a list
    print('info> n smallest: %s, type: %s' % (str(smallestx), type(smallestx)))

    return

def split(alist, n):
    n = min(n, len(alist)) # don't create empty buckets in scenarios like list(split(range(X, Y))) where X < Y
    k, m = divmod(len(alist), n)
    return (alist[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def test(): 
    
    # split a list of elements into N (approx) equal parts 
    parts = list(split(range(11), 3))
    print(parts)

    return

if __name__ == "__main__": 
    test()
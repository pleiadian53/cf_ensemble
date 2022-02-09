import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import time

"""

    Reference 
    ---------
    
    1. Chris Johnson
       https://raw.githubusercontent.com/MrChrisJohnson/implicit-mf/master/mf.py

        + different versions 
            https://www.ethanrosenthal.com/2016/10/19/implicit-mf-part-1/
        
        Benfred, in Cython 
            https://github.com/benfred/implicit

"""

def load_matrix(filename, num_users, num_items):
    t0 = time.time()
    counts = sparse.dok_matrix((num_users, num_items), dtype=float)
    total = 0.0
    num_zeros = num_users * num_items
    for i, line in enumerate(open(filename, 'r')):
        user, item, count = line.strip().split('\t')
        user = int(user)
        item = int(item)
        count = float(count)
        if user >= num_users:
            continue
        if item >= num_items:
            continue
        if count != 0:
            counts[user, item] = count
            total += count
            num_zeros -= 1
        if i % 100000 == 0:
            print 'loaded %i counts...' % i
    alpha = num_zeros / total
    print 'alpha %.2f' % alpha
    counts *= alpha
    counts = counts.tocsr()
    t1 = time.time()
    print 'Finished loading matrix in %f seconds' % (t1 - t0)
    return counts


class ImplicitMF(object):

    def __init__(self, counts, num_factors=40, num_iterations=30,
                 reg_param=0.8):
        self.counts = counts
        self.num_users = counts.shape[0]
        self.num_items = counts.shape[1]
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.reg_param = reg_param

    def train_model(self):
        self.user_vectors = np.random.normal(size=(self.num_users,
                                                   self.num_factors))
        self.item_vectors = np.random.normal(size=(self.num_items,
                                                   self.num_factors))

        for i in range(self.num_iterations):
            t0 = time.time()
            print 'Solving for user vectors...'
            self.user_vectors = self.iteration(True, sparse.csr_matrix(self.item_vectors))
            print 'Solving for item vectors...'
            self.item_vectors = self.iteration(False, sparse.csr_matrix(self.user_vectors))
            t1 = time.time()
            print 'iteration %i finished in %f seconds' % (i + 1, t1 - t0)

    def iteration(self, user, fixed_vecs):
        num_solve = self.num_users if user else self.num_items
        num_fixed = fixed_vecs.shape[0]
        YTY = fixed_vecs.T.dot(fixed_vecs)
        eye = sparse.eye(num_fixed)
        lambda_eye = self.reg_param * sparse.eye(self.num_factors)
        solve_vecs = np.zeros((num_solve, self.num_factors))

        t = time.time()
        for i in range(num_solve):
            if user:
                counts_i = self.counts[i].toarray()
            else:
                counts_i = self.counts[:, i].T.toarray()
            CuI = sparse.diags(counts_i, [0])
            
            pu = counts_i.copy()
            
            pu[np.where(pu != 0)] = 1.0
            
            YTCuIY = fixed_vecs.T.dot(CuI).dot(fixed_vecs)
            YTCupu = fixed_vecs.T.dot(CuI + eye).dot(sparse.csr_matrix(pu).T)
            xu = spsolve(YTY + YTCuIY + lambda_eye, YTCupu)
            solve_vecs[i] = xu
            if i % 1000 == 0:
                print 'Solved %i vecs in %d seconds' % (i, time.time() - t)
                t = time.time()

        return solve_vecs

### ImplicitMF

# multithreaded version
class ImplicitMF2():
    """
    Original code from Chris Johnson:
    https://github.com/MrChrisJohnson/implicit-mf

    Multithreading added by Thierry Bertin-Mahieux (2014)
    """

    def __init__(self, counts, num_factors=40, num_iterations=30,
                 reg_param=0.8, num_threads=1):
        self.counts = counts
        self.num_users = counts.shape[0]
        self.num_items = counts.shape[1]
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.reg_param = reg_param
        self.num_threads = num_threads

    def train_model(self):
        self.user_vectors = np.random.normal(size=(self.num_users,
                                                   self.num_factors))
        self.item_vectors = np.random.normal(size=(self.num_items,
                                                   self.num_factors))

        for i in xrange(self.num_iterations):
            t0 = time.time()

            user_vectors_old = copy.deepcopy(self.user_vectors)
            item_vectors_old = copy.deepcopy(self.item_vectors)

            print 'Solving for user vectors...'
            self.user_vectors = self.iteration(True, sparse.csr_matrix(self.item_vectors))
            print 'Solving for item vectors...'
            self.item_vectors = self.iteration(False, sparse.csr_matrix(self.user_vectors))
            t1 = time.time()
            print 'iteration %i finished in %f seconds' % (i + 1, t1 - t0)
            norm_diff = scipy.linalg.norm(user_vectors_old - self.user_vectors) + scipy.linalg.norm(item_vectors_old - self.item_vectors)
            print 'norm difference:', norm_diff

    def iteration(self, user, fixed_vecs):
        num_solve = self.num_users if user else self.num_items
        num_fixed = fixed_vecs.shape[0]
        YTY = fixed_vecs.T.dot(fixed_vecs)
        eye = sparse.eye(num_fixed)
        lambda_eye = self.reg_param * sparse.eye(self.num_factors)
        solve_vecs = np.zeros((num_solve, self.num_factors))

        batch_size = int(np.ceil(num_solve * 1. / self.num_threads))
        print 'batch_size per thread is: %d' % batch_size
        idx = 0
        processes = []
        done_queue = Queue()
        while idx < num_solve:
            min_i = idx
            max_i = min(idx + batch_size, num_solve)
            p = Process(target=self.iteration_one_vec,
                        args=(user, YTY, eye, lambda_eye, fixed_vecs, min_i, max_i, done_queue))
            p.start()
            processes.append(p)
            idx += batch_size

        cnt_vecs = 0
        while True:
            is_alive = False
            for p in processes:
                if p.is_alive():
                    is_alive = True
                    break
            if not is_alive and done_queue.empty():
                break
            time.sleep(.1)
            while not done_queue.empty():
                res = done_queue.get()
                i, xu = res
                solve_vecs[i] = xu
                cnt_vecs += 1
        assert cnt_vecs == len(solve_vecs)

        done_queue.close()
        for p in processes:
            p.join()

        print 'All processes completed.'
        return solve_vecs

    def iteration_one_vec(self, user, YTY, eye, lambda_eye, fixed_vecs, min_i, max_i, output):
        t = time.time()
        cnt = 0
        for i in xrange(min_i, max_i):
            if user:
                counts_i = self.counts[i].toarray()
            else:
                counts_i = self.counts[:, i].T.toarray()
            CuI = sparse.diags(counts_i, [0])
            pu = counts_i.copy()
            pu[np.where(pu != 0)] = 1.0
            YTCuIY = fixed_vecs.T.dot(CuI).dot(fixed_vecs)
            YTCupu = fixed_vecs.T.dot(CuI + eye).dot(sparse.csr_matrix(pu).T)
            xu = spsolve(YTY + YTCuIY + lambda_eye, YTCupu)
            output.put((i, list(xu)))
            cnt += 1
            if cnt % 1000 == 0:
                print 'Solved %d vecs in %d seconds (one thread)' % (cnt, time.time() - t)
        output.close()
        print 'Process done.'



def test(): 

    return

if __name__ == "__main__": 
    test()

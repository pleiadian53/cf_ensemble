import sys, gc
import random, time

import pandas as pd
import numpy as np
import scipy as sp

from evaluate import Metrics, PerformanceMetrics, plot_roc
from sklearn.metrics import brier_score_loss

import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve  # solve
from sklearn.preprocessing import MinMaxScaler

# import scipy.sparse
# import scipy.sparse.linalg

from sklearn.metrics import mean_squared_error

from cf_spec import System
import utils_sys
from utils_sys import div

# MFEnsemble
# codes = {'tp': 2, 'tn': 1, 'fp': -2, 'fn': -1, 
#             'unk': 0, 't': 3, 'f': -3}

class ImplicitMF(object):
    """

    Reference
    ---------
    1. Chris Johnson
       https://raw.githubusercontent.com/MrChrisJohnson/implicit-mf/master/mf.py
       https://github.com/MrChrisJohnson/implicit-mf/blob/master/mf.py

    2. Collaborative Filtering for Implicit Feedback Datasets, Yifan Hu, Chris Volinsky, 2007

    3. Implicit Recommender System: 

       http://activisiongamescience.github.io/2016/01/11/Implicit-Recommender-Systems-Biased-Matrix-Factorization/

    Memo
    ----
    See demo_ALS4.py for a multithreaded version 

    """

    def __init__(self, confidence, num_factors=40, num_iterations=30, reg_param=0.8, **kargs):
        """

        Params
        ------
            policy: 'rating' => dot(xu, yi) attempts to reconstruct rating scores (or probability scores)
                    'preference' 
                    'rating + label'

        """
        self.confidence = confidence  # C - Ones
        self.polarity = kargs.get('polarity', None)

        self.label_confidence = kargs.get('label_confidence', None)  # in label-regularized mode in which dot(xu, yi) attempts to have a trade off between approximating probabilities and true labels 

        self.num_users = confidence.shape[0]
        self.num_items = confidence.shape[1]
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.reg_param = reg_param

        self.ratings = kargs.get('ratings', None)
        self.p_threshold = kargs.get('p_threshold', [])
        self.labels = kargs['labels'] if 'labels' in kargs else []
        self.policy = kargs.get('policy', 'rating') # options: rating, preference, label

        self.missing_value = self.fill = kargs.get('fill', 0)

        # positive preference 
        self.positive_pref = kargs.get('positive_pref', 1.0)
        self.negative_pref = kargs.get('negative_pref', 0.0)  # or -1.0
        self.pos_label = kargs.get('pos_label', 1) 
        self.neg_label = kargs.get('neg_label', 0)

        # test
        self.train_errors = []  
        self.train_errors_weighted = []

        # ported from MFEnsemble and utils_cf
        self.codes = {'tp': 2, 'tn': 1, 'fp': -2, 'fn': -1, 
            'unk': 0, 't': 3, 'f': -3}


    @staticmethod
    def rmse_score2(C, T, P, Q, fill=0): # [todo]
        # C: masked confidence matrix, X: a given (test) rating matrix (T)

        # I = C != fill  # Indicator function which is zero for missing data
        I = (C != fill).todense()  # Indicator function which is zero for missing data
        
        # ME = I * (T - np.dot(P, Q.T))  # Errors between real and predicted ratings
        ME = np.multiply(I, T-np.dot(P, Q.T))
        
        # MSE = ME**2  
        # return np.sqrt(np.sum(MSE)/np.sum(I))  # sum of squared errors
        return np.sqrt(np.mean(np.square(ME)))
    ##################################################
    # -- alias --
    prediction_error = rmse_score2
    ##################################################

    def rmse_score(self): # [todo]
        C = self.confidence # masked confidence matirx
        R = self.ratings
        P, Q = self.user_vectors, self.item_vectors
        # print('(rmse_score) verify | dim(C): {dc}, dim(R): {dr} | dim(P): {dp}, dim(Q): {dq} | n_factors: {nf}'.format(dc=C.shape, 
        #     dr=R.shape, dp=P.shape, dq=Q.shape, nf=self.num_factors))

        # C is a sparse matrix (multiply a sparse and a dense matrix element-wise leads to 'dimension mismatch' error!)
        I = (C != self.fill).todense()  # Indicator function which is zero for missing data
        
        # print('... dim(I): {di}, dim(R - np.dot(P, Q.T)): {d}, dtype: {tc} vs {t}'.format(di=I.shape, d=(R -np.dot(P, Q.T)).shape, tc=type(I), t=type(R-np.dot(P, Q.T))))
        
        # note: problem with element-wise mul with sparse matrix
        # ME = I * (R - np.dot(P, Q.T))  
        ME = np.multiply(I, R-np.dot(P, Q.T)) # Errors between real and predicted ratings where the corresponding confidence matrix is not zero
        # MSE = ME**2  
        # return np.sqrt(np.sum(MSE)/np.sum(I))  # sum of squared errors
        return np.sqrt(np.mean(np.square(ME)))
    def rmse_score_weighted(self):
        C = self.confidence # masked confidence matirx
        R = self.ratings
        P, Q = self.user_vectors, self.item_vectors
        # print('(rmse_score) verify | dim(C): {dc}, dim(R): {dr} | dim(P): {dp}, dim(Q): {dq} | n_factors: {nf}'.format(dc=C.shape, 
        #     dr=R.shape, dp=P.shape, dq=Q.shape, nf=self.num_factors))

        # C is a sparse matrix (multiply a sparse and a dense matrix element-wise leads to 'dimension mismatch' error!)
        # I = (C != self.fill).todense()  # Indicator function which is zero for missing data
        Cd = C.todense()

        # print('... dim(I): {di}, dim(R - np.dot(P, Q.T)): {d}, dtype: {tc} vs {t}'.format(di=I.shape, d=(R -np.dot(P, Q.T)).shape, tc=type(I), t=type(R-np.dot(P, Q.T))))
        
        # note: problem with element-wise mul with sparse matrix
        # ME = I * (R - np.dot(P, Q.T))  
        ME = np.multiply(Cd, R-np.dot(P, Q.T)) # Errors between real and predicted ratings where the corresponding confidence matrix is not zero
        # MSE = ME**2  
        # return np.sqrt(np.sum(MSE)/np.sum(I))  # sum of squared errors
        return np.sqrt(np.mean(np.square(ME))) 
    
    def evaluate(self, index=0, verbose=True):
        if self.policy.startswith('ra'): 
            assert self.ratings is not None and self.confidence is not None, "Need both the confidence matrix (C) and ratings (R) to compute RMSEs"
            s = self.rmse_score()
            sw = self.rmse_score_weighted()
            s_prev = self.train_errors[-1] if len(self.train_errors) > 0 else s
            sw_prev = self.train_errors_weighted[-1] if len(self.train_errors_weighted) > 0 else sw
            ds = s - s_prev  # hopefully negative
            dsw = sw - sw_prev
            self.train_errors.append(s)
            self.train_errors_weighted.append(sw)
            if verbose and index > 0: print('...... iter: %d | RMSE: %f | WRMSE: %f | decreasing (-)? (delta: %f, delta_w: %f)' % (index, s, sw, ds, dsw))
        return

    def determine_iter_routine(self): 
        print('ImplicitMF> Optimization Policy: %s' % self.policy)

        iter_routine = self.iter_preference
        if self.policy.startswith('ra'): # approximate probabilities (ratings)
            assert self.ratings is not None
            # print('(train_model) iter routine: {}'.format('estimate ratings'))

            if self.polarity is None: 
                iter_routine = self.iteration # [options] self.iteration # self.iteration_colored  
            else: 
                iter_routine = self.iteration_colored

        elif self.policy.startswith('l'): # labels given 
            nL = len(self.labels)
            assert nL > 0
            assert nL == self.confidence.shape[1]
            iter_routine = self.iter_label
        elif self.policy.startswith('pr'): # preference
            # latent factors dot(xu, yi) attempts to approximate p (preference); p(u,1) <- 1 if r(u, r) > 0 
            iter_routine = self.iter_preference_polarized    # self.iter_preference
            print('(train_model) iter routine: {}'.format(iter_routine))
        elif self.policy.startswith('t'):  # t: tradeoff between approximating probabilities and approximating true labels
            assert self.ratings is not None 
            nL = len(self.labels)
            assert nL > 0 and nL == self.confidence.shape[1]
            assert isinstance(self.label_confidence, float) or self.label_confidence.shape == self.confidence.shape
            iter_routine = self.iter_label_regularized 
        return iter_routine

    def train_model(self, verbose=True, verify=True, user_vectors=None, item_vectors=None, add_bias=True):

        ### initialization 
        center = (self.positive_pref+self.negative_pref)/2.0
        if user_vectors is None: 
            self.user_vectors = np.random.normal(loc=center, size=(self.num_users, self.num_factors))
        else: 
            assert user_vectors.shape == (self.num_users, self.num_factors), "Dimension mismatch. Exepcting a matrix of dim: {0} by {1}".format(self.num_users, self.num_factors)
            self.user_vectors = user_vectors

        if item_vectors is None: 
            self.item_vectors = np.random.normal(loc=center, size=(self.num_items, self.num_factors))
        else: 
            assert item_vectors.shape == (self.num_items, self.num_factors), "Dimension mismatch. Exepcting a matrix of dim: {0} by {1}".format(self.num_items, self.num_factors)
            self.item_vectors = item_vectors

        ### choose iteration routine
        iter_routine = self.determine_iter_routine()

        if add_bias: 
            print("(train_model) adding biases ...")
            ### introduce bias parameters 
            nu, ni = self.user_vectors.shape[0], self.item_vectors.shape[0]
            beta = np.zeros( (1, nu) )  # dim needs to be consistent with p
            gamma = np.zeros( (1, ni) )

            # self.user_vectors = np.hstack([ beta.reshape(nu, 1) , user_vectors])
            # self.item_vectors = np.hstack([ gamma.reshape(ni, 1) , item_vectors])

            for i in range(self.num_iterations):
                t0 = time.time()

                # if verbose and i % 10 == 0: print('(ALS) Solving for user vectors | iteration %d of %d ... ' % (i+1, self.num_iterations))
                # augment the vector with bias components 
                item_vectors = sparse.hstack( [np.ones( (self.num_items, 1) ), sparse.csr_matrix(self.item_vectors)] )
                user_vectors = iter_routine(True, item_vectors, beta=beta, gamma=gamma, add_bias=True, it=i)  # it: iteration number, for debugging only
                if sparse.issparse(user_vectors): user_vectors = user_vectors.toarray()
                beta, self.user_vectors = user_vectors[:,0].reshape(1, nu), user_vectors[:,1:]
                # ... new (beta, self.user_vectors)

                # if verbose and i % 10 == 0: print('(ALS) Solving for item vectors | iteration %d of %d ... ' % (i+1, self.num_iterations))
                # augment item_vectors and user_vectors
                user_vectors = sparse.hstack( [np.ones( (self.num_users, 1) ), sparse.csr_matrix(self.user_vectors)] )
                item_vectors = iter_routine(False, user_vectors, beta=beta, gamma=gamma, add_bias=True, it=i)
                if sparse.issparse(item_vectors): item_vectors = item_vectors.toarray()
                gamma, self.item_vectors = item_vectors[:, 0].reshape(1, ni), item_vectors[:,1:]
                # ... new (gamma, self.item_vectors)

                t1 = time.time()

                # test
                if verbose and ((i > 0 and i % 10 == 0) or (i == self.num_iterations-1)): 
                    print('... iteration %i finished in %f seconds ...' % (i + 1, t1 - t0))
                if verify: self.evaluate(index=i, verbose=verbose and i % 10 == 0)  # compute RMSE if ratings (R) was given and policy is 'ratings'
                gc.collect()
        else: 
            for i in range(self.num_iterations):
                t0 = time.time()
                if verbose and i % 10 == 0: 
                    print(f"(ALS) Solving for USER vectors via {iter_routine.__name__} | iteration {i+1} of {self.num_iterations} ...")
                self.user_vectors = iter_routine(True, sparse.csr_matrix(self.item_vectors), it=i)  # it: iteration number, for debugging only

                if verbose and i % 10 == 0: 
                    print(f"(ALS) Solving for ITEM vectors via {iter_routine.__name__} | iteration {i+1} of {self.num_iterations} ...")
                self.item_vectors = iter_routine(False, sparse.csr_matrix(self.user_vectors), it=i)

                t1 = time.time()

                # test
                if verbose and ((i > 0 and i % 10 == 0) or (i == self.num_iterations-1)): 
                    print('... iteration %i finished in %f seconds ...' % (i + 1, t1 - t0))
                if verify: self.evaluate(index=i, verbose=verbose and i % 10 == 0)  # compute RMSE if ratings (R) was given and policy is 'ratings'
                gc.collect()

    def train_model_foldin(self, user_vectors=None, item_vectors=None, verify=True, verbose=True, resume_als=False):
        """

        Params
        ------
        als_mode: if True, then the given user_vectors or item_vectors are used seedings, from which ALS is resumed. 

        """
        if user_vectors is None and item_vectors is None: 
            return self.train_model()
        # if resume_als: 
        #     # at least one of them is given
        #     return self.train_model(user_vectors=user_vectors, item_vectors=item_vectors)
        
        # assuming that either user_vectors or item_vectors are known, determine the other (ALS -> LS)
        fixed_u = fixed_i = False
        if user_vectors is not None: 
            fixed_u = True
            self.user_vectors = user_vectors 
        else: 
            self.user_vectors = np.random.normal(size=(self.num_users, self.num_factors))

        if item_vectors is not None: 
            fixed_i = True      
            self.item_vectors = item_vectors
        else:     
            self.item_vectors = np.random.normal(size=(self.num_items, self.num_factors))

        if fixed_u and fixed_i: 
            print("Warning: Both user and item vector are fixed! No-op.")
            return
        elif (fixed_u and not fixed_i) or (not fixed_u and fixed_i): # XOR 
            print("(train_model) fold-in mode.")

        ### choose iteration routine given policy
        iter_routine = self.determine_iter_routine()
 
        n_iter_plugin = 2
        for i in range(n_iter_plugin): # self.num_iterations
            t0 = time.time()

            if not fixed_u: # user vectors are to be updated while item_vectors are held fixed
                if verbose and i % 10 == 0: 
                    print(f"(LS) Solving for USER vectors via {iter_routine.__name__} | iteration {i+1} of {n_iter_plugin} ...")
                self.user_vectors = iter_routine(True, sparse.csr_matrix(self.item_vectors))   # fix item_vectors and solve for user_vectors

            if not fixed_i: # item vectors are to be updated while user_vectors are held fixed
                if verbose and i % 10 == 0: 
                    print(f"(LS) Solving for ITEM vectors via {iter_routine.__name__} | iteration {i+1} of {n_iter_plugin} ...")
                    # print('     ... item_vector[0][:10]: {iv}'.format(iv=self.item_vectors[0][:10]))
                self.item_vectors = iter_routine(False, sparse.csr_matrix(self.user_vectors))  # fix user_vectors and solve for item_vectors (useful for predicting new items in T)

            t1 = time.time()

            if verbose and ((i > 0 and i % 10 == 0) or (i == n_iter_plugin-1)): 
                print('... iteration %i finished in %f seconds' % (i + 1, t1 - t0))
            if verify: self.evaluate(index=i, verbose=verbose and i % 10 == 0)  # compute RMSE if ratings (R) was given and policy is 'ratings'
            gc.collect()

        if resume_als: 
            # at least one of them is given
            print('... (LS) complete => resume regular ALS using pretrained vectors (n_iter={n}) (verify) #'.format(n=self.num_iterations))
            return self.train_model(user_vectors=self.user_vectors, item_vectors=self.item_vectors)

        return

    def iteration(self, is_user, fixed_vecs, mask_=False, it=0, add_bias=False, beta=None, gamma=None):
        """
        Approximate ratings

        Memo
        ----
        1. Y'(Cu-I)Y is easier to compute when Cu-I has nu non-zeros, where nu << n 

           c_ui <- 1+ alpha * r_ui 
        """
        num_solve = self.num_users if is_user else self.num_items
        num_fixed = fixed_vecs.shape[0]
        # ... if add_bias is True, fixed_vecs will have an additional column of 1's padded in front 

        YTY = fixed_vecs.T.dot(fixed_vecs)
        # ... nf * nf
        
        eye = sparse.eye(num_fixed)  

        num_factors = self.num_factors+1 if add_bias else self.num_factors
        lambda_eye = self.reg_param * sparse.eye(num_factors)

        solve_vecs = np.zeros((num_solve, num_factors))

        t = time.time()
        for i in range(num_solve):
            if is_user:
                counts_i = self.confidence[i].toarray()
                ratings_i = self.ratings[i] # .toarray()  # whole row of ratings across items
                # ... dim(ratings_i): n_items
            else: # is_item 
                counts_i = self.confidence[:, i].T.toarray()
                ratings_i = self.ratings[:, i].T # .toarray()  # whole column of ratings for an item
                # ... dim(ratings_i): n_users

            CuI = sparse.diags(counts_i, [0])  # per-user or per-item diagonal matrix
            
            ## this is what xu dot yi tries to approximate
            # pu = counts_i.copy()  # if pu repr. preferences {0, 1}
            # pu[np.where(pu != 0)] = 1.0  # preferred if count > 0

            pu = ratings_i.copy() # treat pu as 'rating' 
            if add_bias: pu = pu - gamma if is_user else pu - beta 
            
            # Y'CuY => Y'Y + Y'(Cu-I)Y
            YTCuIY = fixed_vecs.T.dot(CuI).dot(fixed_vecs)  # Y'(Cu-I)Y i.e. CuI ~ Cu - I in the paper, sparsity property makes it faster
            # ... (nf * ni) (ni * ni) (ni * nf) -> nf * nf

            YTCupu = fixed_vecs.T.dot(CuI + eye).dot(sparse.csr_matrix(pu).T)  # CuI + eye ~ Cu, sparsity property makes it faster
            # ... (nf * ni) (ni * ni) (ni * 1) => nf * 1

            # Y'Y + Y'(Cu-I)Y + lambda * I
            xu = spsolve(YTY + YTCuIY + lambda_eye, YTCupu)  # <<< this is the core
            solve_vecs[i] = xu
            # ... nf * 1
            
            # if i > 0 and i % 1000 == 0:
            #     print('Solved %d vecs in %d seconds' % (i, time.time() - t))
            #     t = time.time()

        # if add_bias:
        #     if is_user: 
        #         solve_vecs = sparse.hstack([ beta.reshape(self.num_users, 1) , solve_vecs])
        #     else: 
        #         solve_vecs = sparse.hstack([ gamma.reshape(self.num_items, 1) , solve_vecs])

        return solve_vecs  # if add_bias, then solve_vecs has an additional dimension beta(u) or gamma(i)

    def iteration_colored(self, is_user, fixed_vecs, mask_=False, it=0, add_bias=False, beta=None, gamma=None):
        """
        Approximate ratings

        Memo
        ----
        1. Y'(Cu-I)Y is easier to compute when Cu-I has nu non-zeros, where nu << n 

           c_ui <- 1+ alpha * r_ui 
        """
        num_solve = self.num_users if is_user else self.num_items
        num_fixed = fixed_vecs.shape[0]
        # ... if add_bias is True, fixed_vecs will have an additional column of 1's padded in front 

        YTY = fixed_vecs.T.dot(fixed_vecs)
        # ... nf * nf
        
        eye = sparse.eye(num_fixed)  

        num_factors = self.num_factors+1 if add_bias else self.num_factors
        lambda_eye = self.reg_param * sparse.eye(num_factors)

        solve_vecs = np.zeros((num_solve, num_factors))

        t = time.time()
        
        ############################################
        ctp, ctn = self.codes['tp'], self.codes['tn']
        cfp, cfn = self.codes['fp'], self.codes['fn']
        ############################################

        for i in range(num_solve):
            if is_user:
                counts_i = self.confidence[i].toarray() # shape: (1, N)
                # ... 2D: (1, N)
                ratings_i = self.ratings[i] # .toarray()
                # ... 1D: (N, )
                polarity_i = self.polarity[i].toarray()  # assuming self.polarity is a sparse matrix
                # ... 2D: (N, )
            else: # is_item 
                counts_i = self.confidence[:, i].T.toarray()
                ratings_i = self.ratings[:, i].T # .toarray()
                polarity_i = self.polarity[:, i].T.toarray()
                # ... dim(ratings_i): n_users

            CuI = sparse.diags(counts_i, [0])  # per-user or per-item diagonal matrix
            
            ## this is what xu dot yi tries to approximate
            # pu = counts_i.copy()  # if pu repr. preferences {0, 1} ... 
            # pu[np.where(pu != 0)] = 1.0  # ... then preferred if count > 0

            pu = ratings_i.copy() # treat pu as 'rating'
            # ... 1D array (because ratings is a dense array, slicing operation leads to 1D array)
            
            sign_u = polarity_i.copy()  # 2D: 1-by-n 
            # ... 2D: (1, ni) // since polarity is a sparse matrix -> toarray() => sign_u becomes a 1 * ni  (2D array)
            
            sign_u = sign_u.flatten() 
            # ... now sign_u is 1D // since pu is a 1D array (combing from slicing of dense array), sign_u has to be flattened to 1D as well

            # overwrites ratings
            # assert pu.shape == sign_u.shape, "dim(pu): {}, dim(sign_u): {}".format(pu.shape, sign_u.shape)
            pu[np.where(sign_u == ctp)] = self.pos_label  # TPs
            pu[np.where(sign_u == ctn)] = self.neg_label  # TNs

            # [test]
            # assert np.all(counts_i.flatten()[np.where( (sign_u == cfn) | (sign_u == cfp) )] == 0), \
            #           f"zero(counts_i): {np.sum(counts_i==0)}\n{counts_i}\n" # ... ok

            if add_bias: pu = pu - gamma if is_user else pu - beta 
            
            # Y'CuY => Y'Y + Y'(Cu-I)Y
            YTCuIY = fixed_vecs.T.dot(CuI).dot(fixed_vecs)  # Y'(Cu-I)Y i.e. CuI ~ Cu - I in the paper, sparsity property makes it faster
            # ... (nf * ni) (ni * ni) (ni * nf) -> nf * nf

            YTCupu = fixed_vecs.T.dot(CuI + eye).dot(sparse.csr_matrix(pu).T)  # CuI + eye ~ Cu, sparsity property makes it faster
            # ... (nf * ni) (ni * ni) (ni * 1) => nf * 1

            # Y'Y + Y'(Cu-I)Y + lambda * I
            xu = spsolve(YTY + YTCuIY + lambda_eye, YTCupu)  # <<< this is the core
            solve_vecs[i] = xu
            # ... nf * 1
            
            # if i > 0 and i % 1000 == 0:
            #     print('Solved %d vecs in %d seconds' % (i, time.time() - t))
            #     t = time.time()

        # if add_bias:
        #     if is_user: 
        #         solve_vecs = sparse.hstack([ beta.reshape(self.num_users, 1) , solve_vecs])
        #     else: 
        #         solve_vecs = sparse.hstack([ gamma.reshape(self.num_items, 1) , solve_vecs])

        return solve_vecs 

    def iter_preference_polarized(self, is_user, fixed_vecs, it=0, add_bias=False, beta=None, gamma=None):
        """

        Memo
        ----
        1. this is the original ALS routine in implicit CF
        2. say we fix item vectors and solve for user vectors 
           
           each user vector (i) is solved in each iteration 

           CuI is a diagonalized matrix of ui
    


        """
        # fixed_vecs is either X (user latent matrix, m by f) or Y (item latent matrix, n by f) 

        num_solve = self.num_users if is_user else self.num_items
        num_fixed = fixed_vecs.shape[0]
        # ... if add_bias is True, fixed_vecs will have an additional column of 1's padded in front 
        #     ni remains the same; nf <- nf + 1 
        #     dim(fixed_vecs): ni * nf => ni * (nf+1)

        YTY = fixed_vecs.T.dot(fixed_vecs)  # ~XTX for item loop
        # ... nf * nf

        eye = sparse.eye(num_fixed)

        num_factors = self.num_factors+1 if add_bias else self.num_factors
        lambda_eye = self.reg_param * sparse.eye(num_factors)  # lambda * I

        solve_vecs = np.zeros((num_solve, num_factors))

        missing_value = self.fill # 0

        t = time.time()
        for i in range(num_solve):  # n_users or n_items
            if is_user:
                confidence_i = self.confidence[i].toarray()   # assuming self.confidence is a sparse matrix
                polarity_i = self.polarity[i].toarray()  # assuming self.polarity is a sparse matrix
                # ... if polarity is in dense format, then polarity[i] becomes a 1D array
                #     however, confidence_i will be in 2D format assuming confidence is a sparse matrix
                #     => need to use polarity_i[None, :] to turn 1D into 2D if polarity_i is in 1D

                # if it == 10: print('(iter_preference) confidence[{}] | n(zeros)[{}]: {}'.format( i, i, np.sum(confidence_i==missing_value) ))
            else:
                confidence_i = self.confidence[:, i].T.toarray()
                polarity_i = self.polarity[:, i].T.toarray()

            # CuI ~ Cu (in user loop) or Ci in item-loop, see the notational convention in [2]
            CuI = sparse.diags(confidence_i, [0])   # put confidence_i components in the diagonal
            # ... cui = 1 + alpha * rui => min(cui) = 1
            # ... if we want cui to be just 0 for incorrect classifier predictions, then perhaps ... 
            # ... cui <- alpha * rui
            #     => CuI = CuI - eye
            
            # compute p(u) (or p(i))
            pu = confidence_i.copy()  # pu is a function of C 
            # ... 1 * ni  (2D array)
            sign_u = polarity_i.copy()  

            # ... pu is a 2D array (1, N), sign_u will also have to be 2D
            # print('(iter_preference_polarized) dim(pu): {}, dim(sign_u): {}'.format(pu.shape, sign_u.shape))

            # preferr correct predictions
            pu[np.where(sign_u > 0)] = self.positive_pref   # confidence score > 0 => correct predictions => preference <- 1
            pu[np.where(sign_u <= 0)] = self.negative_pref  # confidence score <= 0 => wrong predictions => preference <- 0   
            # ... preference can be in {0, 1}- or {-1, 1} representation 
            # ...... if pref <- {-1, 1}, then expect x't to approximate {-1, 1}

            if add_bias: pu = pu - gamma if is_user else pu - beta 

            # but we want the weight to be > 0, correct the sign    

            # [test]
            # assert len(pu[np.where(pu == missing_value)]) > 0
            
            ### this CuI ~ C - I in the paper because C = alpha * R  (rather than 1 + alpha * R)
            YTCuIY = fixed_vecs.T.dot(CuI).dot(fixed_vecs)  # Y'CuY
            # ... (nf * ni) (ni * ni) (ni * nf) -> nf * nf

            YTCupu = fixed_vecs.T.dot(CuI + eye).dot(sparse.csr_matrix(pu).T)  # Y'Cu pu
            # ... (nf * ni) (ni * ni) (ni * 1) => nf * 1

            xu = spsolve(YTY + YTCuIY + lambda_eye, YTCupu)
            solve_vecs[i] = xu
            # if i % 1000 == 0:
            #     print('Solved %i vecs in %d seconds' % (i, time.time() - t))
            #     t = time.time()

        return solve_vecs # if add_bias, then solve_vecs has an additional dimension beta(u) or gamma(i)

    def iter_preference(self, is_user, fixed_vecs, it=0, add_bias=False, beta=None, gamma=None):
        """

        Memo
        ----
        1. this is the original ALS routine in implicit CF
        2. say we fix item vectors and solve for user vectors 
           
           each user vector (i) is solved in each iteration 

           CuI is a diagonalized matrix of ui
    


        """
        # fixed_vecs is either X (user latent matrix, m by f) or Y (item latent matrix, n by f) 

        num_solve = self.num_users if is_user else self.num_items
        num_fixed = fixed_vecs.shape[0]

        if add_bias: fixed_vecs = sparse.hstack( [np.ones(num_fixed, 1), fixed_vecs] )
        YTY = fixed_vecs.T.dot(fixed_vecs)  # ~XTX for item loop

        eye = sparse.eye(num_fixed)
        lambda_eye = self.reg_param * sparse.eye(self.num_factors)  # lambda * I
        solve_vecs = np.zeros((num_solve, self.num_factors))

        missing_value = self.fill # 0

        t = time.time()
        for i in range(num_solve):  # n_users or n_items
            if is_user:
                counts_i = self.confidence[i].toarray()
                # if it == 10: print('(iter_preference) confidence[{}] | n(zeros)[{}]: {}'.format( i, i, np.sum(counts_i==missing_value) ))
            else:
                counts_i = self.confidence[:, i].T.toarray()

            # CuI ~ Cu (in user loop) or Ci in item-loop, see the notational convention in [2]
            CuI = sparse.diags(counts_i, [0])   # put count_i components in the diagonal
            # ... cui = 1 + alpha * rui => min(cui) = 1
            # ... if we want cui to be just 0 for incorrect classifier predictions, then perhaps ... 
            # ... cui <- alpha * rui
            #     => CuI = CuI - eye
            
            # compute p(u) (or p(i))
            pu = counts_i.copy()  # pu is a function of C
            pu[np.where(pu != 0)] = 1.0   # if counts > 0, then preference <- 1; if counts == 0, then preference <- 0   
            if add_bias: pu = pu - beta if is_user else pu - gamma

            # preferr correct predictions
            # pu[np.where(pu > 0)] = 1.0   # confidence score > 0 => correct predictions => preference <- 1
            # pu[np.where(pu < 0)] = 0.0   # confidence score < 0 => wrong predictions => preference <- 0   

            # but we want the weight to be > 0, correct the sign    

            # [test]
            # assert len(pu[np.where(pu == missing_value)]) > 0
            
            ### this CuI ~ C - I in the paper because C = alpha * R  (rather than 1 + alpha * R)
            YTCuIY = fixed_vecs.T.dot(CuI).dot(fixed_vecs)  # Y'CuY
            YTCupu = fixed_vecs.T.dot(CuI + eye).dot(sparse.csr_matrix(pu).T)  # Y'Cu pu
            xu = spsolve(YTY + YTCuIY + lambda_eye, YTCupu)
            solve_vecs[i] = xu
            # if i % 1000 == 0:
            #     print('Solved %i vecs in %d seconds' % (i, time.time() - t))
            #     t = time.time()

        if add_bias:
            if is_user: 
                solve_vecs = sparse.hstack([ beta.reshape(self.num_users, 1) , solve_vecs])
            else: 
                solve_vecs = sparse.hstack([ gamma.reshape(self.num_items, 1) , solve_vecs])

        return solve_vecs

    def iter_label(self, is_user, fixed_vecs, it=0): 
        # fixed_vecs is either X (user latent matrix, m by f) or Y (item latent matrix, n by f) 

        num_solve = self.num_users if is_user else self.num_items
        num_fixed = fixed_vecs.shape[0]
        YTY = fixed_vecs.T.dot(fixed_vecs)  # ~XTX for item loop
        eye = sparse.eye(num_fixed)
        lambda_eye = self.reg_param * sparse.eye(self.num_factors)  # lambda * I
        solve_vecs = np.zeros((num_solve, self.num_factors))

        t = time.time()
        for i in range(num_solve):  # n_users or n_items
            if is_user:
                counts_i = self.confidence[i].toarray()  # ith user and its associated confidence scores
                labels_i = self.labels
            else:
                counts_i = self.confidence[:, i].T.toarray()
                labels_i = np.repeat(self.labels[i], self.confidence.shape[0]) # the same item references the same label

            # CuI ~ Cu (in user loop) or Ci in item-loop, see the notational convention in [2]
            CuI = sparse.diags(counts_i, [0])   # put count_i components in the diagonal
            
            # compute p(u) (or p(i))
            pu = labels_i.copy()
            # pu[np.where(pu != 0)] = 1.0
            
            YTCuIY = fixed_vecs.T.dot(CuI).dot(fixed_vecs)  # Y'CuY
            YTCupu = fixed_vecs.T.dot(CuI + eye).dot(sparse.csr_matrix(pu).T)  # Y'Cu pu
            xu = spsolve(YTY + YTCuIY + lambda_eye, YTCupu)
            solve_vecs[i] = xu
            # if i % 1000 == 0:
            #     print('Solved %i vecs in %d seconds' % (i, time.time() - t))
            #     t = time.time()

        return solve_vecs

    def iter_label_regularized(self, is_user, fixed_vecs, it=0):
        """

        Memo
        ----
        1. No sparsity in this solution since we consider confidence and its complement (label confidence)

        """

        # fixed_vecs is either X (user latent matrix, m by f) or Y (item latent matrix, n by f) 
        # assert self.label_confidence is not None

        num_solve = self.num_users if is_user else self.num_items
        num_fixed = fixed_vecs.shape[0]
        
        # in general, there's no sparsity to exploit
        # YTY = fixed_vecs.T.dot(fixed_vecs)  # ~XTX for item loop
        # eye = sparse.eye(num_fixed)
        
        lambda_eye = self.reg_param * sparse.eye(self.num_factors)  # lambda * I
        solve_vecs = np.zeros((num_solve, self.num_factors))

        t = time.time()
        for i in range(num_solve):  # n_users or n_items
            if is_user:
                confidence_i = self.confidence[i].toarray()  # Cu, Ci
                label_confidence_i = self.label_confidence[i].toarray()  # a typical choice would be the complement of 'confidence'
 
                # trade off between approximating probability and true labels via confidence and label_confidence
                ratings_i = self.ratings[i] # .toarray()
                labels_i = self.labels
                
            else:
                confidence_i = self.confidence[:, i].T.toarray()
                label_confidence_i = self.label_confidence[:, i].T.toarray()

                ratings_i = self.ratings[:, i].T 
                labels_i = np.repeat(self.labels[i], self.confidence.shape[0])  # the same item references the same label

            # CuI ~ Cu (in user loop) or Ci in item-loop, see the notational convention in [2]
            # [log]
            # ... confidence_i: (1, 3979), label_confidence_i: (1, 3979)
            # ... confidence_i: [[0.    0.    0.    ... 5.786 0.    0.   ]], label_confidence_i: [[70.883 69.437 60.757 ...  0.     1.447  2.893]]
            # print('... confidence_i: {0}, label_confidence_i: {1}'.format(confidence_i.shape, label_confidence_i.shape))
            # print('... confidence_i: {0}, label_confidence_i: {1}'.format(confidence_i[:10], label_confidence_i[:10]))
            # sys.exit(0)

            # [log] LuI: Different number of diagonals and offsets
            # nd = label_confidence_i.size
            # CuI = np.diag(confidence_i) 
            # LuI = np.diag(label_confidence_i)
            # assert CuI.shape == LuI.shape

            assert confidence_i.shape == label_confidence_i.shape, "dim: {0} <> {1}".format(confidence_i.shape, label_confidence_i.shape)
            CuI = sparse.diags(confidence_i, [0])  # put count_i components in the diagonal, 0 off diag 
            try: 
                LuI = sparse.diags(label_confidence_i, [0])
            except Exception as e:
                print('... error in sparse.diags: %s' % e) 
                # LuI = sp.sparse.spdiags(label_confidence_i, 0, nd, nd)
                LuI = sparse.diags(np.diag(label_confidence_i))
            # print('... dim(CuI): {0}, dim(LuI): {1}'.format(CuI.shape, LuI.shape))
            assert CuI.shape == LuI.shape

            CuI_prime = CuI+LuI # sparse.csr_matrix(CuI+LuI) 
            
            # compute p(u) and l(u)
            # ... this is what dot(xu, yi) attempts to approximate
            pu = ratings_i.copy()
            lu = labels_i.copy()
            
            YTCuIY = fixed_vecs.T.dot(CuI_prime).dot(fixed_vecs)  # Y'CuY
            YTCupu = fixed_vecs.T.dot(CuI).dot(sparse.csr_matrix(pu).T)  # Y'Cu pu
            YTLulu = fixed_vecs.T.dot(LuI).dot(sparse.csr_matrix(lu).T)   # Y'Lu lu

            xu = spsolve(YTCuIY + lambda_eye, YTCupu + YTLulu)
            solve_vecs[i] = xu
            # if i % 1000 == 0:
            #     print('Solved %i vecs in %d seconds' % (i, time.time() - t))
            #     t = time.time()

        return solve_vecs

### end class ImplicitMF

# Confidence matrix is alpha * R, not 1 + alpha * R (therefore it corresponds to C-I in the paper)
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
            print('loaded %i counts...' % i)
    alpha = num_zeros / total
    print('alpha %.2f' % alpha)
    counts *= alpha
    counts = counts.tocsr()
    t1 = time.time()
    print('Finished loading matrix in %f seconds' % (t1 - t0))
    return counts

# ALS.b: Victor
def load_data(inputdir=None): 
    """

    Reference
    ---------

    """
    import scipy.sparse as sparse
    # from scipy.sparse.linalg import spsolve
    # from sklearn.preprocessing import MinMaxScaler

    #-------------------------
    # LOAD AND PREP THE DATA
    #-------------------------
    input_file = 'usersha1-artmbid-artname-plays.tsv'
    if inputdir is None: inputdir = System.ProjectPath
    input_path = os.path.join(inputdir, input_file) # 'data/usersha1-artmbid-artname-plays.tsv'

    raw_data = pd.read_table(input_path)
    raw_data = raw_data.drop(raw_data.columns[1], axis=1)
    raw_data.columns = ['user', 'artist', 'plays']
 
    # Drop rows with missing values
    data = raw_data.dropna()
  
    # Convert artists names into numerical IDs
    data['user_id'] = data['user'].astype("category").cat.codes
    data['artist_id'] = data['artist'].astype("category").cat.codes
 
    # Create a lookup frame so we can get the artist names back in 
    # readable form later.
    item_lookup = data[['artist_id', 'artist']].drop_duplicates()
    item_lookup['artist_id'] = item_lookup.artist_id.astype(str)
 
    data = data.drop(['user', 'artist'], axis=1)
 
    # Drop any rows that have 0 plays
    data = data.loc[data.plays != 0]
 
    # Create lists of all users, artists and plays
    users = list(np.sort(data.user_id.unique()))
    artists = list(np.sort(data.artist_id.unique()))
    plays = list(data.plays)
 
    # Get the rows and columns for our new matrix
    rows = data.user_id.astype(int)
    cols = data.artist_id.astype(int)
 
    # Contruct a sparse matrix for our users and items containing number of plays
    # R
    data_sparse = sparse.csr_matrix((plays, (rows, cols)), shape=(len(users), len(artists)))

    return data_sparse, item_lookup

def applyMF(fold, **kargs):
    raise NotImplementedError

def calculate_mse(model, ratings, user_index=None):
    # from sklearn.metrics import mean_squared_error
    preds = model.predict_for_customers()
    if user_index:
        return mean_squared_error(ratings[user_index, :].toarray().ravel(),
                                  preds[user_index, :].ravel())
    
    return mean_squared_error(ratings.toarray().ravel(),
                              preds.ravel())

def implicit_als(Cui, features=20, iterations=20, lambda_val=0.8, **kargs):
    """

    Memo
    ----
    1. https://github.com/MrChrisJohnson/implicit-mf/blob/master/mf.py

    2. ALS: 
          http://dsnotes.com/post/2017-05-28-matrix-factorization-for-recommender-systems/

    """
    R = kargs.get('ratings', None) 
    if R is not None: assert Cui.shape == R.shape, "dim(Cui):{0}, dim(R):{1}".format(Cui.shape, R.shape)
    L = kargs.get('labels', [])
    policy = kargs.get('policy', 'rating')
    Lui = kargs.get('label_confidence', None) # obsolete 
    polarity = kargs.get('polarity', None)   
    p_threshold = kargs.get('p_threshold', []) 
    add_bias = kargs.get('add_bias', False)

    # [test]
    # if policy.startswith('trade'):  # tradeoff between approx. probabilities and labels
    #     assert Lui is not None
    assert not policy.startswith('trade'), "Obsolete"
    if polarity is not None: 
        assert Cui.shape == polarity.shape, "Dimension inconsistency | dim(Cui): {}, dim(polarity): {}".format(Cui.shape, polarity.shape)
    if R is not None: 
        assert R.shape == Cui.shape

    print('(implicit_als) iteration policy: {routine} | L given? {supervised} | ret training error? {ret_type} | n_iter={n_iter}, lambda={l} | caller msg: {msg}'.format(routine=policy, 
            supervised=True if len(L) > 0 else False, ret_type=kargs.get('ret_rmse', False), n_iter=iterations, l=lambda_val, msg=kargs.get('message', 'n/a') ))

    positive_pref, negative_pref = kargs.get('positive_pref', 1.0), kargs.get('negative_pref', 0.0)
    ####################################################
    imf = ImplicitMF(Cui, 
            # label_confidence=Lui,  
            polarity=polarity, 
            p_threshold=p_threshold,
            ratings=R, labels=L, 
                policy=policy, 
                positive_pref=positive_pref, negative_pref=negative_pref,
                    num_factors=features, num_iterations=iterations, reg_param=lambda_val)
    ####################################################

    uvec, ivec = kargs.get('user_vectors', None), kargs.get('item_vectors', None)
    resume_als = kargs.get('resume_als', False)
    if uvec is None and ivec is None: 
        # using regular ALS to train the model
        imf.train_model(add_bias=add_bias)   
    else:  
        # ALS is reduced to LS 

        # maybe num_iterations can be cut down? 
        if resume_als: 
            vec_name = '& '.join([name for name, vec in {'user_vec': uvec, 'item_vec': ivec}.items() if vec is not None])
            div("(implicit_als) learned vectors: {v} are to be used as init vectors in ALS ...".format(v=vec_name), symbol='#', border=2)
        imf.train_model_foldin(user_vectors=uvec, item_vectors=ivec, resume_als=resume_als)

    if kargs.get('ret_rmse', False):
        return (imf.user_vectors, imf.item_vectors, imf.train_errors, imf.train_errors_weighted)

    return (imf.user_vectors, imf.item_vectors)
def prediction_error(C, T, P, Q, fill=0): 
    return ImplicitMF.rmse_score2(C, T, P=P, Q=Q, fill=fill)

def nonzeros(m, row):
    for index in range(m.indptr[row], m.indptr[row+1]):
        yield m.indices[index], m.data[index]

def implicit_als_cg(Cui, features=20, iterations=20, lambda_val=0.1, **kargs):
    """

    Reference
    ---------
    1. https://www.benfrederickson.com/fast-implicit-matrix-factorization/


    """

    user_size, item_size = Cui.shape

    # init random values
    X = np.random.rand(user_size, features) * 0.01
    Y = np.random.rand(item_size, features) * 0.01

    Cui, Ciu = Cui.tocsr(), Cui.T.tocsr()
    R = kargs.get('R', None)

    # ALS
    if R is None: 
        for iteration in range(iterations):
            if iteration % 10 == 0: print('(ALS) iteration %d of %d ... ' % (iteration+1, iterations))

            # alternativing least square
            least_squares_cg(Cui, X, Y, lambda_val)  # updates X, Y in place
            least_squares_cg(Ciu, Y, X, lambda_val)
    else: 
        # P is ratings (R)
        for iteration in range(iterations):
            if iteration % 10 == 0: print('(ALS2) iteration %d of %d ... ' % (iteration+1, iterations))

            # alternativing least square
            least_squares_cg2( (Cui, R), X, Y, lambda_val)  # updates X, Y in place
            least_squares_cg2( (Ciu, R), Y, X, lambda_val)      
    
    return sparse.csr_matrix(X), sparse.csr_matrix(Y)
      
def least_squares_cg(Cui, X, Y, lambda_val, cg_steps=3):
    users, features = X.shape
    
    YtY = Y.T.dot(Y) + lambda_val * np.eye(features)

    for u in range(users):  # note: xrange() doesn't exist in Python3

        x = X[u]
        r = -YtY.dot(x)

        for i, confidence in nonzeros(Cui, u):
            r += (confidence - (confidence - 1) * Y[i].dot(x)) * Y[i]

        p = r.copy()

        rsold = r.dot(r)

        for it in range(cg_steps):
            Ap = YtY.dot(p)
            for i, confidence in nonzeros(Cui, u):
                Ap += (confidence - 1) * Y[i].dot(p) * Y[i]

            alpha = rsold / p.dot(Ap)
            x += alpha * p
            r -= alpha * Ap

            rsnew = r.dot(r)
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        X[u] = x

def least_squares_cg2(CR, X, Y, lambda_val, cg_steps=3):
    Cui, R = CR 
    users, features = X.shape
    
    YtY = Y.T.dot(Y) + lambda_val * np.eye(features)

    for u in range(users):  # note: xrange() doesn't exist in Python3

        x = X[u]
        r = -YtY.dot(x)

        for i, confidence in nonzeros(Cui, u):  # non-zero index and value
            score = R[u, i]
            # print('... confidence:{0}, Y[i].dot(x): {1}'.format(confidence, Y[i].dot(x)))  # confidence score, dot(xu, yi)
            # r += (confidence - (confidence - 1) * Y[i].dot(x)) * Y[i]  # Y[i]: (nf, )
            score_h = Y[i].dot(x)
            r += (confidence * (score - score_h) + score_h) * Y[i]

        print('... dim(r):{0}'.format(r.shape))  # dim: (n_factors, )
        p = r.copy()
        # p = R[u, :].copy() # treat pu as 'ratings' 

        rsold = r.dot(r)

        for it in range(cg_steps):
            Ap = YtY.dot(p)
            for i, confidence in nonzeros(Cui, u):
                Ap += (confidence - 1) * Y[i].dot(p) * Y[i]

            alpha = rsold / p.dot(Ap)
            x += alpha * p
            r -= alpha * Ap

            rsnew = r.dot(r)
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        X[u] = x

## alias 
# implicit_als = implicit_als_cg
#####################################################

# prediction interface 
def predict_by_factors0(P, Q, test_offset, canonicalize=True, epsilon=1e-9): 
    
    train_set_only = False
    if Q.shape[0] == test_offset: # Q: item latent factors
        # only the training split (R) were used to compute P and Q    
        train_set_only = True

    if train_set_only: 
        Qr = Q
        Qt = None 
        print('... dim(P): {0}, dim(Qr): {1}, dim(Qt): {2}'.format(P.shape, Qr.shape, 'N/A'))
    else: 
        Qr = Q[:test_offset, :]
        Qt = Q[test_offset:, :]   # 30 * 10, (768-x) * 10
        print('... dim(P): {0}, dim(Qr): {1}, dim(Qt): {2}'.format(P.shape, Qr.shape, Qt.shape))
    ### 
    Rh = Th = None

    # approximate training split R
    Rh = np.dot(P, Qr.T)
    if not isinstance(Rh, np.ndarray): 
        Rh = np.array(Rh.todense())
    if canonicalize: Rh = canonicalize_prob(Rh, name='Rh')

    if Qt is not None: 
        Th = np.dot(P, Qt.T)   # [todo] predict interface

        if not isinstance(Th, np.ndarray):
            Th = np.array(Th.todense())
        if canonicalize: Th = canonicalize_prob(Th, name='Th')

    return (Rh, Th)

def predict_by_factors(P, Q, canonicalize=True, epsilon=1e-9, name='Xh'): 

    # approximate training split R
    Rh = np.dot(P, Q.T)
    if not isinstance(Rh, np.ndarray): 
        Rh = np.array(Rh.todense())
    if canonicalize: Rh = canonicalize_prob(Rh, name=name)

    return Rh
# alias 
predict = predict_by_factors

def canonicalize_prob(A, name='', verbose=True, epsilon=1e-9):
    
    A[A > 1.0] = 1.0 - epsilon 
    A[A < 0.0] = 0.0 + epsilon

    if verbose: 
        if name: print('(canonicalize_prob) Matrix(%s) has illegal probabilities:' % name)
        n_overflow = np.sum(A > 1.0)
        n_underflow = np.sum(A < 0.0)
        if n_overflow > 0: 
            print('... %d entries with p > 1.0!' % n_overflow)
        if n_underflow > 0: 
            print('... %d entries with p < 0.0!' % n_underflow)
        print(f'...... Number of illegal probabilities: {n_overflow + n_underflow}')
    return A

def t_confidence_weights(test_=True): 
    def select(W, counter, topk=10):
        # for i in range(W.shape[0]):
        #     if not i in np.argsort(W)[::-1][:topk]:
        #        W[i] = 0
        counter.update(np.argsort(W)[::-1][:topk])
        assert all(W[np.argsort(W)[::-1][:topk]] != 0)
        print (counter)
        return counter

    import utils_cf as uc 
    from collections import Counter
    
    n_fold = 5
    missing_value = -1 # marker for missing data
    topk = 30

    ### Pearson correlation (pcorr) with the true labels
    topk = 30
    corrMetrics = Metrics()  # prediction-label correlation
    topKCorrMetrics = Metrics()
    
    topk = 10
    userCounter = Counter()  # the index of users/classifiers mostly selected (those that produce most correlated predictions across CV folds)
    for fold in range(n_fold): 
        if test_ and fold > 2: break
        data = uc.toPredictiveScores(fold)
        R = data['train']
        T = data['test']
        L_train, L_test = data['train_labels'], data['test_labels']
        U = data['users']
        print('(test) n_train: %d, n_test: %d' % (len(L_test), len(L_test)))
        n_users, n_items = R.shape[0], R.shape[1]
    
        tu, ti = 5, 10
        Wu = uc.confidence(R, L_train, T=None, mode='users', topk=topk, scoring=brier_score_loss)
        Wi = uc.confidence(R, L_train, mode='items', topk=None, scoring=brier_score_loss)

        print('(test 1): Wu (n=%d):\n%s\n' % (len(Wu), str(Wu)))
        print('(test 1): Wi (n=%d):\n%s\n' % (len(Wi), str(Wi[:ti])) )
        userCounter = select(Wu, userCounter, topk=topk)
        # W = np.outer(Wu, Wi)

        W = uc.confidence2D(R, L_train, T=None, mode='label', topk=(0, 0), scoring=brier_score_loss)
        print('(test 1b) W:\n%s\n' % W[:tu][:ti])
 
    user_cnt = userCounter.most_common(10)
    print('(test) most selected classifiers:\n%s\n' % user_cnt)
    uids = [u for u, c in user_cnt]
    print('       they are:\n%s\n' % U[uids])

    return 

def t_ensemble(base_perf=None): 
    """
    Use NMF factorized latent matrices to compute similarity
    """ 
    import math
    import utils_cf as uc
    from cf import base_predictors, toRatingMatrix, analyzePerf

    n_fold = 5
    missing_value = 0 # marker for missing data
    p_th = 0.5

    n_users = n_items = 0 
    n_factors = 10
    n_epochs = 300
    topk = 30

    # output
    ret = {}

    # find the performance of the baseline methods; this produces a PerformanceMetrics containing a table (cols: classifiers, rows indexed by metrics)
    if base_perf is None: 
        base_perf = base_predictors()  # consolidated PerformanceMetrics object across CV fold

    # bpMetrics, fullMetrics, topKMetrics = Metrics(), Metrics(), Metrics() # matrix factorization metrics
    fullMetrics, topKMetrics = [], []
    nmfCV, topKNMFCV = [], []
    for fold in range(n_fold): 
     
        # different than toPredictiveScores(), toRatingMatrix() masks FP, FN and possible implement other 
        # strategies that utilize the ground truths 
        R, T, L_train, L_test, U = toRatingMatrix(fold, p_threshold=p_th, missing_value=missing_value, verbose=True)
        n_users, n_items = R.shape[0], R.shape[1]
        print('[nmf_ensemble] dim(T): %s, n_test: %d' % (str(T.shape), len(L_test)))

        # [note] Surprise does not take inputs in the form of R (rating matrix)
        P, Q = uc.applyMF(fold, n_factors=n_factors, n_epochs=n_epochs, fill=missing_value)  # P and Q

        # but we only need the Q from the test split 
        Qt = Q[R.shape[1]:, :]   # 30 * 10, (768-x) * 10
        print('... dim(P): %s, dim(Qt): %s' % (str(P.shape), str(Qt.shape)))
        # Th = np.dot(P, Qt.T)

        # use P (user latent features) to construct Su 
        Su = uc.evalSimilarityByLatentFeatures(P, epsilon=1e-9)
        assert Su.shape[0] == Su.shape[1] == R.shape[0]
        print('(test) Sim (Su[i,j] in [0, 1]?):\n%s\n' % Su[:4, :4])
        Th0 = ucf.predict(T, Su, kind='user')

        # metrics = comparePerfMetrics0(T, L_test, Th=Th0, R=None, L_train=None)
        # bpMetrics.add(metrics['bp'])
        # fullMetrics.add(metrics['cf'])
        perf0 = analyzePerf(L_test, Th0, method='nmf_similarity', aggregate_func=np.mean)  # final prediction
        fullMetrics.append(perf0)
        nmfCV.append((L_test, combiner(Th0, aggregate_func=np.mean)))  # plot K-Fold CV

        # use top K only 
        topk = int(math.floor(n_users/2))
        Th_topK = ucf.predict_topk(T, Su, kind='user', k=topk)
        # metrics = comparePerfMetrics0(T, L_test, Th=Th_topK, R=None, L_train=None)
        # topKMetrics.add(metrics['cf'])
        perf1 = analyzePerf(L_test, Th_topK, method='nmf_similarity_top%d' % topk, aggregate_func=np.mean)  # final prediction
        topKMetrics.append(perf1)
        topKNMFCV.append((L_test, combiner(Th_topK, aggregate_func=np.mean)))

    # op: a combiner function for performance scores across CV folds
    # bpMetrics.report(op=np.mean, message='Average of best BP performance.')
    # fullMetrics.report(op=np.mean, message='Model-based ensemble based on NMF with full user-user similarity')
    # topKMetrics.report(op=np.mean, message='Model-based ensemble based on NMF with topk=%d' % topk)

    plot_roc(nmfCV, file_name='als-user-roc-%s' % System.domain)  # an import from evaluate
    plot_roc(topKNMFCV, file_name='als-top%d-user-roc-%s' % (topk, System.domain))

    ## Compare with baseline methods
    ret['als_sim'] = PerformanceMetrics.consolidate(fullMetrics)  # foreach metric, take average over CV folds
    ret['als_sim_top%d' % topk] = PerformanceMetrics.consolidate(topKMetrics)
    for metric in PerformanceMetrics.tracked: 
        ret[metric] = PerformanceMetrics.sort2(ret['nmf_sim'], metric=metric, verbose=True if metric in ['auc', ] else False) 

    # how does it compare to BP? 
    docs = {'method': 'ALS_SIM'}

    # rule: max => best vs best 
    PerformanceMetrics.report(p_baseline=base_perf, p_target=ret['als_sim'], rule='max', descriptions=docs, verbose=True) # metrics=[] => PerformanceMetrics.tracked
    
    return ret 

def t_als(): 
    inputdir = '/Users/pleiades/Documents/work/data/recommender/lastfm-dataset-360K'
    data_sparse = load_data(inputdir)
    print('(test) dim(data): %s' % str(data_sparse.shape))
    
    ### ALS ### 

    ## very slow ALS 
    # user_vecs, item_vecs = implicit_als_slow(data_sparse, iterations=20, features=20, alpha_val=40)
    
    ## conjugate gradient in ALS 
    alpha_val = 15
    conf_data = (data_sparse * alpha_val).astype('double')
    user_vecs, item_vecs = implicit_als_cg(conf_data, iterations=20, features=20) 

    print('(test) dim(U): %s, dim(V): %s' % (str(user_vecs.shape), str(item_vecs.shape)))


    ### Recommendataion 

    # Let's say we want to recommend artists for user with ID 2023
    user_id = 2023

    #------------------------------
    # GET ITEMS CONSUMED BY USER
    #------------------------------

    # Let's print out what the user has listened to
    # consumed_idx = data_sparse[user_id,:].nonzero()[1].astype(str)
    # consumed_items = item_lookup.loc[item_lookup.artist_id.isin(consumed_idx)]
    # print consumed_items

    # Let's generate and print our recommendations
    # recommendations = recommend(user_id, data_sparse, user_vecs, item_vecs, item_lookup)
    # print recommendations

    return

def test(): 

    ## experiment on confidence weights
    # t_confidence_weights()

    ## slow ALS 
    t_als()

    return

if __name__ == "__main__": 
    test()
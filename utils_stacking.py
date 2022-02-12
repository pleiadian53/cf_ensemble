"""
Stacking classifier and regressor.

Reference
---------
[1] Scikit-learn's implementation
    https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/ensemble/_stacking.py

[2] Multi-layer stacking 
    https://github.com/stgrmks/StackingClassifier/blob/master/stacker.py

"""
# Authors: Barnett Chiu <barnettchiu@gmail.com>
# Reference: Guillaume Lemaitre <g.lemaitre58@gmail.com>

import os, sys
from abc import ABCMeta, abstractmethod
from copy import deepcopy

import numpy as np
from joblib import Parallel
import scipy.sparse as sparse

from sklearn.base import clone
# from ..base import clone # Construct a new unfitted estimator with the same parameters.
# ... base is under sklearn


from sklearn.base import ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.base import is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._estimator_html_repr import _VisualBlock

from sklearn.ensemble._base import _fit_single_estimator
# ... _base is under sklearn.ensemble

from sklearn.ensemble._base import _BaseHeterogeneousEnsemble

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeCV

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import check_cv

from sklearn.preprocessing import LabelEncoder

from sklearn.utils import Bunch # Container object exposing keys as attributes.
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import column_or_1d
from sklearn.utils.fixes import delayed

# Saving large numpy arrays
# import h5py # pip install h5py

# Utilities
# from utils_sys import format_sort_dict
import utils_sys as us
import utils_classifier as uc


class _BaseStacking(TransformerMixin, _BaseHeterogeneousEnsemble, metaclass=ABCMeta):
    """Base class for stacking method."""

    @abstractmethod
    def __init__(
        self,
        estimators,
        final_estimator=None,
        *,
        cv=None,
        stack_method="auto",
        n_jobs=None,
        verbose=0,
        passthrough=False,
    ):
        super().__init__(estimators=estimators)
        self.final_estimator = final_estimator
        self.cv = cv
        self.stack_method = stack_method
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.passthrough = passthrough

    def _clone_final_estimator(self, default):
        if self.final_estimator is not None:
            self.final_estimator_ = clone(self.final_estimator)
        else:
            self.final_estimator_ = clone(default)

    def _concatenate_predictions(self, X, predictions):
        """Concatenate the predictions of each first layer learner and
        possibly the input dataset `X`.

        If `X` is sparse and `self.passthrough` is False, the output of
        `transform` will be dense (the predictions). If `X` is sparse
        and `self.passthrough` is True, the output of `transform` will
        be sparse.

        This helper is in charge of ensuring the predictions are 2D arrays and
        it will drop one of the probability column when using probabilities
        in the binary case. Indeed, the p(y|c=0) = 1 - p(y|c=1)
        """
        X_meta = []
        for est_idx, preds in enumerate(predictions):
            # case where the the estimator returned a 1D array
            if preds.ndim == 1:
                X_meta.append(preds.reshape(-1, 1))
            else:
                if (
                    self.stack_method_[est_idx] == "predict_proba"
                    and len(self.classes_) == 2
                ):
                    # Remove the first column when using probabilities in
                    # binary classification because both features are perfectly
                    # collinear.
                    X_meta.append(preds[:, 1:])
                else:
                    X_meta.append(preds)
        if self.passthrough:
            X_meta.append(X) # include raw features
            if sparse.issparse(X):
                return sparse.hstack(X_meta, format=X.format)

        return np.hstack(X_meta)

    @staticmethod
    def _method_name(name, estimator, method):
        if estimator == "drop":
            return None
        if method == "auto":
            if getattr(estimator, "predict_proba", None):
                return "predict_proba"
            elif getattr(estimator, "decision_function", None):
                return "decision_function"
            else:
                return "predict"
        else:
            if not hasattr(estimator, method):
                raise ValueError(
                    "Underlying estimator {} does not implement the method {}.".format(
                        name, method
                    )
                )
            return method

    def fit(self, X, y, sample_weight=None):
        """
        Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,) or default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

            .. versionchanged:: 0.23
               when not None, `sample_weight` is passed to all underlying
               estimators

        Returns
        -------
        self : object
        """
        # all_estimators contains all estimators, the one to be fitted and the
        # 'drop' string.
        names, all_estimators = self._validate_estimators()
        self._validate_final_estimator()

        stack_method = [self.stack_method] * len(all_estimators)

        # Fit the base estimators on the whole training data. Those
        # base estimators will be used in transform, predict, and
        # predict_proba. They are exposed publicly.
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_single_estimator)(clone(est), X, y, sample_weight)
            for est in all_estimators
            if est != "drop"
        )

        self.named_estimators_ = Bunch() # so that we can access named_estimators_'s attributes by key
        est_fitted_idx = 0
        for name_est, org_est in zip(names, all_estimators):
            if org_est != "drop":
                current_estimator = self.estimators_[est_fitted_idx]
                self.named_estimators_[name_est] = current_estimator
                est_fitted_idx += 1
                if hasattr(current_estimator, "feature_names_in_"):
                    self.feature_names_in_ = current_estimator.feature_names_in_
            else:
                self.named_estimators_[name_est] = "drop" 

        # To train the meta-classifier using the most data as possible, we use
        # a cross-validation to obtain the output of the stacked estimators.

        # To ensure that the data provided to each estimator are the same, we
        # need to set the random state of the cv if there is one and we need to
        # take a copy.
        cv = check_cv(self.cv, y=y, classifier=is_classifier(self))
        if hasattr(cv, "random_state") and cv.random_state is None:
            cv.random_state = np.random.RandomState()

        self.stack_method_ = [
            self._method_name(name, est, meth)
            for name, est, meth in zip(names, all_estimators, stack_method)
        ]
        fit_params = (
            {"sample_weight": sample_weight} if sample_weight is not None else None
        )

        # Run `cross_val_predict` within all CV-folds in parallel
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(cross_val_predict)(
                clone(est),
                X,
                y,
                cv=deepcopy(cv),
                method=meth,
                n_jobs=self.n_jobs,
                fit_params=fit_params,
                verbose=self.verbose,
            )
            for est, meth in zip(all_estimators, self.stack_method_) # self.stack_method_ is a list of method names
            if est != "drop"
        )

        # Only not None or not 'drop' estimators will be used in transform.
        # Remove the None from the method as well.
        self.stack_method_ = [
            meth
            for (meth, est) in zip(self.stack_method_, all_estimators)
            if est != "drop"
        ]

        # Design: Either treat level-1 predictions/features from base predictors collectively as a new representation or treat
        #         these predictions as new features added on top of the raw variables at the base level
        X_meta = self._concatenate_predictions(X, predictions)
        _fit_single_estimator(
            self.final_estimator_, X_meta, y, sample_weight=sample_weight
        )

        return self

    @property
    def n_features_in_(self):
        """Number of features seen during :term:`fit`."""
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute n_features_in_"
            ) from nfe
        return self.estimators_[0].n_features_in_

    def _transform(self, X):
        """Concatenate and return the predictions of the estimators."""
        
        check_is_fitted(self)
        # ... Checks if the estimator is fitted by verifying the presence of
        #     fitted attributes (ending with a trailing underscore) and otherwise
        #     raises a NotFittedError with the given message.

        predictions = [
            getattr(est, meth)(X)
            for est, meth in zip(self.estimators_, self.stack_method_)
            if est != "drop"
        ]
        return self._concatenate_predictions(X, predictions)

    @if_delegate_has_method(delegate="final_estimator_")
    def predict(self, X, **predict_params):
        """Predict target for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        **predict_params : dict of str -> obj
            Parameters to the `predict` called by the `final_estimator`. Note
            that this may be used to return uncertainties from some estimators
            with `return_std` or `return_cov`. Be aware that it will only
            accounts for uncertainty in the final estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_output)
            Predicted targets.
        """
        check_is_fitted(self)
        return self.final_estimator_.predict(self.transform(X), **predict_params)

    def _sk_visual_block_(self, final_estimator):
        names, estimators = zip(*self.estimators)
        parallel = _VisualBlock("parallel", estimators, names=names, dash_wrapped=False)

        # final estimator is wrapped in a parallel block to show the label:
        # 'final_estimator' in the html repr
        final_block = _VisualBlock(
            "parallel", [final_estimator], names=["final_estimator"], dash_wrapped=False
        )
        return _VisualBlock("serial", (parallel, final_block), dash_wrapped=False)

############################################################################################

class _BaseCF(_BaseStacking): 
    """
    Base class for stacking based on collaborative filtering methods. 


    Note 
    ---- 
    1. transform() -> 
    """

    @abstractmethod
    def __init__(
        self,
        estimators,
        final_estimator=None,
        *,
        cv=None,
        stack_method="predict_proba",
        n_jobs=None,
        verbose=0,
        passthrough=False,
        
        # parameters for data and model persistence
        save_itermediate_data=True, # [new]
        data_dir='data', 
        model_dir='model', 
        work_dir = '.',     
        fold_number=0, 
        
    ):
        super().__init__(estimators=estimators)
        self.final_estimator = None # this is going to be a CF method
        self.cv = cv
        self.stack_method = stack_method
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.passthrough = passthrough

        # Save pre-computed probability matrix (X_meta)
        self.save_itermediate_data = save_itermediate_data
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.work_dir = work_dir # [note] all arguments need be also part of the class attribues
        self.fold_number = fold_number

        # Run other initilization logics
        self.init()    
       
    def init(self):

        work_dir = self.work_dir

        # Configure directories for data and model persistence
        for d in ['data_dir', 'model_dir', ]:
            dirtype = d.split('_')[0]
            dir_ = getattr(self, d)
            if not os.path.exists(dir_):  
                assert os.path.exists(work_dir), f"Invalid working directory:\n{work_dir}\ndoes not exist."
                dir_ = os.path.join(work_dir, dir_)
                if not os.path.exists(dir_): 
                    if self.verbose: print(f"(BaseCF) Creating {dirtype} directory at:\n{dir_}\n")
                    os.mkdir(dir_)

                setattr(self, d, dir_) 
            # assert os.path.exists(dir_), f"[error] {dir_} does not exist." 

        # if not os.path.exists(data_dir):  
        #     self.data_dir = os.path.join(work_dir, data_dir)
        #     assert os.path.exists(self.data_dir), f"[error] {self.data_dir} does not exist."
        # if not os.path.exists(model_dir): 
        #     self.model_dir = os.path.join(work_dir, model_dir)
        #     assert os.path.exists(self.model_dir), f"[error] {self.model_dir} does not exist."  

    def fit(self, X, y, sample_weight=None):
        """
        Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,) or default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

            .. versionchanged:: 0.23
               when not None, `sample_weight` is passed to all underlying
               estimators

        Returns
        -------
        self : object
        """
        # all_estimators contains all estimators, the one to be fitted and the
        # 'drop' string.
        names, all_estimators = self._validate_estimators()
        self._validate_final_estimator()

        stack_method = [self.stack_method] * len(all_estimators)

        # Fit the base estimators on the whole training data. Those
        # base estimators will be used in transform, predict, and
        # predict_proba. They are exposed publicly.
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_single_estimator)(clone(est), X, y, sample_weight)
            for est in all_estimators
            if est != "drop"
        )

        self.named_estimators_ = Bunch() # so that we can access named_estimators_'s attributes by key
        est_fitted_idx = 0
        for name_est, org_est in zip(names, all_estimators):
            print(f"(BaseCF) base est | name: {name_est}, estimator: {org_est}")
            if org_est != "drop":
                current_estimator = self.estimators_[est_fitted_idx]
                self.named_estimators_[name_est] = current_estimator
                est_fitted_idx += 1
                if hasattr(current_estimator, "feature_names_in_"):
                    self.feature_names_in_ = current_estimator.feature_names_in_
            else:
                self.named_estimators_[name_est] = "drop" 

        # [todo]
        named_estimators_dict = {}
        for name, model in  self.named_estimators_.items(): 
            named_estimators_dict[name] = str(model)
            # assert uc.is_fitted(model), f"{model} has not been fitted" # [ok]
        print(f"(BaseCF) Base predictors:\n{us.format_sort_dict(named_estimators_dict)}\n") # check_is_fitted()
        # ... At this point, all base predictors are fully trained

        # To train the meta-classifier using the most data as possible, we use
        # a cross-validation to obtain the output of the stacked estimators.

        # To ensure that the data provided to each estimator are the same, we
        # need to set the random state of the cv if there is one and we need to
        # take a copy.
        cv = check_cv(self.cv, y=y, classifier=is_classifier(self))
        if hasattr(cv, "random_state") and cv.random_state is None:
            cv.random_state = np.random.RandomState()

        self.stack_method_ = [
            self._method_name(name, est, meth)
            for name, est, meth in zip(names, all_estimators, stack_method)
        ]
        fit_params = (
            {"sample_weight": sample_weight} if sample_weight is not None else None
        )

        # Transform the training data into next-level representation
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(cross_val_predict)(
                clone(est),
                X,
                y,
                cv=deepcopy(cv),
                method=meth,
                n_jobs=self.n_jobs,
                fit_params=fit_params,
                verbose=self.verbose,
            )
            for est, meth in zip(all_estimators, self.stack_method_) # self.stack_method_ is a list of method names
            if est != "drop"
        )

        # Only not None or not 'drop' estimators will be used in transform.
        # Remove the None from the method as well.
        self.stack_method_ = [
            meth
            for (meth, est) in zip(self.stack_method_, all_estimators)
            if est != "drop"
        ]

        X_meta = self._concatenate_predictions(X, predictions)

        # By default, save a copy of X_meta for future CF-related operations 
        if self.save_itermediate_data: 
            fpath = os.path.join(self.data_dir, f'train-{self.fold_number}.npz')
            with open(fpath, 'wb') as f: 
                if self.verbose: print(f"[info] Saving X_meta (shape={X_meta.shape}) at:\n{fpath}\n")
                np.savez(f, X=X_meta, y=y, U=names)

        # [todo] final estimator is a CF method
        _fit_single_estimator(
            self.final_estimator_, X_meta, y, sample_weight=sample_weight
        )

        return self

############################################################################################

class CFStacker(ClassifierMixin, _BaseCF): 
    def __init__(
        self,
        estimators,
        final_estimator=None, # CF algorithm goes here
        *,
        cv=None,
        stack_method="predict_proba", # CF ensemble learning likely works better with base predictors producing probabilties (instead of binary labels)

        n_jobs=None,
        passthrough=False,        
        verbose=0,

        save_itermediate_data=True,
        fold_number = 0, 
        data_dir = 'data',
        model_dir = 'model', 
        work_dir = '.',

        overwrite_allowed=False, 

    ):
        super().__init__(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            stack_method=stack_method,
            n_jobs=n_jobs,

            passthrough=passthrough,
            verbose=verbose,

            # Parameters for data and model persistence 
            save_itermediate_data=save_itermediate_data, # save X_meta (i.e. save pre-computed probability matrix)
            data_dir=data_dir, 
            model_dir=model_dir, 
            work_dir=work_dir,
            fold_number=fold_number,
        )

        self.overwrite_allowed = overwrite_allowed # if True, allow overwriting meta data (see cf_write())

    # [todo] make CF predictor compatible 
    def _validate_final_estimator(self):
        self._clone_final_estimator(default=LogisticRegression())
        if not is_classifier(self.final_estimator_):
            raise ValueError(
                "'final_estimator' parameter should be a classifier. Got {}".format(
                    self.final_estimator_
                )
            )

    def fit(self, X, y, sample_weight=None):
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

        Returns
        -------
        self : object
            Returns a fitted instance of estimator.
        """
        check_classification_targets(y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_

        return super().fit(X, self._le.transform(y), sample_weight)

    def cf_fit(self, X=None, y=None, sample_weight=None): # [todo]
        return 

    def cf_fetch(self, fold_number=None):
        # Get base predictor-transformed training and test data (i.e. level-1 train and test data)
        if fold_number is None: fold_number = self.fold_number

        # Load pre-trained data
        metas = {}
        for dtype in ['train', 'test', ]: 
            f = os.path.join(self.data_dir, f'{dtype}-{fold_number}.npz')
            if os.path.exists(f): 
                metas[dtype] = dict(np.load(f)) 
        return metas 
    def load_meta(self, fold_number=None): # [alias]
        return self.cf_fetch(self, fold_number) 

    def cf_write(self, dtype='test', **kargs):
        """
        Write meta data into the CF dataset as specified by `dtype` {'training', 'dev', 'test'}

        Parameters
        ----------
        dtype: data set type with the following values 
               'training' or 'train' 
               'dev' 
               'test'
        """
        if not kargs:
            if self.verbose: print("(cf_write) Warning: No meta data given.")
            # no-op 
            return 

        # Normalize dtype to canonical values
        if dtype.startswith('tra'):
            dtype = 'train'
        elif dtype.startswith(('dev', 'val')): 
            dtype = 'validation'
        else: 
            dtype = 'test'

        # Overwrite instance parameters such as `fold_number` 
        # ... not recoomended; only useful in testing when you don't want multiple instances of CFSTacker
        fold_number = kargs.pop('fold_number', self.fold_number)
        #######################################################

        fpath = os.path.join(self.data_dir, f'{dtype}-{fold_number}.npz')
        data_set = dict(np.load(fpath))
        
        for k, v in kargs.items():
            if not k in data_set: 
                if self.verbose: print(f"(cf_write) Adding new attribute {k}:\n{v}\n...")
                data_set[k] = v
            else: 
                if self.overwrite_allowed: 
                    if self.verbose: print(f"(cf_write) Warning: overwriting existing attribute {k}.")
                    data_set[k] = v    

        with open(fpath, 'wb') as f: 
            if self.verbose: print(f"(cf_write) Saving X_meta at:\n{fpath}\n")
            np.savez(f, **data_set) # `y` (label) is unknown
    def write_meta(self, dtype='test', **kargs): # [alias]
        return self.cf_write(self, dtype, **kargs)

    def cf_predict(self, X=None, **kargs): # [todo]
        # if len(data) == 2: # predict new/test data
        #     pass 
        # if len(data) == 4: # include pre-trained level-1 training data and predict the test data
        #     pass 
        # X_train, y_train, X_test, y_test = load_metaset() # load meta data from cross_val_predict() 
        # load pre-trained data from disk 
        
        # Load pre-trained data
        X_train, y_train = kargs.get('X_train', None), kargs.get('y_train', None)
        if X_train is None: 
            f_train = os.path.join(self.data_dir, f'train-{self.fold_number}.npz')
            train_set = np.load(f_train)
            X_train, y_train = train_set['X'], train_set['y']
        
        # Also need the pre-trained base predictors when `X` is not provided
        X_test = X
        if X_test is None: # then we must have already apply trained base predictors to the test data and transform them into level-1
            f_test = os.path.join(self.data_dir, f'test-{self.fold_number}.npz')
            test_set = np.load(f_test)
            X_test = test_set['X']

        # [todo] CF logic 
        y_pred = np.zeros(X_test.shape[0])

        assert len(y_pred) == X_test.shape[0]
        return y_pred

    @if_delegate_has_method(delegate="final_estimator_")
    def predict(self, X, **predict_params):
        """Predict target for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        **predict_params : dict of str -> obj
            Parameters to the `predict` called by the `final_estimator`. Note
            that this may be used to return uncertainties from some estimators
            with `return_std` or `return_cov`. Be aware that it will only
            accounts for uncertainty in the final estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_output)
            Predicted targets.
        """
        y_pred = super().predict(X, **predict_params)
        return self._le.inverse_transform(y_pred)

    @if_delegate_has_method(delegate="final_estimator_")
    def predict_proba(self, X):
        """Predict class probabilities for `X` using the final estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_classes) or \
            list of ndarray of shape (n_output,)
            The class probabilities of the input samples.
        """
        check_is_fitted(self)
        return self.final_estimator_.predict_proba(self.transform(X))

    @if_delegate_has_method(delegate="final_estimator_")
    def decision_function(self, X):
        """Decision function for samples in `X` using the final estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        decisions : ndarray of shape (n_samples,), (n_samples, n_classes), \
            or (n_samples, n_classes * (n_classes-1) / 2)
            The decision function computed the final estimator.
        """
        check_is_fitted(self)
        return self.final_estimator_.decision_function(self.transform(X))

    def transform(self, X):
        """Return class labels or probabilities for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        y_preds : ndarray of shape (n_samples, n_estimators) or \
                (n_samples, n_classes * n_estimators)
            Prediction outputs for each estimator.
        """
        # By default, also save a copy of the transformed data 
        # ... this is useful as a preparatory step prior to making predictions on unseen data

        X_meta = self._transform(X)
        if self.save_itermediate_data: # Output of `transform` is considered as a "test split" by default
            fpath = os.path.join(self.data_dir, f'test-{self.fold_number}.npz')
            with open(fpath, 'wb') as f: 
                if self.verbose: print(f"[info] Saving X_meta (shape={X_meta.shape}) at:\n{fpath}\n")
                np.savez(f, X=X_meta) # `y` (label) is unknown

        # base predictors must have been fitted prior to this call
        return X_meta # -> _BaseCF._transform(X) -> _BaseCF._concatenate_predictions

    def _sk_visual_block_(self):
        # If final_estimator's default changes then this should be
        # updated.
        if self.final_estimator is None:
            final_estimator = LogisticRegression()
        else:
            final_estimator = self.final_estimator
        return super()._sk_visual_block_(final_estimator)

    

############################################################################################    

class StackingClassifier(ClassifierMixin, _BaseStacking):
    """Stack of estimators with a final classifier.

    Stacked generalization consists in stacking the output of individual
    estimator and use a classifier to compute the final prediction. Stacking
    allows to use the strength of each individual estimator by using their
    output as input of a final estimator.

    Note that `estimators_` are fitted on the full `X` while `final_estimator_`
    is trained using cross-validated predictions of the base estimators using
    `cross_val_predict`.

    Read more in the :ref:`User Guide <stacking>`.

    .. versionadded:: 0.22

    Parameters
    ----------
    estimators : list of (str, estimator)
        Base estimators which will be stacked together. Each element of the
        list is defined as a tuple of string (i.e. name) and an estimator
        instance. An estimator can be set to 'drop' using `set_params`.

    final_estimator : estimator, default=None
        A classifier which will be used to combine the base estimators.
        The default classifier is a
        :class:`~sklearn.linear_model.LogisticRegression`.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy used in
        `cross_val_predict` to train `final_estimator`. Possible inputs for
        cv are:

        * None, to use the default 5-fold cross validation,
        * integer, to specify the number of folds in a (Stratified) KFold,
        * An object to be used as a cross-validation generator,
        * An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and y is
        either binary or multiclass,
        :class:`~sklearn.model_selection.StratifiedKFold` is used.
        In all other cases, :class:`~sklearn.model_selection.KFold` is used.
        These splitters are instantiated with `shuffle=False` so the splits
        will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. note::
           A larger number of split will provide no benefits if the number
           of training samples is large enough. Indeed, the training time
           will increase. ``cv`` is not used for model evaluation but for
           prediction.

    stack_method : {'auto', 'predict_proba', 'decision_function', 'predict'}, \
            default='auto'
        Methods called for each base estimator. It can be:

        * if 'auto', it will try to invoke, for each estimator,
          `'predict_proba'`, `'decision_function'` or `'predict'` in that
          order.
        * otherwise, one of `'predict_proba'`, `'decision_function'` or
          `'predict'`. If the method is not implemented by the estimator, it
          will raise an error.

    n_jobs : int, default=None
        The number of jobs to run in parallel all `estimators` `fit`.
        `None` means 1 unless in a `joblib.parallel_backend` context. -1 means
        using all processors. See Glossary for more details.

    passthrough : bool, default=False
        When False, only the predictions of estimators will be used as
        training data for `final_estimator`. When True, the
        `final_estimator` is trained on the predictions as well as the
        original training data.

    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels.

    estimators_ : list of estimators
        The elements of the estimators parameter, having been fitted on the
        training data. If an estimator has been set to `'drop'`, it
        will not appear in `estimators_`.

    named_estimators_ : :class:`~sklearn.utils.Bunch`
        Attribute to access any fitted sub-estimators by name.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying classifier exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimators expose such an attribute when fit.
        .. versionadded:: 1.0

    final_estimator_ : estimator
        The classifier which predicts given the output of `estimators_`.

    stack_method_ : list of str
        The method used by each base estimator.

    See Also
    --------
    StackingRegressor : Stack of estimators with a final regressor.

    Notes
    -----
    When `predict_proba` is used by each estimator (i.e. most of the time for
    `stack_method='auto'` or specifically for `stack_method='predict_proba'`),
    The first column predicted by each estimator will be dropped in the case
    of a binary classification problem. Indeed, both feature will be perfectly
    collinear.

    References
    ----------
    .. [1] Wolpert, David H. "Stacked generalization." Neural networks 5.2
       (1992): 241-259.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.ensemble import StackingClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> estimators = [
    ...     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ...     ('svr', make_pipeline(StandardScaler(),
    ...                           LinearSVC(random_state=42)))
    ... ]
    >>> clf = StackingClassifier(
    ...     estimators=estimators, final_estimator=LogisticRegression()
    ... )
    >>> from sklearn.model_selection import train_test_split
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, stratify=y, random_state=42
    ... )
    >>> clf.fit(X_train, y_train).score(X_test, y_test)
    0.9...
    """

    def __init__(
        self,
        estimators,
        final_estimator=None,
        *,
        cv=None,
        stack_method="auto",
        n_jobs=None,
        passthrough=False,
        verbose=0,
    ):
        super().__init__(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            stack_method=stack_method,
            n_jobs=n_jobs,
            passthrough=passthrough,
            verbose=verbose,
        )

    def _validate_final_estimator(self):
        self._clone_final_estimator(default=LogisticRegression())
        if not is_classifier(self.final_estimator_):
            raise ValueError(
                "'final_estimator' parameter should be a classifier. Got {}".format(
                    self.final_estimator_
                )
            )

    def fit(self, X, y, sample_weight=None):
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

        Returns
        -------
        self : object
            Returns a fitted instance of estimator.
        """
        check_classification_targets(y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        return super().fit(X, self._le.transform(y), sample_weight)

    @if_delegate_has_method(delegate="final_estimator_")
    def predict(self, X, **predict_params):
        """Predict target for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        **predict_params : dict of str -> obj
            Parameters to the `predict` called by the `final_estimator`. Note
            that this may be used to return uncertainties from some estimators
            with `return_std` or `return_cov`. Be aware that it will only
            accounts for uncertainty in the final estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_output)
            Predicted targets.
        """
        y_pred = super().predict(X, **predict_params)
        return self._le.inverse_transform(y_pred)

    @if_delegate_has_method(delegate="final_estimator_")
    def predict_proba(self, X):
        """Predict class probabilities for `X` using the final estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_classes) or \
            list of ndarray of shape (n_output,)
            The class probabilities of the input samples.
        """
        check_is_fitted(self)
        return self.final_estimator_.predict_proba(self.transform(X))

    @if_delegate_has_method(delegate="final_estimator_")
    def decision_function(self, X):
        """Decision function for samples in `X` using the final estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        decisions : ndarray of shape (n_samples,), (n_samples, n_classes), \
            or (n_samples, n_classes * (n_classes-1) / 2)
            The decision function computed the final estimator.
        """
        check_is_fitted(self)
        return self.final_estimator_.decision_function(self.transform(X))

    def transform(self, X):
        """Return class labels or probabilities for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        y_preds : ndarray of shape (n_samples, n_estimators) or \
                (n_samples, n_classes * n_estimators)
            Prediction outputs for each estimator.
        """
        return self._transform(X)

    def _sk_visual_block_(self):
        # If final_estimator's default changes then this should be
        # updated.
        if self.final_estimator is None:
            final_estimator = LogisticRegression()
        else:
            final_estimator = self.final_estimator
        return super()._sk_visual_block_(final_estimator)

### End class StackingClassifier


class StackingRegressor(RegressorMixin, _BaseStacking):
    """Stack of estimators with a final regressor.

    Stacked generalization consists in stacking the output of individual
    estimator and use a regressor to compute the final prediction. Stacking
    allows to use the strength of each individual estimator by using their
    output as input of a final estimator.

    Note that `estimators_` are fitted on the full `X` while `final_estimator_`
    is trained using cross-validated predictions of the base estimators using
    `cross_val_predict`.

    Read more in the :ref:`User Guide <stacking>`.

    .. versionadded:: 0.22

    Parameters
    ----------
    estimators : list of (str, estimator)
        Base estimators which will be stacked together. Each element of the
        list is defined as a tuple of string (i.e. name) and an estimator
        instance. An estimator can be set to 'drop' using `set_params`.

    final_estimator : estimator, default=None
        A regressor which will be used to combine the base estimators.
        The default regressor is a :class:`~sklearn.linear_model.RidgeCV`.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy used in
        `cross_val_predict` to train `final_estimator`. Possible inputs for
        cv are:

        * None, to use the default 5-fold cross validation,
        * integer, to specify the number of folds in a (Stratified) KFold,
        * An object to be used as a cross-validation generator,
        * An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and y is
        either binary or multiclass,
        :class:`~sklearn.model_selection.StratifiedKFold` is used.
        In all other cases, :class:`~sklearn.model_selection.KFold` is used.
        These splitters are instantiated with `shuffle=False` so the splits
        will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. note::
           A larger number of split will provide no benefits if the number
           of training samples is large enough. Indeed, the training time
           will increase. ``cv`` is not used for model evaluation but for
           prediction.

    n_jobs : int, default=None
        The number of jobs to run in parallel for `fit` of all `estimators`.
        `None` means 1 unless in a `joblib.parallel_backend` context. -1 means
        using all processors. See Glossary for more details.

    passthrough : bool, default=False
        When False, only the predictions of estimators will be used as
        training data for `final_estimator`. When True, the
        `final_estimator` is trained on the predictions as well as the
        original training data.

    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    estimators_ : list of estimator
        The elements of the estimators parameter, having been fitted on the
        training data. If an estimator has been set to `'drop'`, it
        will not appear in `estimators_`.

    named_estimators_ : :class:`~sklearn.utils.Bunch`
        Attribute to access any fitted sub-estimators by name.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying regressor exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimators expose such an attribute when fit.
        .. versionadded:: 1.0

    final_estimator_ : estimator
        The regressor to stacked the base estimators fitted.

    stack_method_ : list of str
        The method used by each base estimator.

    See Also
    --------
    StackingClassifier : Stack of estimators with a final classifier.

    References
    ----------
    .. [1] Wolpert, David H. "Stacked generalization." Neural networks 5.2
       (1992): 241-259.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.linear_model import RidgeCV
    >>> from sklearn.svm import LinearSVR
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.ensemble import StackingRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> estimators = [
    ...     ('lr', RidgeCV()),
    ...     ('svr', LinearSVR(random_state=42))
    ... ]
    >>> reg = StackingRegressor(
    ...     estimators=estimators,
    ...     final_estimator=RandomForestRegressor(n_estimators=10,
    ...                                           random_state=42)
    ... )
    >>> from sklearn.model_selection import train_test_split
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=42
    ... )
    >>> reg.fit(X_train, y_train).score(X_test, y_test)
    0.3...
    """

    def __init__(
        self,
        estimators,
        final_estimator=None,
        *,
        cv=None,
        n_jobs=None,
        passthrough=False,
        verbose=0,
    ):
        super().__init__(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            stack_method="predict",
            n_jobs=n_jobs,
            passthrough=passthrough,
            verbose=verbose,
        )

    def _validate_final_estimator(self):
        self._clone_final_estimator(default=RidgeCV())
        if not is_regressor(self.final_estimator_):
            raise ValueError(
                "'final_estimator' parameter should be a regressor. Got {}".format(
                    self.final_estimator_
                )
            )

    def fit(self, X, y, sample_weight=None):
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        y = column_or_1d(y, warn=True)
        return super().fit(X, y, sample_weight)

    def transform(self, X):
        """Return the predictions for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        y_preds : ndarray of shape (n_samples, n_estimators)
            Prediction outputs for each estimator.
        """
        return self._transform(X)

    def _sk_visual_block_(self):
        # If final_estimator's default changes then this should be
        # updated.
        if self.final_estimator is None:
            final_estimator = RidgeCV()
        else:
            final_estimator = self.final_estimator
        return super()._sk_visual_block_(final_estimator)

###########################################################

# Simple (level-1) pre-trained model loader bypassing training loop
def get_pretrained_model(input_dir, base_learners, fold_number=0, **kargs):
    import utils_cf as uc 
    
    verbose = kargs.get('verbose', 1)
    final_estimator = kargs.get('final_estimator', LogisticRegression())
    
    clf = CFStacker(estimators=base_learners, 
                        final_estimator=final_estimator, 
                        work_dir = input_dir,
                        fold_number = fold_number, # use this to index traing and test data 
                        verbose=verbose) 
    meta_set = clf.cf_fetch()
    X_train, y_train = meta_set['train']['X'], meta_set['train']['y'] 
    X_test = meta_set['test']['X']
    y_test = None
    try: 
        y_test = meta_set['test']['y']
    except: 
        print("(get_pretrained_model) Warning: Test label (y_test) is not available.")
    R = X_train.T # transpose because we need users by items (or classifiers x data) for CF
    T = X_test.T
    U = meta_set['train']['U']
    L_train = y_train
    p_threshold = uc.estimateProbThresholds(R, L=L_train, pos_label=1, policy='fmax')
    lh = uc.estimateLabels(T, p_th=p_threshold) # We cannot use L_test (cheating), but we have to guesstimate
    L = np.hstack((L_train, lh)) # true labels (for R) concatenated with estimated labels (for T)
    X = np.hstack((R, T))
    n_train = R.shape[1]
    return (X, L, U, p_threshold, n_train)

def get_pretrained_model_with_confidence_matrix(input_dir, base_learners, fold_number=0, **kargs): 
    import utils_cf as uc

    ret = {}
    X, L, U, p_threshold, n_train = get_pretrained_model(input_dir, base_learners, fold_number=0, **kargs)
    ret['X'] = X 
    ret['L'] = L 
    ret['U'] = U
    ret['p_threshold'] = p_threshold
    ret['n_train'] = n_train
   
    # Key parameters
    policy_threshold= kargs.get('policy_threshold', 'fmax')
    conf_measure = kargs.get('conf_measure', 'brier')
    alpha = kargs.get('alpha', 100.0)
    beta = kargs.get('beta', 1.0)

    CX = uc.evalConfidenceMatrix(X, L=L, U=U, 
                             p_threshold=p_threshold, # not needed if L is given (suggested use: estimate L outside of this call)
                             policy_threshold=policy_threshold,
                             conf_measure=conf_measure, 
                             fill=0, is_cascade=True, n_train=n_train, 
                             fold=fold_number, 
                             verbose=0) 
    C0, Pc, p_threshold2, *CX_res = CX

    Cn = uc.mask_neutral_and_negative(C0, Pc, is_unweighted=False, weight_negative=0.0, sparsify=True)
    Cn = uc.balance_and_scale(Cn, X=X, L=L, Po=Pc, p_threshold=p_threshold, U=U, 
                        alpha=alpha, beta=beta, 
                            conf_measure=conf_measure, 
                                    n_train=n_train, fold=fold_number, verbose=0)
    assert np.allclose(p_threshold2, p_threshold)

    ret['C0'] = ret['confidence_matrix'] = C0
    ret['Pc'] = ret['color_matrix'] = Pc 
    ret['Cn'] = Cn  # masked confidence matrix where FP, FN (negative) and entries with high uncertainty (neutral) are set to 0s

    return ret

def verify_shape(X, R, T, L, U, p_threshold):
    if X.shape[1] != R.shape[1]+T.shape[1]: 
        raise ValueError(f"Total sample size: {X.shape[1]} != {R.shape[1]+T.shape[1]} = n(train):{R.shape[1]} + n(test):{T.shape[1]}")

    # if n_train != R.shape[1]: 
    #     raise ValueError(f"Size of training set {R.shape[1]} != n_train: {n_train}")

    if len(L) != X.shape[1]: 
        raise ValueError(f"Size of labels {len(L)} != sample size: {X.shape[1]} > Forgot to include estimated labels for T?")

    if len(U) != X.shape[0]: 
        raise ValueError(f"Inconsistent n(users/classifiers): {len(U)} != X.shape[0]: {X.shape[0]}")

    if len(p_threshold) != X.shape[0]:
        raise ValueError(f"Inconsitent n(users/classifiers): {X.shape[0]} != n(thresholds): {len(p_threshold)}") 
    return


# get the dataset
def get_dataset():
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
    return X, y

def test(): 
    from sklearn.datasets import load_iris

    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
    from sklearn.metrics._classification import cohen_kappa_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    #####################################################################################
    X, y = load_iris(return_X_y=True)
    print(f'> Number of classes: {len(np.unique(y))}')

    base_learners = [
                 ('RF', RandomForestClassifier(n_estimators= 200, 
                                                   oob_score = True, 
                                                   class_weight = "balanced", 
                                                   random_state = 20, 
                                                   ccp_alpha = 0.1)), 
                 ('KNNC', KNeighborsClassifier(n_neighbors = len(np.unique(y))
                                                     , weights = 'distance')),
                 ('SVC', SVC(kernel = 'linear'
                                   , class_weight = 'balanced'
                                  , break_ties = True)), 
                 ('GNB', GaussianNB()), 
                 ('QDA',  QuadraticDiscriminantAnalysis()), 
                 # ('MLPClassifier', MLPClassifier(alpha=1, max_iter=1000)), 
                 # ('DT', DecisionTreeClassifier(max_depth=5)),
                 # ('GPC', GaussianProcessClassifier(1.0 * RBF(1.0))),
                ]

    # Initialize Stacking Classifier with the Meta Learner
    # clf = StackingClassifier(estimators=base_learners, 
    #                          final_estimator=LogisticRegression())
    clf = CFStacker(estimators=base_learners, 
                             final_estimator=LogisticRegression(), verbose=1)

    # Extract score
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    clf.fit(X_train, y_train)

    print('[result]', clf.score(X_test, y_test))

    return

if __name__ == "__main__": 
    test()
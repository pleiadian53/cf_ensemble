#!/usr/bin/env python

"""

    Use 
    ---
    python combine.py /Users/chiup04/Documents/work/data/diabetes_cf


    datasink: A Pipeline for Large-Scale Heterogeneous Ensemble Learning
    Copyright (C) 2013 Sean Whalen

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see [http://www.gnu.org/licenses/].
"""

from glob import glob
import gzip
import os
from os.path import abspath, exists, isdir
from sys import argv

from common import load_properties
from pandas import concat, read_csv

path = abspath(argv[1])
assert exists(path)
p = load_properties(path)
fold_count = int(p['foldCount'])
nested_fold_count = int(p['nestedFoldCount'])
bag_count = max(1, int(p['bagCount']))

config_file = 'config.txt' 
dirnames = sorted(filter(isdir, glob('%s/weka.classifiers.*' % path)))

# validation-<outer fold> => level 1 training data
for fold in range(fold_count):
    dirname_dfs = []
    for dirname in dirnames:  # foreach classifier (each classifier has its down directory created by weka; weka.classifier....)
        classifier = dirname.split('.')[-1]
        nested_fold_dfs = []
        for nested_fold in range(nested_fold_count): 
            bag_dfs = []
            for bag in range(bag_count):
                filename = '%s/validation-%s-%02i-%02i.csv.gz' % (dirname, fold, nested_fold, bag)
                df = read_csv(filename, skiprows = 1, index_col = [0, 1], compression = 'gzip')
                df = df[['prediction']]
                df.rename(columns = {'prediction': '%s.%s' % (classifier, bag)}, inplace = True)
                bag_dfs.append(df)  # given a classifier, combine all its bagged versions
                print('... outer fold: {ocv} > inner fold: {icv}, bag: {bag} > size: {N}'.format(ocv=fold, icv=nested_fold, bag=bag, N=df.shape[0]))
            nested_fold_dfs.append(concat(bag_dfs, axis = 1))  # each bagged classifier is a column

            # [test]
            # bag_dfs_icv = concat(bag_dfs, axis = 1)
            # print('...... one partition of inner fold (cls={cls} combining {nb} bags) > {dim}'.format(cls=os.path.basename(dirname), nb=len(bag_dfs), dim=bag_dfs_icv.shape))
        ### combined all the inner-training splits (across all inner folds)

        # concat(nested_fold_dfs) ~> bagged classifier X (X.0, X.1, ... X.10)
        dirname_dfs.append(concat(nested_fold_dfs, axis = 0))  

        # [test]
        one_cls_inner = concat(nested_fold_dfs, axis = 0)
        print('......... all inner folds combined (cls={cls}) > {dim}'.format(cls=os.path.basename(dirname), dim=one_cls_inner.shape))
    
    df_path = '%s/validation-%s.csv.gz' % (path, fold)
    concat(dirname_dfs, axis = 1).sort_index().to_csv(df_path, compression='gzip')  # combine all classifiers

    # [test]
    all_cls_inner = concat(dirname_dfs, axis = 1)
    print('............ all inner folds combined (cls={cls}) > {dim} > complete fold {ocv}'.format(cls=os.path.basename(dirname), dim=all_cls_inner.shape, ocv=fold))    

    # with gzip.open('%s/validation-%s.csv.gz' % (path, fold), 'wb') as f:
    #     # concat(dirname_dfs, axis = 1).sort().to_csv(f)
    #     concat(dirname_dfs, axis = 1).sort_index().to_csv(f)  # [log] In Python3.7> memoryview: a bytes-like object is required, not 'str'

# prediction-<outer fold> => level-1 test data
for fold in range(fold_count):
    dirname_dfs = []
    for dirname in dirnames:
        classifier = dirname.split('.')[-1]
        bag_dfs = []
        for bag in range(bag_count):
            filename = '%s/predictions-%s-%02i.csv.gz' % (dirname, fold, bag)
            df = read_csv(filename, skiprows = 1, index_col = [0, 1], compression = 'gzip')
            df = df[['prediction']]
            df.rename(columns = {'prediction': '%s.%s' % (classifier, bag)}, inplace = True)
            bag_dfs.append(df)
        dirname_dfs.append(concat(bag_dfs, axis = 1))

    df_path = '%s/predictions-%s.csv.gz' % (path, fold)
    concat(dirname_dfs, axis = 1).sort_index().to_csv(df_path, compression='gzip')
    
    # with gzip.open('%s/predictions-%s.csv.gz' % (path, fold), 'wb') as f:
    #     # concat(dirname_dfs, axis = 1).sort().to_csv(f)
    #     concat(dirname_dfs, axis = 1).sort_index().to_csv(f)



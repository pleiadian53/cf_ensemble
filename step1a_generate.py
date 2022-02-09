#!/usr/bin/env python

"""

    Modified generate module from datasink. 

    Use
    ---
        python step1a_generate.py <project_path> 
           e.g. /Users/chiup04/Documents/work/data/pf2

    Reference
    ---------
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

from itertools import product
from os import environ, system
from os.path import abspath, dirname, exists
from sys import argv
import sys, os

from utilities import load_arff_headers, load_properties, cluster_cmd, get_num_cores

# datasink module
# from common import load_arff_headers, load_properties

from sklearn.externals.joblib import Parallel, delayed

import utils_sys
from utils_sys import div

# used for nested CV setting
def generate_nested_cv(parameters):
    working_dir, project_path, classifier, fold, bag = parameters

    # [note]
    # 1. predictions corresponds to the test set in flat CV 
    # 2. nested_fold_values is used here
    expected_filenames = ['%s/%s/predictions-%s-%02i.csv.gz' % (project_path, classifier.split()[0], fold, bag)] + ['%s/%s/validation-%s-%02i-%02i.csv.gz' % (project_path, classifier.split()[0], fold, nested_fold, bag) for nested_fold in nested_fold_values]
    if sum(map(exists, expected_filenames)) == len(expected_filenames):
        return

    # Pipeline.groovy followed by project dir, fold, bag, classifier params
    cmd = 'groovy -cp %s %s/Pipeline.groovy %s %s %s %s' % (classpath, working_dir, project_path, fold, bag, classifier)
    if use_cluster:
        # cmd = '%s \"%s\"' % (cluster_cmd, cmd)
        cmd = 'python %s \"%s\"' % (cluster_cmd(), cmd)

    try: 
        system(cmd)
    except Exception as e: 
        raise RuntimeError(e)

    return

# used for 'flat' CV setting
def generate(parameters):
    """

    Input
    -----
    seed: an index used to distinguish between different experiments; used only in the flat CV setting

    """
    working_dir, project_path, classifier, bag, seed, fold = parameters
    classifier_name = classifier.split()[0]

    classifier_dir = working_dir # by default, put classifier config file in the same directory that hosts the input data
    expected_filenames = ['%s/%s/valid-b%s-f%s-s%s.csv.gz' % (classifier_dir, classifier_name, bag, fold, seed)] + ['%s/%s/test-b%s-f%s-s%s.csv.gz' % (classifier_dir, classifier_name, bag, fold, seed)]
    if sum(map(exists, expected_filenames)) == len(expected_filenames):
        return

    cmd = 'groovy -cp %s %s/generate.groovy %s %s %s %s %s' % (classpath, working_dir, project_path, bag, seed, fold, classifier)
    if use_cluster:
        # run rc.py --cores 1 --walltime 00:10 --queue low ... 
        cmd = 'python %s \"%s\"' % (cluster_cmd(), cmd)
    system(cmd)


# ensure project directory exists
code_dir = working_dir = dirname(abspath(argv[0]))
domain = 'pf2'
try: 
    project_path = abspath(argv[1])
except: 
    src_path = utils_sys.getProjectPath(domain=domain)
    msg = 'Hint: python step1a_generate.py %s (assuming domain=%s)  ...' % (src_path, domain)
    raise RuntimeError(msg)
assert exists(project_path)

# load and parse project properties
p = load_properties(project_path)  # config.txt is similar to datasink's weka.properties
classifier_dir  = p.get('classifierDir', project_path)  # lens parameter
classifiers_fn  = '%s/%s' % (classifier_dir, p['classifiersFilename'])  # classifier spec file (e.g. classifiers.txt)
input_fn    = '%s/%s' % (classifier_dir, p['inputFilename'])
assert exists(input_fn)
# print('(step1a_generate) classifier_dir:%s\n... classifiers_fn:%s\n' % (classifier_dir, classifiers_fn))

use_cluster     = True if str(p['useCluster']).lower() in ['y', 'yes', 'true', '1' ] else False
n_jobs = 1 if use_cluster else get_num_cores()

# generate cross validation values for leave-one-value-out or k-fold
assert ('foldAttribute' in p) or ('foldCount' in p)
if 'foldAttribute' in p:
    headers = load_arff_headers(input_fn)
    fold_values = headers[p['foldAttribute']]
else:
    fold_values = range(int(p['foldCount']))

# repetitions of the experiments (in terms of seeds used for randomizing the data)
# [note] this is used only when nested CV is not enabled (see step1_generate.py)
seed_count = int(p['seeds']) if 'seeds' in p else 1
seeds      = range(seed_count) if seed_count > 1 else [0]  

# only used in nested CV
nested_fold_values = range(int(p['nestedFoldCount'])) if 'nestedFoldCount' in p else range(5)

bag_count = int(p['bagCount']) if 'bagCount' in p else int(p['bags']) 
bags = bag_values = range(bag_count) if bag_count > 1 else [0]

# ensure java's classpath is set
try: 
    classpath = environ['CLASSPATH']
except: 
    print('(step1a_generate) source set_env.sh to set CLASSPATH ...')
    raise ValueError( "CLASSPATH has not been defined (e.g. export CLASSPATH=/Users/<username>/java/weka.jar)." )

# datasink old code
# command for cluster execution if enabled
# use_cluster = False if 'useCluster' not in p else p['useCluster'] == 'true'
# cluster_cmd = 'rc.py --cores 1 --walltime 06:00:00 --queue small --allocation acc_9'

# load commandline instructions for each classifier specified in the classifier spec (classifiers.txt), skip commented lines
classifier_specs = filter(lambda x: not x.startswith('#'), open(classifiers_fn).readlines())

# ... further processing 
# a. remove trailing comments
# b. filter blank lines
classifiers = []
for spec in classifier_specs: 
    spec = spec.strip()
    if spec: 
        # any commands in it? 
        delimit = spec.find("#")
        if delimit > 0: 
            spec = spec[:delimit].strip() 
        classifiers.append(spec)
# classifiers = [_.strip() for _ in classifiers]

### verify classifier specs
print('... found {0} classifier instructions:'.format(len(classifiers)))
for c in classifiers: 
    print('...... {spec}'.format(spec=c))
div(message='Classpath=%s' % classpath, symbol='*', border=2)
print('... classifier dir:\n%s\n' % classifier_dir) 
print('... classifiers (n=%d, n_bags:%d, cv_outer:%d, cv_inner:%d, n_seeds:%d):\n%s\n' % \
    (len(classifiers), len(bags), len(nested_fold_values), len(fold_values), len(seeds), classifiers))

############################################################

def run(nested_cv=True, dry_run=False): 

    all_parameters = list(product([working_dir], [project_path], classifiers, fold_values, bag_values))
    generate_fn = generate_nested_cv if nested_cv else generate
    assert hasattr(generate_fn, '__call__')

    # [test]
    if dry_run: 
        for i, params in enumerate(all_parameters): 
            print('[%d] %s' % (i+1, params))
        sys.exit(0)

    Parallel(n_jobs = n_jobs, verbose = 50)(delayed(generate_fn)(parameters) for parameters in all_parameters)

    print("... step1 completed > BPs generated.")

    return

if __name__ == "__main__": 
    run(nested_cv=True) # dry_run=True





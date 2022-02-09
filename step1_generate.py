#!/usr/bin/env python
"""
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
from random import randrange
from utilities import load_arff_headers, load_properties, cluster_cmd, get_num_cores
from sklearn.externals.joblib import Parallel, delayed
from utils_sys import div

def generate(parameters):
    code_dir, project_path, classifier, bag, seed, fold = parameters
    classifier_name = classifier.split()[0]
    expected_filenames = ['%s/%s/valid-b%s-f%s-s%s.csv.gz' % (classifier_dir, classifier_name, bag, fold, seed)] + ['%s/%s/test-b%s-f%s-s%s.csv.gz' % (classifier_dir, classifier_name, bag, fold, seed)]
    if sum(map(exists, expected_filenames)) == len(expected_filenames):
        return

    # print('(generate) classpath: %s' % classpath)
    # print('(generate) classifier: %s' % classifier_name)

    cmd = 'groovy -cp %s %s/generate.groovy %s %s %s %s %s' % (classpath, code_dir, project_path, bag, seed, fold, classifier)
    if use_cluster:
        # run rc.py --cores 1 --walltime 00:10 --queue low ... 
        cmd = 'python %s \"%s\"' % (cluster_cmd(), cmd)
    system(cmd)


print "\nStarting . . ."

# ensure project directory exists
project_path = abspath(argv[1])
code_dir     = dirname(abspath(argv[0]))
assert exists(project_path)

# load and parse project properties
p = load_properties(project_path)
classifier_dir 	= p['classifierDir']
classifiers_fn 	= '%s' % (p['classifiersFilename'])
input_fn 	= '%s' % (p['inputFilename'])
use_cluster     = True if p['useCluster'] in ['Y', 'y', 'yes', 'true', 'True'] else False
assert exists(input_fn)

# generate cross validation values for leave-one-value-out or k-fold
assert ('foldAttribute' in p) or ('foldCount' in p)
if 'foldAttribute' in p:
    headers = load_arff_headers(input_fn)
    fold_values = headers[p['foldAttribute']]
else:
    fold_values = range(int(p['foldCount']))

# repetitions of the experiments (in terms of seeds used for randomizing the data)
seed_count = int(p['seeds'])
seeds 	   = range(seed_count) if seed_count > 1 else [0]

# bags of experiments (in terms of resampled training data to generate different versions of the same algorithm)
bag_count = int(p['bags'])
bags      = range(bag_count) if bag_count > 1 else [0]

# ensure java's classpath is set
classpath = environ['CLASSPATH']

# load classifiers from file, skip commented lines
classifiers = filter(lambda x: not x.startswith('#'), open(classifiers_fn).readlines())
classifiers = [_.strip() for _ in classifiers]

div(message='Classpath=%s' % classpath, symbol='*', border=2)
print('... classifier dir:\n%s\n' % classifier_dir) 
print('... classifiers:\n%s\n' % classifiers)
print('... bags:\n%s\n' % bags)

all_parameters = list(product([code_dir], [project_path], classifiers, bags, seeds, fold_values))
n_jobs = 1 if use_cluster else get_num_cores()
Parallel(n_jobs = n_jobs, verbose = 50)(delayed(generate)(parameters) for parameters in all_parameters)

print "Done!\n"



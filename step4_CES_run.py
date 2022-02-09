from itertools import product
from os import system, makedirs
from os.path import abspath, dirname, exists, isdir
from sys import argv
from glob import glob
from utilities import load_properties, cluster_cmd, get_num_cores
from sklearn.externals.joblib import Parallel, delayed

def CES_run(parameters):
    code_dir, project_path, size, seed, fold, RULE, start, metric = parameters
    cmd = 'python %s/CES_run.py %s %s %s %s %s %s %s' % (code_dir, project_path, size, fold, seed, RULE, start, metric)
    if use_cluster:
        cmd = 'python %s \"%s\"' % (cluster_cmd(), cmd)
    print cmd
    system(cmd)


print "\nStarting . . ."

# ensure project directory exists
project_path = abspath(argv[1])
assert exists(project_path)
code_dir     = dirname(abspath(argv[0]))
directory    = 'CES_OUTPUT'
subdirectory = 'ORDER'
dirnames     = sorted(filter(isdir, glob('%s/weka.classifiers.*' % project_path)))


# load and parse project properties
p            = load_properties(project_path)
fold_count   = int(p['foldCount'])
seeds        = int(p['seeds'])
metric       = p['metric']
RULE         = p['RULE']
use_cluster  = True if p['useCluster'] in ['Y', 'y', 'yes', 'true', 'True'] else False
start_state  = '1'  #initialize ensemble with top model
max_num_clsf = len(dirnames) * seeds. # num_classifiers * num_experiments
sizes        = range(1,max_num_clsf+1)


if not exists('%s/%s/' % (project_path, directory)):
    makedirs('%s/%s/' % (project_path, directory))

for o in range(seeds):
    if not exists("%s/%s/%s%i" % (project_path, directory, subdirectory, o)):
        makedirs("%s/%s/%s%i" % (project_path, directory, subdirectory, o))

all_parameters = list(product([code_dir], [project_path], sizes, range(seeds), range(fold_count), [RULE], [start_state], [metric]))
Parallel(n_jobs = get_num_cores(), verbose = 50)(delayed(CES_run)(parameters) for parameters in all_parameters)

print "Done!\n"




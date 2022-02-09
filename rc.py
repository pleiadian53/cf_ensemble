from optparse import OptionParser
import os, subprocess
from os import getcwd, system, mkdir
from os.path import abspath, exists
from time import sleep
import time, random, math
import glob, re
import numpy as np

# highlight
div = None
try: 
    from utils_sys import div
except:
    div = print

def generate_id(): 
    import numpy as np
    return math.ceil(np.random.random() * random.choice(range(20000, 500000)))

def interpret(cmd, verbose=True):
    """

    Memo
    ----
    1. Final bsub command: 

    work_dir <- /Users/pleiades/Documents/work/data/pf1

    echo "python /Users/pleiades/Documents/work/cluster_ensemble/cf.py /Users/pleiades/Documents/work/data/pf1" 
        | bsub -o /Users/pleiades/Documents/work/data/pf1/log/cf_ensemble_pf1-1552441879.497678.out.txt 
               -e /Users/pleiades/Documents/work/data/pf1/log/cf_ensemble_pf1-1552441879.497678.err.txt 
               -q expressalloc -n 4 -W 06:10 -J cf_ensemble_pf1 -P acc_pandeg01a -m manda -M 8000 -R "rusage[mem=4000]"

    """
    ret = {}
    components = cmd.split()
    # e.g. python /Users/tauceti/Documents/work/cluster_ensemble/cf.py /Users/tauceti/Documents/work/data/pf1 -s 5
    ret['compiler'] = components[0]
    ret['src_path'] = components[1]
    ret['project_path'] = components[2]
    ret['options'] = ' '.join(components[3:])

    ret['domain'] = os.path.basename(ret['project_path'])

    if verbose: 
        exec_cmd = ret['src_path']+ ' ' +ret['options'] if ret['options'] else ret['src_path']
        print('(rc) executable: {0}\n... domain:{1}\n... project_path: {2}\n'.format(exec_cmd, ret['domain'], ret['project_path']))
    return ret

def parse_output(s, save=True, file_name='job_submitted.log', prefix=None):
    # example: 'Job <121519944> is submitted to queue <premium>.\n'
    ret = {}
    p = re.compile(r'[j|J]ob \<(?P<job_id>\d+)\> is submitted to queue \<(?P<queue>\w+)\>.*')
    m = p.match(s)
    if m: 
        ret['job_id'] = m.group('job_id')
        ret['queue'] = m.group('queue')

        ### global vars ### 
        ret['job_name'] = job_name

        pjob = re.compile(r'(?P<job_stem>[_a-zA-Z0-9]+\-[_a-zA-Z0-9]+.*)\.out\.txt')
        mjob = pjob.match(new_job)
        if mjob: 
            ret['job_full_name'] = mjob.group('job_stem')  # includes .out.txt

        ret['output'] = stdout_fn 
        ret['error'] = stderr_fn
    else: 
        print('(parse_output) pattern {0} did not capture the output:\n{1}\n'.format(p.pattern, s))

    ### parse the main process command: cmd
    app_name = os.path.basename(components['src_path'])
    options = components['options']
    if options: app_name = '%s %s' % (app_name, options)

    domain = os.path.basename(components['project_path'])

    if save and len(ret) > 0: 
        # >>> time format structured to facilitate sorting
        timestamp2 = time.strftime("%y/%m/%d %H:%M:%S", time.localtime(time.time()))
        line = 'ID: {id} | name: {name} | exec: {app} | domain: {domain} | queue: {queue} | time: {time}\n'.format(id=ret['job_id'], name=ret['job_name'], app=app_name, domain=domain, queue=ret['queue'], time=timestamp2)
        line += '... out:{o}\n... err:{eo}\n'.format(o=ret['output'], eo=ret['error'])
        
        if prefix is None: prefix = path_logging  # global
        fpath = os.path.join(prefix, file_name)

        print('... Saving job descriptor <<%s>> to:\n%s\n' % (line, fpath))
        with open(fpath, "a+") as f: 
            f.write(line)

    return ret

timestamp = now = time.strftime("%y%m%d-%H%M%S", time.localtime(time.time())) # time()
parser = OptionParser()
parser.add_option('-c', '--cores', dest = 'cores', type = 'int')
parser.add_option('-w', '--walltime', dest = 'walltime')
parser.add_option('-a', '--allocation', dest = 'allocation', default = 'TO_BE_ASSIGNED_BY_USER')
parser.add_option('-q', '--queue', dest = 'queue', default = 'expressalloc')
parser.add_option('-n', '--name', dest = 'name', default = timestamp)
parser.add_option('--memory', dest='memory', type='int', default=32768)
parser.add_option('--interactive', action="store_true", dest="interactive", default=False)
parser.add_option('--workdir', dest='workdir', default=getcwd())
parser.add_option('--dryrun', action="store_true", dest="dryrun", default=False)

(options, args) = parser.parse_args()
cores = options.cores
walltime = options.walltime
queue = options.queue
job_name = options.name
allocation = options.allocation
dry_run = options.dryrun
interactive = options.interactive
memory = options.memory
working_dir = options.workdir
### Main Process

main_process = cmd = ' '.join(args)
div('(rc) dry run? {0} | main process: {1}'.format(dry_run, cmd))  # e.g. python /Users/tauceti/Documents/work/cluster_ensemble/cf.py /Users/tauceti/Documents/work/data/pf1
print('... memory: {0}, interactive? {1}'.format(memory, interactive))
print('... jobname: {0}, timestamp: {1}'.format(job_name, timestamp))
print('... job output dir (workdir): {0}\n\n'.format(working_dir))
### interpret the project path and domain 
components = interpret(cmd)

###################################

# make sure the logs subdirectory exists

# working directory is specified by the options above instead
# working_dir = components['project_path'] # getcwd() # project_path # getcwd() 

log_dir = 'log'
if not exists('%s/%s' % (working_dir, log_dir)):
    mkdir('%s/%s' % (working_dir, log_dir))
path_logging = abspath('%s/%s' % (working_dir, log_dir))

# set the log filenames
prefix = "{prefix}/{log}/{job}*.out.txt".format(prefix=abspath(working_dir), log=log_dir, job=job_name)
stdout_fn = abspath('{prefix}/{log}/{job}-{meta}.out.txt'.format(prefix=working_dir, log=log_dir, job=job_name, meta=timestamp)) # meta=%J
stderr_fn = abspath('{prefix}/{log}/{job}-{meta}.err.txt'.format(prefix=working_dir, log=log_dir, job=job_name, meta=timestamp))

# normalize stdout/stderr file names
# print(stdout_fn)
new_job = os.path.basename(stdout_fn)
existing_jobs = sorted([os.path.basename(name) for name in glob.glob(prefix)])
print('... job log: {0}'.format(path_logging))
print('... new job: {0}\n... existing jobs:\n{1}\n'.format(new_job, ['%s' % job for job in existing_jobs]))
# new_job_id = sorted(existing_jobs + [new_job, ]).index(new_job)
# stdout_fn = abspath('{prefix}/{log}/{job}-{meta}.out.txt'.format(prefix=working_dir, log=log_dir, job=job_name, meta=new_job_id)) # meta=timestamp
# stderr_fn = abspath('{prefix}/{log}/{job}-{meta}.err.txt'.format(prefix=working_dir, log=log_dir, job=job_name, meta=new_job_id))

### build the final command and run

# optional: -m manda -M 8000
# qsub_cmd = 'echo \"%s\" | bsub -o %s -eo %s -q %s -n %i -W %s -J %s -P %s -R \"rusage[mem=%s]\"' % (cmd, stdout_fn, stderr_fn, queue, cores, walltime, job_name, allocation, memory)


# [note] memory is defined wrt job slot
#        
qsub_cmd = 'echo \"{cmd}\" | bsub -o {out} -eo {err} -q {q} -n {n_slots} -W {W} -J {J} -P {P} -R \"rusage[mem={M}] affinity[core({n_cores})]\"'.format(cmd=cmd, 
    out=stdout_fn, err=stderr_fn, q=queue, n_slots=1, W=walltime, J=job_name, P=allocation, M=memory, n_cores=cores)

if interactive: qsub_cmd = '{cmd} -Ip /bin/bash'.format(cmd=qsub_cmd)
if queue.startswith('loc'):
    div("(rc) Final submission command:\n%s\n ... (verify) #" % cmd, symbol='#', border=2)
else: 
    div("(rc) Final submission command:\n%s\n ... (verify) #" % qsub_cmd, symbol='#', border=2)

### set enviornment variables 
# export CLASSPATH=/hpc/users/chiup04/work/java/weka.jar
# export JAVA_OPTS="-Xmx8g"

# e.g. 
# python rc.py --cores 1 --walltime 00:10 --queue expressalloc --allocation acc_pandeg01a "python /Users/<user/Documents/work/cluster_ensemble/cf.py /Users/pleiades/Documents/work/data/pf1"
# 1. "--cores 1 --walltime 00:10 --queue expressalloc --allocation acc_pandeg01a" will be 'consumed' by the args parser in rc.py

if dry_run: 
    # do nothing, display the final bsub command above 
    result = "Job <{id}> is submitted to queue <{queue}>.".format(id=generate_id(), queue=random.choice(['blackhole', 'quasar', 'zetareticuli']))
    print('\n(rc) output:\n%s\n' % result)
    parse_output(result, save=True, file_name='job_submitted.log', prefix=None)
else:
    ### method A. Send a customized job script

    ### method B. send the command
    # system(qsub_cmd)
    if queue.startswith('loc'):  # local: use local host
        result = subprocess.check_output(cmd, shell=True)
    else: 
        result = subprocess.check_output(qsub_cmd, shell=True)
    result = result.decode("utf-8")
    print('\n(rc) output:\n%s\n' % result)
    parse_output(result, save=True, file_name='job_submitted.log', prefix=None)

    # result = subprocess.run(qsub_cmd, stdout=subprocess.PIPE) # result is only a bytes object
    # print('\n(rc) output:\n%s\n' % result.stdout.decode('utf-8'))
    # parse_output(result.stdout.decode('utf-8'), save=True, file_name='job_submitted.log', prefix=None)

    sleep(np.random.uniform(40, 50, 1)[0])

# 10 sec of delay 
# polite delay in case called via external loop

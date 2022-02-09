from itertools import product
import os, re, random, time
from os import environ, system, makedirs
from os.path import abspath, dirname, exists, isdir
from sys import argv
import glob
from utilities import load_properties, get_num_cores  # cluster_cmd
from sklearn.externals.joblib import Parallel, delayed

# cluster_ensemble dependent modules
from utils_sys import div

# moniter execution time
import timing

# import rl

### system global variables
# job_spec
class System(object): 
    dry_run = False
    classifier_dirnames = ''
    code_dir = path_exec = os.getcwd()
    compiler = 'python'

    parse_mode = True
    filter_mode = False
    kill_pending = False

# refactor: utils_cluster
class Job(object): 
    app = exe = ''
    args = ''
    options = ''  # whole command line options, some of which are not specifically for job submission command
    options_exe = ''  # command line options specifically for the execuable (e.g. the '-s 2' in 'cf.py -s 2' where exe is 'cf.py')

    chain = [] # [app, ]  # job chain: a sequence of related jobs (but without dependency)
    meta = ''  # internal job descriptor (e.g. those that helps to differentiate the same type of job on different settings)

    # I/O
    log_dir = 'log'
    prefix = project_path = os.getcwd()  # root directory of the job/project
    path_log = os.path.join(prefix, log_dir)  # path to the job files
    domain = ''  # derived from project_path, os.path.basename(project_path)
    output_pattern = ''

    # Job request: cluster attributes
    user = ''
    n_cores = cores = 1
    walltime = '01:00'
    queue = 'expressalloc'
    name = 'test'    # related: keyword
    allocation = 'guest'
    interactive = False
    memory = 8096

    ### Job properties (that depend on external factors such as the execuable, HTC environment, etc)
    compiler = 'python'
    time_format = '%y%m%d-%H%M%S'
    timestamp = ''
    keyword = 'generate'  # keyword in the jobname

    # Job states
    p_submitted = re.compile(r"^.*Job (?P<id>\d+): <(?P<name>.*)> in cluster <(?P<cluster>\w+)>\s+(?P<status>\w+)")

    # job submitted with errors: e.g. Subject: Job 122894888: <cf-s1-remanei> in cluster <minerva> Exited
    p_exited = re.compile(r"^.*Job (?P<id>\d+): <(?P<name>[_a-zA-Z0-9]+(\-[_a-zA-Z0-9]+)?)> in cluster <(?P<cluster>\w+)> Exited")
    p_done = re.compile(r"^.*Job (?P<id>\d+): <(?P<name>[_a-zA-Z0-9]+(\-[_a-zA-Z0-9]+)?)> in cluster <(?P<cluster>\w+)> Done")

    p_err_end_msg = re.compile(r"^.*\<(?P<path>.*)\> for stderr output of this job")
    p_exit = re.compile(r"^.*exit code (?P<code>\d+)")  # e.g. Exited with exit code 140.

    p_exception = re.compile(r"(?P<error_type>\w+Error):\s+(?P<message>.*)")

    job_success = "Successfully completed"
    end_job = "Completed"

    # Example incomplete status 
    # TERM_RUNLIMIT: job killed after reaching LSF run time limit.
    # Exited with exit code 140.

    @staticmethod
    def config(prefix):   
        import getpass, os  

        diagnoses = "\n1. specified --domain instead of --domains for multiple domains/datasets?\n"
        diagnoses += "2. dataset does not yet exist on this host?\n"
        assert os.path.exists(prefix), "Invalid prefix: {p}\n... Diagnoses:\n{d}\n".format(p=prefix, d=diagnoses)

        ### attributes configured
        # prefix/project_path
        # path_log
        # domain 
        # user 
        # output_pattern

        ### configure logging directory
        Job.prefix = Job.project_path = prefix # e.g. $HOME/work/data/pf1 => domain: pf1

        Job.path_log = os.path.join(prefix, Job.log_dir)
        assert os.path.exists(Job.prefix), "Invalid prefix/base directory:\n%s\n" % Job.prefix

        ### configure other job parameters that depend on prefix 
        Job.domain = os.path.basename(Job.project_path)
        
        if not os.path.exists(Job.path_log):
            print("(config) Creating log directory for domain: '%s':\n%s\n" % (Job.domain, Job.path_log))
            os.mkdir(Job.path_log) 

        ### other job paramters 
        Job.user = getpass.getuser() # 'pleiades'

        Job.output_pattern = Job.job_file_pattern()

        # configured: Job.log_path, Job.domain, Job.user, Job.output_pattern
        return  

    @staticmethod
    def job_file_pattern(prefix='', keyword=''):
        # keyword: keyword in the job name (e.g. 'cf')
        #    e.g. step1a_generate-pacificus-190320-183002.out.txt
        
        if not prefix: prefix = Job.path_log
        pattern = os.path.join(prefix, '*.out.txt')
        if keyword:   
            pattern = os.path.join(prefix, "*{keyword}*.out.txt".format(keyword=keyword))
        return pattern

    @staticmethod
    def parse(exec_cmd):   # the main executable (e.g. cf.py -s 2)
        Job.exe, Job.options_exe = ' '.join(exec_cmd.split()[:1]), ' '.join(exec_cmd.split()[1:])

    @staticmethod
    def name_job(app='', opt='', domain='', job_id=''): 
        if not domain: domain = Job.domain 
        if not app: app = Job.app  # see CF_run()

        # opt: options for the main execuable
        if not opt: opt = Job.options  # see CF_run()
        if not job_id: job_id = Job.meta

        assert app.find('.') > 0, "Invalid executable? %s" % app
        app = app.split('.')[0]
        if opt: app += ''.join(opt.split())

        job_name = '{exec}-{domain}'.format(exec=app, domain=domain) 
        if job_id: job_name = '{base}-{id}'.format(base=job_name, id=job_id)
        return job_name

    @staticmethod
    def parse_args(time_format=''): 
        import time, os
        from optparse import OptionParser

        if time_format: Job.time_format = time_format
        Job.timestamp = now = time.strftime(Job.time_format, time.localtime(time.time())) # time()
        
        parser = OptionParser()
        parser.add_option('-c', '--cores', dest = 'cores', type = 'int')
        parser.add_option('-w', '--walltime', dest = 'walltime')
        parser.add_option('-a', '--allocation', dest = 'allocation', default = 'andromeda')
        parser.add_option('-q', '--queue', dest = 'queue', default = 'expressalloc')
        parser.add_option('-n', '--name', dest = 'name', default = timestamp)
        parser.add_option('-k', '--keyword', dest = 'keyword', default = 'cf')
        parser.add_option('--memory', dest='memory', type='int', default=32768)
        parser.add_option('--interactive', action="store_true", dest="interactive", default=False)
        parser.add_option('--workdir', dest='workdir', default=os.getcwd())
        # parser.add_option('--dryrun', action="store_true", dest="dryrun", default=False)

        ### use command line options to configure the job spec and system variables
        (options, args) = parser.parse_args()
        Job.cores = options.cores
        Job.walltime = options.walltime
        Job.queue = options.queue
        Job.name = options.name
        Job.allocation = options.allocation
        Job.interactive = options.interactive
        Job.memory = options.memory
        Job.working_dir = options.workdir
        Job.keyword = options.keyword
        # System.dry_run = options.dryrun

        return options, args

    @staticmethod
    def isIncomplete(fpath, descriptors=[], level=0, reset=False):
        tError = False 
        
        # status_pattern = "Read file </hpc/users/chiup04/work/data/remanei/log/cf-s1-remanei-190320-144121.err.txt> for stderr output of this job."
        #p_err_end_msg = re.compile(r"^.*\<(?P<path>.*)\> for stderr output of this job")
        #p_exit = re.compile(r"^.*exit code (?P<code>\d+)")  # e.g. Exited with exit code 140.

        # level: 0
        with open (fpath, 'rt') as fp:  # Open file for reading of text data. 't': open in text mode (default)
            if reset: fp.seek(0)
            for line in fp: 
                line = line.strip()
                m_exit = Job.p_exit.match(line)
                if m_exit: 
                    tError = True
                    print('(isIncomplete) Found exit code: %s\n' % m_exit.group('code'))
                    break
        
        if tError and level >= 1: 
            for line in readlines_reverse(fpath): 
                line = line.strip()
                m = Job.p_err_end_msg.match(line)
                if m: 
                    print('... error file: %s' % os.path.basename(m.group('path')))
            
        return tError

    @staticmethod
    def parse_output(fpath, level=0, reset=True, verbose=1):
        tError = False
        ret = {}

        p_exit, p_err_end_msg = Job.p_exit, Job.p_err_end_msg
        n_nonempty = n_submit = 0 
        with open (fpath, 'rt') as fp:  # Open file for reading of text data. 't': open in text mode (default)
            if reset: fp.seek(0)

            ret['output'] = os.path.basename(fpath)
            ret['submitted'] = True

            for lnum, line in enumerate(fp): 
                line = line.strip()
                if len(line) > 0: n_nonempty +=1
                if verbose and n_nonempty == 1: print('(parse_output) First nonempty line:\n%s\n' % line)
                ##########################################

                # general query 1: job submission
                m = Job.p_submitted.match(line)
                if m: 
                    ret['id'], ret['name'], ret['queue'] = m.group('id'), m.group('name'), m.group('cluster')
                    n_submit += 1
                else: 
                    ret['submitted'] = False

                # general query 2: executable command line
                if Job.isExecPath(line): 
                    cmd = line.strip()
                    if verbose: print("(parse) Cmd? %s" % cmd)
                    # e.g. python /sc/orga/work/chiup04/cluster_ensemble/cf.py /hpc/users/chiup04/work/data/pf1 -s 5
                    exec_components = interpret(line, verbose=True) # 'compiler', 'src_path', 'project_path', 'options'
                    ret['cmd'] = exec_components['src_path'] + ' ' + exec_components['options']

                ##########################################

                # error query 1: job exit code
                m_exit = p_exit.match(line)
                if m_exit: 
                    tError = True
                    ret['code'] = m_exit.group('code')
                    # print('... Found error code in line {n}'.format(n=lnum+1))
                    break

        # assert n_submit > 0, "Did not find the job submission line!"
        if n_submit == 0 and verbose: 
            print('(***) did not find the job submission line > ID: {id}, file: {o}'.format(id=ret.get('id', '?'), o=ret['output']))
                    
        
        if tError and level > 0: 
            for line in readlines_reverse(fpath): 
                line = line.strip()
                m = p_err_end_msg.match(line)
                if m: 
                    ret['path'] = os.path.basename(m.group('path'))
        return ret  # keys: output, submitted, id, name, queue, cmd, code, path

    @staticmethod
    def parse_error(fpath, level=0, reset=True, verbose=1): 
        # e.g. OSError: [Errno 122] Disk quota exceeded
        ret = {}
        n_nonempty = 0
        with open (fpath, 'rt') as fp:  # Open file for reading of text data. 't': open in text mode (default)
            if reset: fp.seek(0)

            ret['output'] = os.path.basename(fpath)
            for lnum, line in enumerate(fp): 
                line = line.strip()
                if len(line) > 0: n_nonempty +=1
                if verbose and n_nonempty == 1: print('(parse_error) First nonempty line:\n%s\n' % line)
                ##########################################

                # general query 1: python exceptions
                if line.find('Error:') > 0: 
                    m = p_exception.match(line)
                    if m: 
                        ret['err_type'] = m.group('error_type')
                        ret['err_msg'] = m.group('message')
        return ret  # keys: err_type, err_msg, output

    @staticmethod
    def isASuccess(fpath, descriptors=[], offset=0, verbose=1): 
        # offset: start reading from this line 
        tSuccess = False
        missed = set()
        with open (fpath, 'rt') as fp:  # Open file for reading of text data. 't': open in text mode (default)
            if not descriptors: descriptors = [Job.job_success, Job.end_job ]
            for descriptor in descriptors:  

                # [note] the 2nd descriptor will start from where it was left off from reading the first descriptor (first iteration finished on line 18, then the next will start from 19)
                tSuccess, offset = processLine(fp, descriptor=descriptor)
                if not tSuccess: 
                    missed.add(descriptor) 
                    
                # if not tSuccess: break
            # print('(isASuccess) completed %s\n' % os.path.basename(fpath))
        if verbose and len(missed) > 0: 
            print('(isASuccess) Found missing {n} descriptors for job {j} to be considiered a success ...'.format(n=len(missed), j=os.path.basename(fpath)))
            for i, descriptor in enumerate(missed): 
                print('... [{n}] missing {d}'.format(n=i, d=descriptor))
            # show line number offset? ... line # offset: %d
        return tSuccess

    @staticmethod
    def isExecPath(line): 
        line = line.strip()
        # e.g. python /sc/orga/work/chiup04/cluster_ensemble/cf.py /hpc/users/chiup04/work/data/remanei -s 6
        if line.find(Job.compiler) == 0 and line.find(Job.domain) > 0 and line.find("/") > 0:
            return True 
        return False 

### end Class Job

def readlines_reverse(filename, fromstart=None):
    """

    Reference
    ---------
    1. https://stackoverflow.com/questions/2301789/read-a-file-in-reverse-order-using-python
    """
    with open(filename) as qfile:
        # if fromstart: qfile.seek(0)
        qfile.seek(0, os.SEEK_END)
        position = qfile.tell()
        line = ''
        while position >= 0:
            qfile.seek(position)
            next_char = qfile.read(1)
            if next_char == "\n":
                yield line[::-1]
                line = ''
            else:
                line += next_char
            position -= 1
        yield line[::-1]

def processLine(fp, descriptor='Successfully completed'):
    # fp: file object
    tval = False
    offset = 0
    n_nonempty = 0
    for i, line in enumerate(fp): 
        line = line.strip()
        if len(line) > 0: n_nonempty += 1 
        # if n_nonempty == 1: print('(processLine) first nonempty line: %s' % line)
        if line.find(descriptor) >= 0: 
            tval = True
            offset = i
            break
    return (tval, offset)

# cf_output
def parse_output():
    # Imbalanced class distribution: nPos:1285, nNeg:125053, rPos:0.01017112824328389
    class_distribution = re.compile(r'Imbalanced class distribution: nPos:\d+, nNeg:\d+, rPos:\d+(\.\d+)?')


    return

def interpret(cmd, verbose=True):
    """
    Given a command, interpret its parts. 

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
        print('(interpret) executable:\n... {0}\n... domain:{1}\n... project_path: {2}\n'.format(exec_cmd, ret['domain'], ret['project_path']))
    return ret

def name_job(project_path, job_id='', domain=''): 
    if not domain: domain = os.path.basename(project_path)
    job_name = 'cf_{domain}'.format(domain=domain) 
    job_id = kargs.get('job_id', '')
    if job_id: job_name = '{base}-{id}'.format(base=job_name, id=job_id)
    return job_name

def cluster_cmd(job_name='?', allocation='acc_pandeg01a', **kargs):
    """

    Memo
    ----
    1. If the 'cmd' in return cmd is left out 
       ~> python: can't open file 'None': [Errno 2] No such file or directory 

    2. allocation='acc_pandeg01a' doesn't seem to work
       allocation='acc_rg_parsons' is only for testing? 
       use acc_pandeg01a

    """
    if job_name in ('?', '', ): job_name = 'cf_{domain}'.format(domain=domain) 
    job_id = kargs.get('job_id', '')
    if job_id: job_name = '{base}-{id}'.format(base=job_name, id=job_id)

    walltime = kargs.get('walltime', '00:10')
    if isinstance(walltime, int) or (isinstance(walltime, str) and walltime.find(':') < 0): # only given hours 
        walltime = '%s:00' % walltime

    n_cores = kargs.get('n_cores', 6)
    queue = kargs.get('queue', 'expressalloc') 

    cmd = 'rc.py --name {J} --cores {n} --walltime {t} --queue {q}'.format(J=job_name, n=n_cores, t=walltime, q=queue)
    if allocation: 
        cmd += ' ' + '--allocation %s' % allocation  
    if kargs.get('interactive', False): 
        cmd += ' ' + '--interactive'

    ### extract switches to modify the cluster command
    if 'memory' in kargs: 
        assert isinstance(kargs['memory'], int) or isinstance(kargs['memory'], str)
        cmd += ' ' + '--memory {mem}'.format(mem=kargs['memory'])
    if 'workdir' in kargs: 
        assert os.path.exists(kargs['workdir']), "Invalid workdir: %s" % kargs['workdir']
        cmd += ' ' + '--workdir {path}'.format(path=kargs['workdir'])
    if kargs.get('dry_run', False): 
        cmd += ' ' + '--dryrun'
    return cmd  

def CF_run(app='cf.py', **kargs):
    # code_dir, project_path, size, fold, seed, RULE, strategy, exit, algo, conv, age, epsilon, start = parameters
    # cmd = 'python %s/rl/run.py -i %s -o %s/RL_OUTPUT/ORDER%s/ -np %s -fold %s -m %s -seed %i -epsilon %s -rule %s -strategy %s -exit %i -algo %s -age %i -conv %i -start %s' % \
    #         (code_dir, project_path, project_path, seed, size, fold, metric, seed, epsilon, RULE, strategy, exit, algo, age, conv, start) 
    print('(CF_run) code_dir: {0}\n... domain:{1}\n... project_path: {2}\n'.format(code_dir, Job.domain, Job.path))
    
    project_path = kargs.get('project_path', Job.project_path)
    if app.find(' ') > 0: # app has options 
        app, options = ' '.join(app.split()[:1]), ' '.join(app.split()[1:])
        src_path = '{prefix}/{main}'.format(prefix=code_dir, main=app)
        cmd = 'python {app} {input} {options}'.format(app=src_path, input=project_path, options=options)
    else: 
        src_path = '{prefix}/{main}'.format(prefix=code_dir, main=app)
        cmd = 'python {app} {input}'.format(app=src_path, input=project_path)     

    if kargs.pop('use_cluster', False):
        cmd = 'python {hpc_cmd} \"{src_cmd}\"'.format(hpc_cmd=cluster_cmd(**kargs), src_cmd=cmd)
    print("(CF_run) Intermediate command:\n%s\n" % cmd)
    # system(cmd)

    return

def hasLine(fp, descriptor='Successfully completed'):
    # fp: file object
    tval = False
    for line in fp: 
        line = line.strip()
        if line.find(descriptor) >= 0: 
            tval = True
            break
    return tval


###########################

# test_suite = 'test_wms_probs'
# job_success = "Successfully completed"
# end_job = "Completed testing"

# cf_run
def parse_args(): 
    import time, os
    from optparse import OptionParser

    timestamp = now = time.strftime("%y%m%d-%H%M%S", time.localtime(time.time())) # time()
    parser = OptionParser()
    parser.add_option('-c', '--cores', dest = 'cores', type = 'int')
    parser.add_option('-w', '--walltime', dest = 'walltime')
    parser.add_option('-a', '--allocation', dest = 'allocation', default = 'andromeda')
    parser.add_option('-q', '--queue', dest = 'queue', default = 'expressalloc')
    parser.add_option('-n', '--name', dest = 'name', default = timestamp)
    parser.add_option('-k', '--keyword', dest = 'keyword', default = 'cf')
    parser.add_option('--memory', dest='memory', type='int', default=32768)
    parser.add_option('--interactive', action="store_true", dest="interactive", default=False)
    parser.add_option('--workdir', dest='workdir', default=os.getcwd())
    parser.add_option('--dryrun', action="store_true", dest="dryrun", default=False)
    parser.add_option('--kill-pending', action='store_true', dest="kill_pending", default=False)

    ### use command line options to configure the job spec and system variables
    (options, args) = parser.parse_args()
    Job.cores = options.cores
    Job.walltime = options.walltime
    Job.queue = options.queue
    Job.name = options.name
    Job.allocation = options.allocation
    Job.interactive = options.interactive
    Job.memory = options.memory
    Job.working_dir = options.workdir
    Job.keyword = options.keyword

    System.dry_run = options.dryrun
    System.kill_pending = options.kill_pending
    if System.kill_pending: 
        System.parse_mode, System.filter_mode = False, True

    return options, args

def config():
    ### obtain job info from the command line, configuraiton file (e.g. config.txt), ... 
    #   ... and weka-generated classifier directories
    import time, getpass
    # from utilities import load_properties, get_num_cores  # cluster_cmd

    ret = {}
    timestamp = ret['timestamp'] = time.strftime("%y%m%d-%H%M%S", time.localtime(time.time())) # time()
    print("(config) starting at %s . . .\n" % timestamp)

    System.path_exec = ret['code_dir'] = dirname(abspath(argv[0]))
    print('... exec path: %s' % System.path_exec)

    # user = getpass.getuser() # 'pleiades' 
    Job.options, Job.args = parse_args()  # Job.parse_args()
    # Job.compiler = 'python'

    # ensure project directory exists
    try: 
        Job.config(prefix=abspath(Job.args[0]))
    except: 
        if System.parse_mode: 
            print('Hint: Add project directory e.g. /hpc/users/<userid>/work/data/pf3')
            raise ValueError
        else: 
            # it's okay to ignore project path :)
            pass

    if System.parse_mode:  # ... typical use case is to parse the job output and identify successful jobs
        assert exists(Job.project_path)
        print('... project path: {path}\n... user: {name}\n... domain: {subject}'.format(name=Job.user, 
            path=Job.project_path, subject=Job.domain))
 
        # directory    = 'RL_OUTPUT'
        # subdirectory = 'ORDER'
    
        System.classifier_dirnames =  ret['dirnames'] = sorted(filter(isdir, glob.glob('%s/weka.classifiers.*' % Job.project_path)))

        ### load and parse project properties
        # p            = load_properties(Job.project_path)
        # fold_count   = int(p['foldCount'])
        # seeds        = int(p['seeds'])
        # # metric       = p['metric']
        # # RULE         = p['RULE']
        # use_cluster  = True if str(p['useCluster']).lower() in ['1', 'y', 'yes', 'true', ] else False
    else:  # ... sometimes we wish to use it as a wrapper of HPC commands (e.g. kill pending jobs)
        # filter mode 
        pass

    return ret 

def parse(job_keyword='cf', test_suite='test_wms_probs'): 
    ### keywords
    # Job.job_success = "Successfully completed"
    # Job.end_job = "Completed testing"

    div("Job output file keyword: '{w}'".format(w=job_keyword), symbol="%")
    output_path = Job.job_file_pattern(keyword=job_keyword) # keywords in the job output file

    # e.g. Subject: Job 122904254: <step1a_generate-elegans> in cluster <minerva> Done
    p_job = Job.p_submitted
    success_descriptors = [Job.job_success, Job.end_job ]
    n_jobs = n_success = n_error = n_id_missing = 0
    jobs = {}
    jobs_common = {}
    jobs_incomplete = {}
    jobs_questionable = {}

    ofiles = sorted([os.path.basename(fp) for fp in glob.glob(output_path)])
    div('(parse) Init: path:\n%s\n... found (n=%d):\n%s\n' % (output_path, len(ofiles), ofiles), symbol='#', border=2)
    n_files = len(ofiles)
    tidx = set( random.sample( range(n_files), min(n_files, 10)) )
    for ofile in sorted([os.path.basename(fp) for fp in glob.glob(output_path)]):  # foreach (sorted) files according to timestamps
        
        ### file info
        fpath = os.path.join(Job.path_log, ofile)
        # mtime = os.path.getmtime(fpath)
        tSuccess =  Job.isASuccess(fpath, offset=0, verbose=1 if n_jobs in tidx else 0)  # params: descriptors=success_descriptors 

        cur_job_id = '?'
        n_queries = 2
        if tSuccess: 
            n_success += 1
            
            n_nonempty = 0
            exec_components = {}
            match_states = [0] * n_queries # n_queries
            with open(fpath, 'rt') as fp: 
                fp.seek(0)
                for i, line in enumerate(fp): 
                    line = line.strip('\n') 
                    if len(line) > 0: n_nonempty += 1
                    
                    # hypothesis: all jobs success or not should have the following info ... 
                    ########################################

                    # general query 1 
                    # if line.find(System.path_exec) > 0: # will not work for different computers 
                    if Job.isExecPath(line): 
                        cmd = line.strip()
                        print("(parse) Cmd? %s" % cmd)
                        # e.g. python /sc/orga/work/chiup04/cluster_ensemble/cf.py /hpc/users/chiup04/work/data/pf1 -s 5
                        exec_components = interpret(line, verbose=True) # 'compiler', 'src_path', 'project_path', 'options'
                        match_states[0] = True

                    # general query 2: extract job info
                    m = p_job.match(line)
                    if m: 
                        job_id, job_name, queue, status = m.group('id'), m.group('name'), m.group('cluster'), m.group('status')
                        jobs[job_id] = {}
                        jobs[job_id]['name'] = job_name
                        jobs[job_id]['queue'] = queue
                        jobs[job_id]['status'] = status

                        cur_job_id = job_id
                        match_states[1] = True
                    ########################################


            # verify
            assert jobs[cur_job_id]['status'].lower() == 'done', "Success but questionable status? %s" % jobs[cur_job_id]['status']
            if all(match_states): 
                cmd = os.path.basename(exec_components['src_path']) + ' ' + exec_components['options']
                jobs[cur_job_id]['cmd'] = cmd
                print('... job_id: %s was a success | name: %s, main cmd: %s' % (cur_job_id, jobs[cur_job_id]['name'], cmd))
            # print('... read %d nonempty lines in %s' % (n_nonempty, os.path.basename(fpath)) )
        else: 
            ret = Job.parse_output(fpath, reset=True, verbose=1 if n_jobs in tidx else 0) # keys: output, submitted, id, name, queue, cmd, code, path

            if Job.isIncomplete(fpath, level=0, reset=True):     
                cur_job_id = job_id = ret['id']
                print('... Incomplete job: {id} | exit code: {code}, err file: {path}'.format(id=job_id, code=ret.get('code', '?'), path=ret.get('path', '?')))
                if not job_id in jobs_incomplete: jobs_incomplete[job_id] = {}
                jobs_incomplete[job_id].update(ret) 
                
            else: 
                # other errors (may be in err file)
                fpath_err = fpath.replace('.out.txt', '.err.txt')
                ret.update(Job.parse_error(fpath, reset=True, verbose=1 if n_jobs in tidx else 0)) # keys: err_type, err_msg, output (note that output will be overwritten)

                cur_job_id = job_id = ret.get('id', '?')
                if not job_id in jobs_questionable: jobs_questionable[job_id] = {}
                jobs_questionable[job_id].update(ret)  # should contain at least keys: 'output'
            
        ### extra job info
        if not cur_job_id in jobs_common: jobs_common[cur_job_id] = {}
        # Q: do all jobs have a job ID? 
        if cur_job_id == '?': n_id_missing += 1 
        jobs_common[cur_job_id]['output'] = os.path.basename(fpath)

        mtime = '?'
        try: 
            mtime = time.strftime("%m/%d/%H:%M", time.localtime(os.path.getmtime(fpath))) 
        except: 
            pass

        ctime = '?'
        try: 
            ctime = time.strftime("%m/%d/%H:%M", time.localtime(os.path.getctime(fpath)))
        except: 
            pass
        jobs_common[cur_job_id]['mtime'] = mtime
        jobs_common[cur_job_id]['ctime'] = ctime

        ### end if not a success 

        n_jobs += 1
    ### end foreach job output

    n_error = len(jobs_incomplete)+len(jobs_questionable)
    print("(parse) Success ratio (r: {ratio} = {n}/{N})".format(ratio=n_success/(n_jobs+0.0) if n_jobs > 0 else 'n/a', n=n_success, N=n_jobs))
    print("... total error detected: {ne} | n_error+n_success: {Nh} =?= n_jobs: {N} | n_missing_ids: {nm}".format(ne=n_error, Nh=n_success+n_error, N=n_jobs, nm=n_id_missing))

    if jobs_incomplete: 
        div("Failed jobs (n=%d)" % len(jobs_incomplete), symbol='#')
        for job_id, entry in jobs_incomplete.items(): 
            shared_entry = jobs_common[job_id]
            print('ID: {id} | {output} | time: {time}, exit code: {code} | name: {name}, queue: {queue}, exec:\n>>> {cmd}'.format(id=job_id, 
                output=shared_entry['output'], time=shared_entry['mtime'],
                code=entry.get('code', '?'),  
                name=entry['name'], queue=entry['queue'], cmd=entry['cmd'] ))
            print('... code: {code}, error file: {file}\n'.format(code=entry['code'], file=entry.get('path', '?')))
    
    # questionable jobs 
    #    a. submission failture 
    #    b. premature exits  
    if jobs_questionable: 
        div("Questionable jobs (n=%d)" % len(jobs_questionable), symbol='#')
        nq = len(jobs_questionable.keys())
        for job_id in random.sample(jobs_questionable.keys() , min(nq, 10)): 
            shared_entry = jobs_common[job_id]
            entry = jobs_questionable[job_id]
            print('ID: {id} is questionable | output: {o}, exit code? {e} | time: {t}'.format(id=job_id, e=entry.get('code', '?'), o=shared_entry['output'], t=shared_entry['mtime']))

    print()
    if jobs: 
        div("Completed jobs (n=%d)" % len(jobs), symbol='#')
        for job_id, entry in jobs.items(): 
            shared_entry = jobs_common[job_id]
            print('ID: {id} | {output} | time: {time} | name: {name}, queue: {queue}, exec:\n>>> {cmd}\n'.format(id=job_id, 
                output=shared_entry['output'], time=shared_entry['mtime'], name=entry['name'], queue=entry['queue'], cmd=entry['cmd'] ))
    else: 
        div("<<< No completed jobs of domain: {dataset} at {path} >>>".format(dataset=Job.domain, path=os.path.dirname(output_path)), symbol='#')
    
    

    return

def remove_pending_jobs(): 
    """

    Memo
    ----
    1. Delete empty line using 'sed'

       https://stackoverflow.com/questions/16414410/delete-empty-lines-using-sed

    """
    import subprocess, re 
    # command that list all pending jobs (but will return blank lines)
    cmd = "bjobs -u chiup04 -p | cut -d' ' -f1 | sed '/^$/d' | grep -v 'JOBID'"

    result = subprocess.check_output(cmd, shell=True)
    result = result.decode("utf-8")
    # print('\n(remove_pending) output:\n%s\n' % result)
    p_id = re.compile(r'\d+')
    jobs = []
    for line in result.split(): 
        job_id = line.strip()
        assert p_id.match(job_id) is not None, "Invalid job_id: %s" % job_id
        jobs.append(job_id)

    # kill the pending jobs 
    cmd = 'bkill {job_ids}'.format(job_ids=" ".join(jobs))
    print('(remove_pending) Final command:\n%s\n' % cmd)
    result = subprocess.check_output(cmd, shell=True)
    result = result.decode("utf-8")
    print('\n%s\n' % result)

    return

def run(): 
    """

    Usage
    -----
    <parser mode> 
        * parse job output (e.g. domain = 'drosophila')

          python parse_job.py /hpc/users/chiup04/work/data/drosophila

    <filter mode> 
    
        * kill pending jobs: 

            python parse_job.py --kill-pending

    """

    config() 
    # ... now we know the spec for Job and System
    actions = {k:0 for k in ['parse', 'remove', ]}
    if System.parse_mode: # parse mode 
        Job.end_job = '(test_wmf_probs_suite) Completed'  # pattern: (routine) Completed.
        parse(job_keyword=Job.keyword, test_suite='test_wms_probs')
        actions['parse'] = 1
    else: # filter mode
        if System.filter_mode: 
            if System.kill_pending: 
                remove_pending_jobs()
                actions['remove'] = 1
        # ... other usage modes go here
    if sum(actions.values()) == 0: 
        raise NotImplementedError("Unrecognized mode> Opts: %s, args: %s" % (Job.options, Job.args))

    return

if __name__ == "__main__": 
    run()




from itertools import product
import os, sys
from os import environ, system, makedirs
from os.path import abspath, dirname, exists, isdir
from sys import argv
from glob import glob
from utilities import load_properties, get_num_cores  # cluster_cmd
from sklearn.externals.joblib import Parallel, delayed
# import numpy as np

### local modules 
import parse_job
from utils_sys import div, parse_params_list

# moniter execution time
import timing

# import rl

### Job related global variables
# utils_job 
class Job(parse_job.Job): 
    job_type = 'main'  # values: {'generate', 'combine', 'main'/'cf', }
    job_type_main = ['main', 'cf', ]
    raw_opts = ''

    bag_count = 5 
    fold_count = 5 
    nested_fold_count = 5

    # Job request: cluster attributes
    user = ''
    n_cores = cores = None 
    walltime = None
    queue = None
    name = None    # related: keyword
    allocation = None
    interactive = False
    memory = 8096

    job_id = meta = ''  # additional user-specified job ID

    ### parameters for LENS project incorporating RL
    seeds        = 1 
    # metric       = p['metric']
    # RULE         = p['RULE']
    use_cluster  = True 
    # strategies   = ['greedy', 'pessimistic', 'backtrack']
    # conv_iters   = int(p['convIters'])
    # age          = int(p['age'])
    # epsilon      = p['epsilon']
    # exits        = [0]
    # algos        = ['Q']
    # start_states = '0' #start randomly ('best' also an option, see rl/run.py)
    dirnames = []
    max_num_clsf = len(dirnames) * seeds
    sizes        = range(1,max_num_clsf+1)

    ### algorithm parameters 
    
    # key algorithmic parameters 
    # alpha = None 
    # n_factors = None

    # maps from parameter name to its option (for reconstructing options)
    control_params = {'alpha': '--alpha', 
                      'n_factors': '--n-factors', 
                      'policy_opt': '--policy-opt', 
                      'n_runs': '--runs', 
                      'n_runs_modelselect': '--runs-model-select'}

    # params that play into the role of specifying job_id
    # >>> ordering matters
    meta_params = [('n_factors', 'n'), ('alpha', 'a'), ('policy_opt', 'p'), ] 
    params = {}

    @staticmethod
    def name_job(app='', opt='', domain='', job_id=''): 
        if not domain: domain = Job.domain 
        if not app: app = Job.app  # see CF_run()

        # opt: options for the main execuable
        if not opt: 
            opt = Job.options_exe  # see CF_run()
            print('(name_job) options_exec: %s' % opt) # e.g. -s 1 --alpha 78 --n-factors 12
            
            # remove a subset of control parameters from the naming (too verbose)
            ########################################################################
            opt_values = Job.options_exe.split()
            opt = ''.join(opt_values[:2])   # [todo]

            # print('... opt_values: {ov} > opt: {o}'.format(ov=opt_values, o=opt))
            ########################################################################

        assert app.find('.') > 0, "Invalid executable? %s" % app if len(app) > 0 else 'app string is empty'
        app = app.split('.')[0]
        if opt: app += ''.join(opt.split())

        job_name = '{exec}-{domain}'.format(exec=app, domain=Job.domain)  

        if not job_id: job_id = Job.meta
        print('(name_job) domain: %s | app: %s, opts: %s | job_id: %s' % (domain, app, opt, 'n/a' if not job_id else job_id))
        print('...        job name: %s domain: %s, id: %s' % (job_name, domain, job_id))

        if job_id: job_name = '{prefix}-{suffix}'.format(prefix=job_name, suffix=job_id)
        return job_name
    
    @staticmethod
    def parse_params(pstr, seq_eq='=', sep=','): 
        if pstr is None or not pstr: return {}

        # parse
        for param in pstr.split(sep): 
            param = param.strip()
            assert param.find(seq_eq) > 0, "Invalid parameter string: %s" % pstr
            pvar, pval = param.spilt(sep_eq)
            pvar, pval  = p.strip(), p.strip()
            Job.params[pvar] = pval
        return params
    @staticmethod
    def parse_meta(): 
        """
        Convert parameters (or their subset) into a string represention for file or job naming. 

        """
        # from utils_sys import parse_params_list
        # e.g. python cf_run.py --batch -t main -m "n10a50" --nfact 10 --alpha 50
        #      we want --nfact 10 --alpha 50 
        #              -m "n10a50"
        job_id = ''
        for param, p in Job.meta_params:
            val = Job.options.get(param, None)  # raw value
            if val is not None: 
                # parse comma-separated list
                vlist = parse_params_list(val)
                if len(vlist) > 1: 
                    # multiple parameters (part of parameter grid)
                    val = '%s-%s' % (min(vlist), max(vlist))
                job_id += '{short_name}{value}'.format(short_name=p, value=val)
        return job_id

    @staticmethod
    def rebuild_options(): 
        opts = ''
        for param, opt in Job.control_params.items(): 
            val = Job.options.get(param, None) 
            if val is not None: # if job.options has its value ...
                if str(val).find(' ') > 0:  
                    opts += "{s} '{v}'{tail}".format(s=opt, v=val, tail=' ') 
                else: 
                    opts += '{s} {v}{tail}'.format(s=opt, v=val, tail=' ') 
        opts = opts.rstrip()
        return opts

class System(parse_job.System): 
    multi_domains = False
    dry_run = False
    has_configured = False
    domains = []   # if not empty, then we are in a multi-domain mode
    current_domain = ''  # single domain mode; focus only on this particular domain and its dataset

    # system default values
    working_dir = os.path.join(os.path.dirname(os.getcwd()), 'data')

    @staticmethod
    def parse_domains(): 
        pass

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
    # job_id is an extra job descriptor to distinguish jobs of similar nature
    if job_name in ('?', '', ): Job.name = job_name = Job.name_job(job_id=kargs.get('job_id', '')) # domain is a global var
    
    walltime = kargs.get('walltime', Job.walltime)
    if isinstance(walltime, int) or (isinstance(walltime, str) and walltime.find(':') < 0): # only given hours 
        walltime = '%s:00' % walltime

    n_cores = kargs.get('n_cores', Job.n_cores)
    queue = kargs.get('queue', Job.queue) 

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
    
    code_dir = System.path_exec
    print('(CF_run) cwd: {0}\n... domain:{1}\n... project_path: {2}\n... app: {3}\n ... (verify) #'.format(code_dir, Job.domain, Job.project_path, app))

    # normalize command line options (put input file first)
    if Job.raw_opts:  # user provided the option string
        Job.app = app 
        Job.options_exe = Job.raw_opts
        src_path = '{prefix}/{main}'.format(prefix=code_dir, main=Job.app)

        cmd = 'python {app} {options}'.format(app=src_path, options=Job.options_exe)
        print('(CF_run) raw cmd: %s' % cmd)
    else: 
        if app.find(' ') > 0: # app has options 
            Job.app, Job.options_exe = ' '.join(app.split()[:1]), ' '.join(app.split()[1:])
            src_path = '{prefix}/{main}'.format(prefix=code_dir, main=Job.app)

            #############################################
            # cmd templates that depend on the job type
            if Job.job_type.startswith( ('anal', )):  # analysis 
                cmd = 'python {app} {options}'.format(app=src_path, options=Job.options_exe)
            else: 
                cmd = 'python {app} {input} {options}'.format(app=src_path, input=Job.project_path, options=Job.options_exe) 
            #############################################
        else: 
            Job.app = app
            src_path = '{prefix}/{main}'.format(prefix=code_dir, main=app)

            #############################################
            # cmd templates that depend on the job type
            if Job.job_type.startswith(('anal', )):
                cmd = 'python {app}'.format(app=src_path)
            else:   
                cmd = 'python {app} {input}'.format(app=src_path, input=Job.project_path)  
            #############################################   

    # condition: class Job has been configured 
    if kargs.pop('use_cluster', False):
        cmd = 'python {hpc_cmd} \"{src_cmd}\"'.format(hpc_cmd=cluster_cmd(**kargs), src_cmd=cmd)
    div("(CF_run) Intermediate command:\n%s\n ... (verify)" % cmd, symbol='#')
    
    # [note] if any errors occurred within rc.py, the error may not be shown on the terminal
    system(cmd)

    return
    
print("(cf_run) Starting . . .\n")

### cf_run expects only 1 argument (i.e. the project path), everything else is specified in the config.txt

# ensure project directory exists
# try: 
#     project_path = abspath(argv[1])
# except: 
#     print('Hint: Add project directory e.g. /hpc/users/<userid>/work/data/pf3')
#     raise ValueError

# Job.domain = domain = os.path.basename(project_path)

# assert exists(project_path)
# code_dir     = dirname(abspath(argv[0]))
# # directory    = 'RL_OUTPUT'
# # subdirectory = 'ORDER'
# dirnames     = sorted(filter(isdir, glob('%s/weka.classifiers.*' % project_path)))

# # load and parse project properties
# p            = load_properties(project_path)
# fold_count   = int(p['foldCount'])
# bag_count = 
# nested_fold_count = int(p['nestedFoldCount'])

# seeds        = int(p['seeds'])
# # metric       = p['metric']
# # RULE         = p['RULE']
# use_cluster  = True if str(p['useCluster']).lower() in ['1', 'y', 'yes', 'true', ] else False
# # strategies   = ['greedy', 'pessimistic', 'backtrack']
# # conv_iters   = int(p['convIters'])
# # age          = int(p['age'])
# # epsilon      = p['epsilon']
# # exits        = [0]
# algos        = ['Q']
# start_states = '0' #start randomly ('best' also an option, see rl/run.py)
# max_num_clsf = len(dirnames) * seeds
# sizes        = range(1,max_num_clsf+1)


# if not exists("%s/RL_OUTPUT/" % project_path):
#     makedirs("%s/RL_OUTPUT/" % project_path)

# for o in range(seeds):
#     if not exists("%s/RL_OUTPUT/ORDER%i" % (project_path, o)):
#         makedirs("%s/RL_OUTPUT/ORDER%i" % (project_path, o))

# all_parameters = list(product([code_dir], [project_path], sizes, range(fold_count), range(seeds), [RULE], strategies, exits, algos, [conv_iters], [age], [epsilon], [start_states]))
# Parallel(n_jobs = get_num_cores(), verbose = 50)(delayed(RL_run)(parameters) for parameters in all_parameters)

def parse_params(pstr, seq_eq='=', sep=','): 
    if pstr is None or not pstr: return {}

    params = {}
    for param in pstr.split(sep): 
        param = param.strip()
        assert param.find(seq_eq) > 0, "Invalid parameter string: %s" % pstr
        pvar, pval = param.spilt(sep_eq)
        pvar, pval  = p.strip(), p.strip()
        params[pvar] = pval

    return params

def parse_args(): 
    import time, os, sys
    from optparse import OptionParser

    timestamp = now = time.strftime("%y%m%d-%H%M%S", time.localtime(time.time())) # time()
    parentdir = os.path.dirname(os.getcwd())

    # home_dir = os.path.expanduser('~')
    # working_dir_default = '/'.join([home_dir, 'work/data', ])
    working_dir_default = os.path.join(parentdir, 'data')  # e.g. /Users/<user>/work/data
    # ... cf: project_path, e.g. /Users/<user>/work/data/pf1, which includes the domain or dataset name

    parser = OptionParser()

    ### wrapper options for HPC jobs (they are mostly pre-determined within this wrapper module; see runJobs())

    parser.add_option('-c', '--cores', dest = 'cores', type = 'int')
    parser.add_option('-w', '--walltime', dest = 'walltime')
    # parser.add_option('-a', '--allocation', dest = 'allocation', default = '')
    parser.add_option('-q', '--queue', dest = 'queue', default = 'premium')
    # parser.add_option('--memory', dest='memory', type='int', default=32768)
    parser.add_option('-n', '--name', dest = 'name', default = '')

    # parser.add_option('-k', '--keyword', dest = 'keyword', default = 'cf')  
    # parser.add_option('-e', '--exec', dest = 'exec', default = 'cf')
    parser.add_option('-m', '--meta', dest = 'meta', default = '')
    parser.add_option('-d', '--domains', dest='domains', default='')
    parser.add_option('--domain', dest='domain', default='')
    
    parser.add_option('-t', '--type', dest = 'job_type', default = 'main')
    parser.add_option('--chain', dest='chain', default='')

    parser.add_option('-o', '--options', dest='raw_opts', default='')   # command line options of the given executable (specified via the job type)
 
    ### wrapper options (wrapping around cf.py)
    # [note] cannot use short-hand switches -a, -s because they are being taken
    parser.add_option('--nfact', '--n-factors', dest='n_factors')  # can be a comma separated list of values or just a single value; None if not specified
    parser.add_option('--alpha', dest='alpha')  # can be a comma separated list of values or just a single value
    # a more generalized format 
    parser.add_option('--params', dest='params')  # comma separated, assignment via equal sign e.g. "n_factors=5, alpha=6, policy='rating'"
    parser.add_option('--runs', dest='n_runs', type='int', default=5)
    parser.add_option('--runs-model-select', dest='n_runs_modelselect', type='int', default=1)
    parser.add_option('-s', '--settings', dest = 'settings', default='')

    # run multiple domains, if given
    parser.add_option('--batch', action="store_true", dest='multi_domains', default=False)
    parser.add_option('--interactive', action="store_true", dest="interactive", default=False)
    parser.add_option('--workdir', dest='workdir', default=working_dir_default)
    parser.add_option('--dryrun', action="store_true", dest="dryrun", default=False)

    ### use command line options to configure the job spec and system variables
    (options, args) = parser.parse_args()
    Job.options = vars(options)  # to dictionary object
    Job.args = args

    Job.n_cores = options.cores
    Job.walltime = options.walltime
    Job.queue = options.queue
    Job.name = options.name
    # Job.allocation = options.allocation
    Job.interactive = options.interactive
    # Job.memory = options.memory

    if options.workdir is not None: assert os.path.exists(options.workdir), "(parse_args) Invalid working dir: %s" % options.workdir
    Job.working_dir = options.workdir  # if --batch is on, this is the default 'prefix' of the working directory ... 
    if options.domain: 
        System.multi_domains = False  # single domain mode
        Job.domain = System.current_domain = options.domain

    # ... if --batch is not on, it's necessary to provide a working directory

    # used in parse_job
    # Job.keyword = options.keyword   # keyword in the job output file (which is determined by the name of the executable)
    # Job.exe = options.exec
    Job.job_type = options.job_type  # used to choose the executable (e.g. cf.py vs combine.py) 

    # raw command line options for the executable 
    if options.raw_opts:
        Job.raw_opts = options.raw_opts   # used for executables (e.g. analyze_performance.py) whose command line option string is taken as it is
    
    # note: don't parse the parameters just yet
    #############################################################################
    Job.params['alpha'] = options.alpha        # corresponds to cf.py -a
    Job.params['n_factors'] = options.n_factors   # corresponds to cf.py -n
    Job.params['n_runs'] = options.n_runs 
    Job.params['n_runs_modelselect'] = options.n_runs_modelselect

    if len(options.settings) > 0: 
        Job.params['cases'] = parse_params_list(options.settings, sep=',', dtype=int) # each case is an integer
    Job.parse_params(options.params)
    ############################################################################

    # job chain 
    Job.chain = options.chain
    if Job.chain: Job.chain = parse_params_list(options.chain, sep=' ', type=str) # [j.strip() for j in Job.chain.split(',')]

    Job.job_id = Job.meta = options.meta  # extra user job ID (experiment descriptor such as hyperparameter settings that can be used to distinguish jobs of the same type)
    if not Job.meta: Job.job_id = Job.meta = Job.parse_meta()

    # system instructions
    System.dry_run = options.dryrun
    System.multi_domains = options.multi_domains

    if options.domains: 
        System.domains = [d.strip() for d in options.domains.split(',')]  
        # print('>>> System.domains: %s' % System.domains)
        if len(System.domains) > 0:  # not Job.job_type.startswith(('analy', 'gen', 'combo')) 
            Job.options['multi_domains'] = System.multi_domains = True
   
    # single domain or multiple domains: 
    if System.multi_domains: 
        print('... Job domains (ndom >= 1): {domains}\n...... Job.params:\n{params}\n ... (verify)'.format(domains=System.domains, params=Job.params))
    else:  # single domain 
        print('... Job domain: {domain}\n...... Job.params:\n{params}\n ... (verify)'.format(domain=Job.domain, params=Job.params))

    return 

def config(**kargs):
    ### obtain job info from the command line, configuraiton file (e.g. config.txt), ... 
    #   ... and weka-generated classifier directories
    import time, getpass
    # from utilities import load_properties, get_num_cores  # cluster_cmd

    ret = {}
    System.path_exec = ret['code_dir'] = dirname(abspath(argv[0]))
    # if System.has_configured: 
    #     # do nothing 
    #     return ret 

    print('(config) exec path: %s' % System.path_exec)
    timestamp = ret['timestamp'] = time.strftime("%y%m%d-%H%M%S", time.localtime(time.time())) # time()
    print("... starting at %s . . .\n" % timestamp)

    ### parse job specs
    parse_args()  # Job.parse_args() only parse information pertaining to Job objects

    # Job.compiler = 'python'
    ####################################################
    # e.g. 
    # {'cores': None, 'walltime': None, 'allocation': '', 'queue': 'premium', 
    #  'memory': 32768, 'name': '', 'keyword': 'cf', 'exec': 'cf', 'meta': 'nG1-500aG1-1000', 'domains': '', 
    #  'job_type': 'main', 'chain': '', 'n_factors': '1, 10, 50, 100, 500', 'alpha': '1, 10, 100, 1000', 
    #  'params': None, 'multi_domains': False, 'interactive': False, 'workdir': '/Users/pleiades/Documents/work/data', 'dryrun': True}
    #  args: ['/Users/pleiades/Documents/work/data/pf3']
    if kargs.get('verify', False): print('... (verify) options (type: %s):\n%s\n, args:\n%s\n' % (type(Job.options), Job.options, Job.args))

    # ensure project directory exists
    tSpecifyJob = True 
    if len(Job.args) == 0:
        # assert (System.multi_domains is True and len(System.domains) > 0) or len(System.current_domain) > 0, \
        #     "Missing project path while not in batch mode (i.e. running multiple domains)!"
        if System.multi_domains: 
            assert len(System.domains) > 0
            tSpecifyJob = False   # run config() again with each domain (see runMultiDomains())
        else: 
            # does the user provide her own option string? 
            if Job.raw_opts:
                tSpecifyJob = False    # no need to infer the job parameters
            else:  
                assert len(System.current_domain) > 0, "Missing project path while not in batch mode (i.e. running multiple domains)!"
                tSpecifyJob = True
    
    if tSpecifyJob:   
        project_path = os.getcwd() # the 1st of the remaining argment(s) after removing all options should be the project path
        try: 
            ### attributes configured
            # prefix/project_path
            # path_log
            # domain 
            # user 
            # output_pattern
            # working_dir_default = System.working_dir
            if len(Job.args) > 0: 
                prefix = kargs.get('prefix', abspath(Job.args[0]))   # order kargs -> Job.args[0] -> working_dir_default
                # Job.domain = os.path.basename(prefix) # this is in Job.config()
            else: 
                prefix = os.path.join(Job.working_dir, Job.domain) 

            Job.config(prefix=prefix)  # configure Job.log_path, Job.domain, Job.user, Job.output_pattern, Job.project_path
            project_path = Job.project_path
        except: 
            print('Hint: Add project directory e.g. /hpc/users/<userid>/work/data/pf3')
            raise ValueError
        assert exists(Job.project_path)
        
        # configure Job.user? 
        div('Project path: {path}\n... user: {name}\n... domain: {subject}'.format(name=getpass.getuser(),  
            path=Job.project_path, subject=Job.domain), symbol='%', border=2)
     
        # directory    = 'RL_OUTPUT'
        # subdirectory = 'ORDER'
        System.classifier_dirnames = Job.dirnames = ret['dirnames'] = sorted(filter(isdir, glob('%s/weka.classifiers.*' % Job.project_path)))

        ### load and parse project properties

        # load and parse project properties
        p            = load_properties(project_path)
        Job.fold_count = fold_count   = int(p['foldCount'])
        Job.bag_count =  bag_count = int(p['bagCount'])
        Job.nested_fold_count = nested_fold_count = int(p['nestedFoldCount'])

        Job.seeds        = int(p['seeds'])
        # metric       = p['metric']
        # RULE         = p['RULE']
        Job.use_cluster  = True if str(p['useCluster']).lower() in ['1', 'y', 'yes', 'true', ] else False
        # strategies   = ['greedy', 'pessimistic', 'backtrack']
        # conv_iters   = int(p['convIters'])
        # age          = int(p['age'])
        # epsilon      = p['epsilon']
        # exits        = [0]
        # algos        = ['Q']
        # start_states = '0' #start randomly ('best' also an option, see rl/run.py)
        Job.max_num_clsf = len(Job.dirnames) * Job.seeds
        Job.sizes        = range(1, Job.max_num_clsf+1)

        ### send signals to the System 
        # System.has_configured = True

    return ret  

def runJobs(**kargs): 
    """

    Todo
    ----
    1. input options ovewrite the default setting

    Examples
    --------
    * generate base predictors for the dataset 'remanei' 

    python cf_run.py /hpc/users/chiup04/work/data/remanei -t generate


    * run cf.py on multiple domains

    python cf_run.py --batch -t main

    Memo
    ----
    descriptions = {0: 'baseline', 1: 'item_centered', 2: 'user-centered', 3: 'item-centered-unsupervised', 
                    4: 'user-centered-unsupervised', 5: 'item-centered-low-support', 6: 'preference-masked', 
                    7: 'item-centered-mask-test', 8: 'user-centered-mask-test'}


    """
    def create_job_chain(cases=[]): 
        chain = []
        if Job.params: # the job has parameter settings specified (e.g. n_factors, alpha)
            for case in cases:
               chain.append('cf.py -s {c} {opts}'.format(c=case, opts=Job.rebuild_options()))
        else: 
            chain = ['cf.py -s {c}'.format(c=case) for case in cases] 
        return chain 

    from sklearn.model_selection import ParameterGrid

    config(**kargs)  # reinterpret the command line
    if Job.job_type.startswith(('main', 'cf', 'wmf', )): 
        print('(runJobs) domain: %s, dryrun? %s algorithm params: %s' % (Job.domain, System.dry_run, Job.params))

    # todo: command line option
    job_type = Job.job_type # 'main' # {'generate', 'combine', 'main', }
    walltimes = {'remanei': '80:30',  
                 'thaliana': '80:45', 'drosophila': '80:45', 'elegans': '80:35', 
                 'pacificus': '80:30', 
                 'sl': '90:45'}
    cores = {'pacificus': 48, }  # different datasets have different averaged usage of the number of threads

    job_chain = ['cf.py -s 0', ]  # default job chain (just a single test job)
    hard_jobs = {'remanei': ['cf.py -s 1', 'cf.py -s 3', 'cf.py -s 6'], 
                 'elegans': ['cf.py -s 2', 'cf.py -s 6'], }

    walltime = Job.walltime # '00:50'
    memory = Job.memory     # 8G: 8192
    Job.n_cores = n_cores = cores.get(Job.domain, 48)      # Job.fold_count+1
    if job_type.startswith('gen'): 
        Job.chain = job_chain = ['step1a_generate.py', ]
        walltime = walltimes.get(Job.domain, '10:45') # 10:45 => 10h:45m
        memory = 16000 # 76800 # 51200  # 50G
        n_cores = Job.fold_count * Job.bag_count + 1
        div('(runJobs) Job type: GENERATE BPS | mem: {m}, n_cores: {nc}, walltime: {wt}'.format(m=memory, nc=n_cores, wt=walltime))
    elif job_type.startswith('combine'):
        Job.chain = job_chain = ['combine.py', ]
        walltime = '10:30'
        memory = 8192 # 32768  # 32G 
        n_cores = 1
        div('(runJobs) Job type: COMBINE | mem: {m}, n_cores: {nc}, walltime: {wt}'.format(m=memory, nc=n_cores, wt=walltime))
    elif job_type.startswith('analy'):  # analyze, analysis
        exec_n = 'analyze_performance.py'
        # if Job.raw_opts: 
        #     exec_n = "{exec} {opts}".format(exec=exec_n, opts=Job.raw_opts)

        Job.chain = job_chain = [exec_n, ]

        walltime = '20:30'
        memory = 32768 # 32768  # 32G 
        n_cores = 48 
        div('(runJobs) Job type: ANALYSIS | mem: {m}, n_cores: {nc}, walltime: {wt}'.format(m=memory, nc=n_cores, wt=walltime))
        # print('... Analysis Mode | domains: {d}'.format(d=System.domains))
    else:  
        # example command 
        # python cf.py /Users/pleiades/Documents/work/data/pf1 --nfact 40 --alpha 5
        #   ~> n_factors <- 40, alpha <- 5
 
        if job_type.startswith(('main', 'test_wmf', 'wmf', )):  # e.g. test_wmf_probs_suite 
            entry_exec = 'cf.py'

            # parameters for the main app
            ############################################################
            n_factors, alpha = Job.params.get('n_factors', None), Job.params.get('alpha', None)
            n_runs, n_runs_modelselect = Job.params.get('n_runs', 5), Job.params.get('n_runs_modelselect', 10) 
            ############################################################

            # -s <number> corresponds to different setting of the same test (e.g. test_wmf_probs_suite)
            hardOnly = False

            ### hard jobs
            minMem, maxMem = 16000, 76800
            if Job.chain:  # if specified, must be a comma-separated string  
                # command line can specifiy a set of options for the execuable

                # check if the executable is already specified
                chain = []
                for je in Job.chain: 
                    if je.find('.py') > 0:
                        chain.append(je) # a complete job spec/cmd 
                    else: 
                        cmd_exec = "{exec} {opt}".format(entry_exec, je)
                        chain.append(cmd_exec)
                Job.chain = chain
            else: 
                # if hardOnly: 
                #     cases = [1, 3, 6, ]
                #     Job.chain = job_chain = create_job_chain(cases) 
                #     walltime = '130:30' if Job.walltime
                #     # memory = 76800
                # else: 

                ### general
                # promising: [2, 4, 7, 8] # all: list(range(1, 8+1)), hard: [1, 3, 6]

                # NOTE: unless the setting is incorporated in the dataset name, please DO NOT mix different 
                #       policy pairs ... 
                #       31, 32: policy_threshold <- fmax  (instead of 'prior')
                #       51, 52: conf_measure <- uniform (instead of Brier score)
                cases = Job.params.get('cases', [5, 6, 9, 10, ]) # [7, 8], [21, 22]
                # ... [1, 2, 22, 32, 42, 44, 46, 52, 63, 72]
                Job.chain = job_chain = create_job_chain(cases)
                
                if Job.walltime: 
                    # [todo] parse walltimes
                    walltime = [Job.walltime, ] * len(job_chain)
                else: 
                    walltime = [walltimes.get(Job.domain, '100:30')] * len(job_chain) 

                memory = [80000, ] * len(cases) # [8192, ] + [minMem] * (len(job_chain)-1)  # 75G: 76800
                # ... polarity modeling via CRF may consume lots of memory

            div('(runJobs) Job type: MAIN ({exe}) | mem: {m}, n_cores: {nc}, walltime: {wt}'.format(exe=Job.chain[-1], m=memory, nc=n_cores, wt=walltime))
        else: 
            raise NotImplementedError("Unknown job type: %s" % job_type)

        ### ... add other use/test cases here indexed by job_type

        # if n_factors is not None: Job.chain = job_chain = ['%s --nfact %s' % (jc, n_factors) for jc in job_chain] 
        # if alpha is not None: Job.alpha = alpha = ['%s --alpha %s' % (jc, alpha) for jc in job_chain]

    # [note] cf.py -s 0 corresponds to setting #0, which is just a reference job (for the purpose of testing the cluster)
    param_grid = kargs.get('param_grid', 
        {'app':job_chain, })  # e.g. ['cf.py -s 1', 'cf.py -s 2', ] => test setting 1 and setting 2
    
    print('... queue: {q}, dry_run? {dr}, user job ID {meta}, size of job chain: {n}'.format(q=Job.queue, dr=System.dry_run, meta=Job.meta, n=len(Job.chain)))
    print('... (verify) job chain: %s' % param_grid['app'])  # e.g. ['cf.py -s 1 --alpha 78 --n-factors 12', ]

    # very hyperparameters 
    n_target_methods = 0
    for i, params in enumerate(list(ParameterGrid(param_grid))): 
        CF_run(app=params['app'], walltime=walltime[i] if isinstance(walltime, list) else walltime, 
            job_id=Job.meta, workdir=Job.project_path, 
                use_cluster=True, dry_run=System.dry_run, 
                    interactive=Job.interactive, 
                    queue=Job.queue, n_cores=n_cores, 
                    memory=memory[i] if isinstance(memory, list) else memory)  # memory is units of Mb
     
    return
### alias
runSingleDomain = runJobs

def runMultiDomains(**kargs): 
    """
    Run the same executable (e.g. cf.py) but with different use cases specified via command line options.

    Todo
    ----
    1. reconstruct options from the option dictionary produced after parsing the command line

    """
    import subprocess
    from time import sleep
    import numpy as np

    # submit jobs wrt to a set of data sources
    # other data sources: 'sl', 'metabric', 'gi'
    domains_baseline = ['pf1', 'pf2', 'pf3', ]
    domains_complex = ['thaliana', 'drosophila', 'elegans', 'pacificus', 'remanei', 'sl', ]
    domains = domains_baseline + domains_complex if not System.domains else System.domains
    print('(runMultiDomains) domains: {}'.format(domains))
    
    # parameters for the main app
    ############################################################ 
    n_factors, alpha = Job.params.get('n_factors', None), Job.params.get('alpha', None)
    n_runs, n_runs_modelselect = Job.params.get('n_runs', 5), Job.params.get('n_runs_modelselect', 10) 
    ############################################################
    job_type = Job.job_type # 'main' # {'generate', 'combine', 'main'}
    meta = user_job_id = Job.meta  # user-defined job ID; will get appended to the job name
    dry_run = System.dry_run
    

    basedir = Job.working_dir
    if basedir is None: basedir = os.path.join(os.path.dirname(os.getcwd()), 'data')
    assert os.path.exists(basedir), "Invalid prefix: %s" % basedir

    # 'basedir' is parent directory for all datasets ... 
    # e.g. domain: 'remanei' => <basedir>/remanei will be the project path (where remanei's dataset is hosted)
    print('(runMultiDomains) data base dir: %s' % basedir)  

    app = 'cf_run.py'
    acc = 0
    for domain in domains:  # foreach domain ... one domain per job
        project_path = os.path.join(basedir, domain)
        assert os.path.exists(project_path)
        
        # reconstruct options
        opts='-t %s' % job_type

        if not job_type.startswith(('main', 'wmf', 'cf', )): 
            raise ValueError('Multi-domain mode only supports the main app (cf.py) but job type: {jt}'.format(jt=job_type))
        else: 
            # options for configuring jobs
            if Job.walltime: opts = "{opts} --walltime {walltime}".format(opts=opts, walltime=Job.walltime)
            if Job.n_cores: opts = "{opts} --cores {n_cores}".format(opts=opts, n_cores=Job.n_cores)
            if Job.queue: opts = "{opts} --queue {q}".format(opts=opts, q=Job.queue)
            if meta: opts = "{opts} --meta {m}".format(opts=opts, m=meta)
            if dry_run: opts = "{opts} --dryrun".format(opts=opts)

            # [note] will have to modify the job chain for the wrapper options to work 
            ###################################################################
            # ... options for the main executable
            if job_type in Job.job_type_main: # ['main', 'cf', ]: 
                if n_factors is not None: opts = "{opts} --n-factors '{n}'".format(opts=opts, n=n_factors)
                if alpha is not None: opts = "{opts} --alpha '{a}'".format(opts=opts, a=alpha)
                if n_runs: opts = "{opts} --runs {n}".format(opts=opts, n=n_runs)
                if n_runs_modelselect: opts = "{opts} --runs-model-select {n}".format(opts=opts, n=n_runs_modelselect)
            ###################################################################
            cmd = '{compiler} {exec} {path} {opts}'.format(compiler=Job.compiler, exec=app, path=project_path, opts=opts)  # note: Job.args should only contain project_path, which is being re-interpreted here
        
        ###################################################################
        # ... exec cf_run.py per domain/dataset
        div("Domain: {dataset} | Cmd: {cmd}".format(dataset=domain, cmd=cmd), symbol='#', border=1)
        result = subprocess.check_output(cmd, shell=True)
        result = result.decode("utf-8")
        print('\n(runMultiDomains) ......\n%s\n<<<Completed Domain %s>>>' % (result, domain))
        ###################################################################
        
        sleep(np.random.uniform(0.5, 1.2, 1)[0])
        acc += 1
    div("Submitted {n} sets of jobs ###".format(n=acc), symbol='=', border=1)

    return

def run(**kargs):
    ### to execute simpler jobs as a test: 
    # System.domains = ['pf1', 'pf2', 'pf3', ]
    ###############################################
    config(**kargs)
    if System.multi_domains: 
        runMultiDomains(**kargs)
    else: 
        runJobs(**kargs)

    return

if __name__ == "__main__":
    run()
    print("\n(cf_run) Done!")






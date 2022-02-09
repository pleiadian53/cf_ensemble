


### system global variables
class System(object): 
    dry_run = False

# refactor: utils_cluster
class Job(object): 
    app = 'cf.py'
    options = '' 
    name = ''
    domain = ''  # derived from project_path, os.path.basename(project_path)
    chain = [app, ]
    meta = ''  # job_id
    path = os.getcwd()

    @staticmethod
    def parse(exec_cmd):   # the main executable (e.g. cf.py --setting 2)
        Job.app, Job.options = ' '.join(exec_cmd.split()[:1]), ' '.join(exec_cmd.split()[1:])

    @staticmethod
    def name_job(app='', opt='', domain='', job_id=''): 
        if not domain: domain = Job.domain 
        if not app: app = Job.app  # see CF_run()
        if not opt: opt = Job.options  # see CF_run()
        if not job_id: job_id = Job.meta

        assert app.find('.') > 0, "Invalid executable? %s" % app
        app = app.split('.')[0]
        if opt: app += ''.join(opt.split())

        job_name = '{exec}-{domain}'.format(exec=app, domain=domain) 
        if job_id: job_name = '{base}-{id}'.format(base=job_name, id=job_id)
        return job_name
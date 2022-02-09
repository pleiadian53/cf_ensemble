import os

class Analysis(object): 
    """

    Memo
    ----
    1. 
    """
    # columns of 'performance_table_threshold_{t}.csv' 
    columns = ['seed', 'predict_label', 'threshold', 'precision', 'recall', 'f_score', 'model'] # 'model' NOT 'method' here

    working_dir = os.getcwd()
    project_path = os.getcwd()
    analysis_path = os.path.join(os.getcwd(), 'analysis')

    models = ['ada' 'enet' 'knn' 'log' 'naive' 'qda' 'rf' 'svm', 'mean'] + ['latent_mean', 'masked_latent_mean', ]
    modelDict = {'log': 'Logistic', 'qda': 'QDA', 'enet': 'ElasticNet', 'svm': 'SVM', 
                   'naive': 'NaiveBayes', 'rf': 'RandomForest', 'ada': 'AdaBoost', 'knn': 'kNN', 
                   'mean': 'mean', 'latent_mean': 'LatentMean', 'latent_mean_masked': 'MaskedLatentMean'}
    orders = ['Logistic', 'ElasticNet', 'SVM', 'QDA', 'NaiveBayes', 'kNN',  'AdaBoost', 'RandomForest', 'mean', ] # ['AdaBoostM1','LogitBoost','NaiveBayes','Logistic','SMO','VotedPerceptron','IBk','PART','J48','RandomForest']

    name_map = replacement = {'user': 'classifier', 'item': 'sample'}  # e.g. replace 'user' by 'classifier' in method naming
    inv_name_map = {v: k for k, v in name_map.items()}

    @staticmethod
    def config(domain='go_terms0', analysis_dn='analysis', create_dir=False):
        if domain: 
            # resolve project path e.g. /Users/<user>/work/data/pf1
            # home_dir = os.path.expanduser('~')
            # working_dir_default = '/'.join([home_dir, 'work/data', ])
            parentdir = os.path.dirname(os.getcwd())
            Analysis.working_dir = os.path.join(parentdir, 'data')  # e.g. /Users/<user>/work/data
            Analysis.project_path = os.path.join(Analysis.working_dir, domain) # e.g. /Users/<user>/work/data/pf1
            if not os.path.exists(Analysis.project_path): 
                if create_dir: 
                    os.mkdir(Analysis.project_path)
                else: 
                    msg = "Invalid project path (which includes domain): {data_path}".format(data_path=Analysis.project_path)
                    raise ValueError(msg)
            Analysis.analysis_path = os.path.join(Analysis.project_path, analysis_dn)  # dn: directory name
            if not os.path.exists(Analysis.analysis_path):
                os.mkdir(Analysis.analysis_path)
        else: 
            # use default 
            pass 
        # print('(verify) analysis_path: {p}'.format(p=Analysis.analysis_path))
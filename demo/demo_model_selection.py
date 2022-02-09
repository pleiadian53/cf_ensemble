import sys, os 
parentdir = os.path.dirname(os.getcwd())
sys.path.insert(0,parentdir) # include parent directory to the module search path
import utils_sys
from utils_sys import div

"""
    An experimental module for model selections with CV.

    

"""

def sysConfig(domain='pf2'):
    import utils_sys, common
    from cf_spec import System, MFEnsemble
    from evaluate import PerformanceMetrics
    
    System.projectPath = utils_sys.getProjectPath(domain=domain, verify_=False)  # default
    System.domain = domain

    # all directories depend on this prefix including data_dir, log_dir, plot_dir
    PerformanceMetrics.set_path(prefix=System.projectPath) 
    MFEnsemble.set_path(prefix=System.projectPath)

    p = common.load_properties(System.projectPath, config_file=System.configFile)  # parse config.txt (instead of weka.properties)
    System.foldCount = int(p['foldCount'])
    System.bagCount = int(p['bagCount']) if 'bagCount' in p else int(p['bags']) 

    return

def read_arff(path, verbose=False): 
    """

    Related 
    -------
    1. liac-arff

        pip install liac-arff

    """
    from scipy.io import arff
    import pandas as pd

    data = arff.loadarff(path)
    df = pd.DataFrame(data[0])

    if verbose: print('(read_arff) dim: {0}\n... {1}\n'.format(df.shape, df.head()))
    return df 

def read_arff2(path, verbose=False): 
    """

    Reference
    ---------
    1. liac-arff:

       pip install liac-arff

       https://pypi.org/project/liac-arff/
    """

    import arff   # pip install liac-arff
    data = arff.load(open(path, 'rb'))

    df = pd.DataFrame(data['data'], columns=data['attributes'])
    if verbose: print('(read_arff) dim: {0}\n... {1}\n'.format(df.shape, df.head()))

    return
   
def test(): 
    ### read arff file 
    path = '/Users/pleiades/Documents/work/data/pf2/pf2.arff'
    df = read_arff(path, verbose=True)

    return


if __name__ == "__main__": 
    sysConfig(domain='pf2')
    test()


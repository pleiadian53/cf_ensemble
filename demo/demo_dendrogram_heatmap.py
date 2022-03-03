
# Local Modules
import sys, os 
# parentdir = os.path.dirname(os.getcwd())
main_mod_dir = os.path.abspath('../../.')
sys.path.insert(0,main_mod_dir) # include parent directory to the module search path
import utils_sys
from utils_sys import div
from utils_plot import saveFig


# Libraries
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

"""

Reference 
---------
   1. Dendrogram with heatmap 

          https://python-graph-gallery.com/404-dendrogram-with-heat-map/

   2. discovering structure in heatmap data

          https://seaborn.pydata.org/examples/structured_heatmap.html

Memo
----
   1. seaborn.clustermap 

        The returned object has a savefig method that should be used if you want to save the figure object without clipping the dendrograms.

        To access the reordered row indices, use: clustergrid.dendrogram_row.reordered_ind

        Column indices, use: clustergrid.dendrogram_col.reordered_ind

"""

# utils_plot
def plot_path(name='heatmap', basedir=None, ext='tif', create_dir=True):
    import os
    # create the desired path to the plot by its name
    if basedir is None: basedir = os.path.join(os.getcwd(), 'plot') 
    if not os.path.exists(basedir) and create_dir:
        print('(plot) Creating plot directory:\n%s\n' % basedir)
        os.mkdir(basedir) 
    return os.path.join(basedir, '%s.%s' % (name, ext))

# Data set
def get_data(**kargs): 
    url = 'https://python-graph-gallery.com/wp-content/uploads/mtcars.csv'
    df = pd.read_csv(url)
    df = df.set_index('model')
    del df.index.name
   
    print(df)
    return df

def demo_clustermap(df): 
    # OK now we determined the distance between 2 individuals. But how to do the clusterisation? Several methods exist.
    # If you have no idea, ward is probably a good start.
    g1 = sns.clustermap(df, metric="euclidean", standard_scale=1, method="single")
    g2 = sns.clustermap(df, metric="euclidean", standard_scale=1, method="ward")

    saveFig(g2, plot_path(name='dendrogram_heatmap-test'), dpi=300)
    return 

def test(): 

    df = get_data()
    demo_clustermap(df)

    return
    

if __name__ == "__main__": 
    test()

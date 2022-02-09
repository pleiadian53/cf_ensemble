import sys, os 
parentdir = os.path.dirname(os.getcwd())
sys.path.insert(0,parentdir) # include parent directory to the module search path
from utils_plot import saveFig

import utils_sys
from utils_sys import div
from evaluate import Metrics, PerformanceMetrics, plot_path

# libraries
import numpy as np

import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
import matplotlib.pyplot as plt

# select plotting style 
plt.style.use('seaborn')  # values: {'seaborn', 'ggplot', }
 
# Make fake dataset
def plot(): 
    height = [3, 12, 5, 18, 45]
    bars = ('A', 'B', 'C', 'D', 'E')
    y_pos = np.arange(len(bars))
     
    # Create horizontal bars
    plt.barh(y_pos, height)
     
    # Create names on the y-axis
    plt.yticks(y_pos, bars)
     
    # Show graphic
    # plt.show()
    saveFig(plt, plot_path(name='demo_h_bar'))

    return


# ================================================================================
#                lasso_stacker  enet_stacker  rf_stacker  gb_stacker
# auc                 0.834182      0.828921    0.835765    0.813202
# fmax                0.709088      0.698084    0.708811    0.690529
# fmax_negative       0.788445      0.788445    0.788445    0.788445
# sensitivity         0.604206      0.589730    0.565329    0.579225
# specificity         0.844092      0.865397    0.863941    0.832637
# ================================================================================
def plot_performance(perf, metric='fmax', methods=[], sort_=True, ascending=False): 
    # perf: PerformanceMetrics object

    plt.clf()

    # performance metric vs methods
    perf_methods = perf.table.loc[metric] if not methods else perf.table.loc[metric][methods]  # a Series
    if sort_: 
        perf_methods = perf_methods.sort_values(ascending=ascending)

    # adjust figure size
    matplotlib.rcParams['figure.figsize'] = (10, 20)

    ax = perf_methods.plot(kind = "barh")
    plt.yticks(fontsize=6)
    plt.title("Performance comparison (metric: %s)" % metric, fontsize=12)

    if file_name == '': file_name = 'coeffs'
    saveFig(plt, plot_path(name=file_name), dpi=500)

    return

def visualizeCoeffs(model, features, file_name='', exception_=False): 
    """
    Given a trained model (fitted with training data), visualize the model coeffs

    """
    assert hasattr(model, 'coef_'), "%s has no coef_!" % model

    try: 
        coef = pd.Series(model.coef_[0], index = features)   # coeff_ is a 2D array
    except: 
        msg = "%s has not coef_!" % model
        if exception_: 
            raise ValueError(msg)
        else: 
            print( msg )

            # [todo] do something else
            return

    print("(visualize_coeffs) Elastic Net picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
    imp_coef = coef.sort_values()

    plt.clf()
    matplotlib.rcParams['figure.figsize'] = (16, 20)

    ax = imp_coef.plot(kind = "barh")
    plt.yticks(fontsize=6)
    plt.title("Coefficients in the Elastic Net Model", fontsize=12)

    if file_name == '': file_name = 'coeffs'
    saveFig(plt, plot_path(name=file_name), dpi=500)

    return

def test(): 

    plot()

    return

if __name__ == "__main__": 
    test()

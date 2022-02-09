import sys, os 
parentdir = os.path.dirname(os.getcwd())
sys.path.insert(0,parentdir) # include parent directory to the module search path
from utils_plot import saveFig

import utils_sys
from utils_sys import div
from evaluate import Metrics, PerformanceMetrics, plot_path

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.

import matplotlib.pyplot as plt

# select plotting style 
plt.style.use('seaborn')  # values: {'seaborn', 'ggplot', }


"""

Reference
---------
   1. Chris Albon: https://chrisalbon.com/python/data_visualization/matplotlib_grouped_bar_plot/

"""


raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
        'pre_score': [4, 24, 31, 2, 3],
        'mid_score': [25, 94, 57, 62, 70],
        'post_score': [5, 43, 23, 23, 51]}
df = pd.DataFrame(raw_data, columns = ['first_name', 'pre_score', 'mid_score', 'post_score'])
print df

# Setting the positions and width for the bars


# ================================================================================
#                lasso_stacker  enet_stacker  rf_stacker  gb_stacker
# auc                 0.834182      0.828921    0.835765    0.813202
# fmax                0.709088      0.698084    0.708811    0.690529
# fmax_negative       0.788445      0.788445    0.788445    0.788445
# sensitivity         0.604206      0.589730    0.565329    0.579225
# specificity         0.844092      0.865397    0.863941    0.832637
# ================================================================================

def plot_performance(self, metrics=['fmax', 'fmax_negative', 'auc', ], methods=[]): 
    
    width = 0.25 
    for i, metric in enumerate(metrics): 

        # n metrics, n groups
        pos = list(range(len(metrics)))  

        # Plotting the bars
        fig, ax = plt.subplots(figsize=(20,10)) 

        # Create a bar with metric data,
        # in position pos,
        plt.bar(pos, 
                # using the row associated with the metric,
                self.table.loc[metric],    # a Series
                # of width
                width, 
                # with alpha 0.5
                alpha=0.5, 
                # with color
                color='#EE3224', 
                # with label the first value in first_name
                label=df['first_name'][0]) 
    
    # [todo]
    return 

#   first_name  pre_score  mid_score  post_score
# 0      Jason          4         25           5
# 1      Molly         24         94          43
# 2       Tina         31         57          23
# 3       Jake          2         62          23
# 4        Amy          3         70          51

def plot(): 
    pos = list(range(len(df['pre_score']))) 
    width = 0.25 
        
    # Plotting the bars
    fig, ax = plt.subplots(figsize=(10,5))

    # Create a bar with pre_score data,
    # in position pos,
    plt.bar(pos, 
            #using df['pre_score'] data,
            df['pre_score'], 
            # of width
            width, 
            # with alpha 0.5
            alpha=0.5, 
            # with color
            color='#EE3224', 
            # with label the first value in first_name
            label=df['first_name'][0]) 

    # Create a bar with mid_score data,
    # in position pos + some width buffer,
    plt.bar([p + width for p in pos], 
            #using df['mid_score'] data,
            df['mid_score'],
            # of width
            width, 
            # with alpha 0.5
            alpha=0.5, 
            # with color
            color='#F78F1E', 
            # with label the second value in first_name
            label=df['first_name'][1]) 

    # Create a bar with post_score data,
    # in position pos + some width buffer,
    plt.bar([p + width*2 for p in pos], 
            #using df['post_score'] data,
            df['post_score'], 
            # of width
            width, 
            # with alpha 0.5
            alpha=0.5, 
            # with color
            color='#FFC222', 
            # with label the third value in first_name
            label=df['first_name'][2]) 

    # Set the y axis label
    ax.set_ylabel('Score')

    # Set the chart's title
    ax.set_title('Test Subject Scores')

    # Set the position of the x ticks
    ax.set_xticks([p + 1.5 * width for p in pos])

    # Set the labels for the x ticks
    ax.set_xticklabels(df['first_name'])

    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos)-width, max(pos)+width*4)
    plt.ylim([0, max(df['pre_score'] + df['mid_score'] + df['post_score'])] )

    # Adding the legend and showing the plot
    plt.legend(['Pre Score', 'Mid Score', 'Post Score'], loc='upper left')
    plt.grid()
    # plt.show()

    saveFig(plt, plot_path(name='demo_grouped_barplot'))

    return

def test(): 

    # barplots in groups 
    plot()

    return

if __name__ == "__main__":
    test() 



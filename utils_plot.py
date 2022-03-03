
# import plotly.plotly as py
# import plotly.graph_objs as go
# ... plotly.plotly is obsolete
# ... `pip install chart_studio`

import os, sys
try:
    from chart_studio import plotly 
except ImportError:
    import subprocess
    # pip internal is to be deprecated 
    # from pip._internal import main as pip
    # pip(['install', 'chart_studio'])

    subprocess.check_call([sys.executable, "-m", "pip", "install", 'chart_studio'])
    from chart_studio import plotly

import plotly as py
import plotly.graph_objs as go

# non-interactive mode
import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
from matplotlib import pyplot as plt

import pandas as pd
from pandas import DataFrame, Series
# import cufflinks as cf  # cufflinks binds plotly to pandas
import numpy as np

# import heatmap
# from sns import heatmap

import random

#######################################################################
#
#
#  References
#  ----------
#  1. cufflinks: https://github.com/santosjorge/cufflinks
#
#
#

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    import matplotlib.colors as colors
    
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def t_colormap(): 
    # import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    arr = np.linspace(0, 50, 100).reshape((10, 10))
    fig, ax = plt.subplots(ncols=2)

    cmap = plt.get_cmap('jet')
    new_cmap = truncate_colormap(cmap, 0.2, 0.8)

    # [test]
    # ax[0].imshow(arr, interpolation='nearest', cmap=cmap)
    # ax[1].imshow(arr, interpolation='nearest', cmap=new_cmap)
    # plt.show()
    
    return

def plot_path(name='test', basedir=None, ext='tif', create_dir=False):
    import os
    # create the desired path to the plot by its name
    if basedir is None: basedir = os.path.join(os.getcwd(), 'plot')
    if not os.path.exists(basedir) and create_dir:
        print('(plot) Creating plot directory:\n%s\n' % basedir)
        os.mkdir(basedir) 
    return os.path.join(basedir, '%s.%s' % (name, ext))

def saveFig(plt, fpath, ext='tif', dpi=500, message='', verbose=True):
    """
    fpath: 
       name of output file
       full path to the output file

    Memo
    ----
    1. supported graphing format: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff 

    """
    import os
    outputdir, fname = os.path.dirname(fpath), os.path.basename(fpath) 

    # [todo] abstraction
    supported_formats = ['eps', 'jpeg', 'jpg', 'pdf', 'pgf', 'png', 'ps', 'raw', 'rgba', 'svg', 'svgz', 'tif', 'tiff', ]  

    # supported graphing format: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff.
    if not outputdir: outputdir = os.getcwd() # sys_config.read('DataExpRoot') # ./bulk_training/data-learner)
    assert os.path.exists(outputdir), "Invalid output path: %s" % outputdir
    ext_plot = ext  # eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff.
    if not fname: fname = 'generic-test.%s' % ext_plot
    fbase, fext = os.path.splitext(fname)
    assert fext[1:] in supported_formats, "Unsupported graphic format: %s" % fname

    fpath = os.path.join(outputdir, fname)

    if verbose: print('(saveFig) Saving plot to:\n%s\n... description: %s' % (fpath, 'n/a' if not message else message))
    
    # pylab leaves a generous, often undesirable, whitespace around the image. Remove it by setting bbox_inches to tight
    plt.savefig(fpath, bbox_inches='tight', dpi=dpi)   
    return

def t_overlaid_historgram(**kargs): 
    """
    An example of plotting overlaid histogram (with two metrics: ['nuniq', 'nrow', ]). 

        1. bar plot 
       #F2F0CE   : yellow urine 
       #C8E3EF   : blue antibio
       #D0EFDA   : green microbio
       #EED0D7   : red blood

       #BFC3E3   : light blue/violet, combined at level 1

    2. scatter map 
       #5E88DB   : blue curve, antibio

       #283BE5   : dark blue, combined at level 1

    Memo
    ----
    1. Color picker: http://www.w3schools.com/colors/colors_picker.asp

    """
    import pandas as pd

    rootdir = '.'

    ifiles = []
    active_models = ['microbio', 'antibio', 'blood', 'urine',  ]  
    active_models_std= ['Microbiology', 'Antibiotic', 'Blood Test', 'Urine Test', ]
    sortbyattr = 'nuniq' # number of unique patients
    metrics = ['nuniq', 'nrow', ]  # overlaid histogram

    color_markers = {'microbio': ['#00b300', '#00cc99'], 
                     'antibio': ['#0066ff', '#66ccff'], 
                     'urine': ['#ff9900', '#ffcc80'],     
                     'blood': ['#ff0066', '#ff99cc']}  
                     # ['#C8E3EF', '#F2F0CE']
    f_dtype = {'code': str}

    # 'tset_stats_antibio-sort_by_nuniq.csv'
    for m in active_models: 
        ifile = 'tset_stats_%s-sort_by_%s.csv' % (m, sortbyattr)
        fpath = os.path.join(rootdir, ifile)
        assert os.path.exists(fpath), 'invalid path: %s' % fpath
        ifiles.append(fpath)  # 'performance_test-cross.pkl'

    nrows_total = nuniq_total = 0
    for i, fpath in enumerate(ifiles): 
        # ifile = 'performance_test-%s.pkl' % mode

        params = {}
        params['traces'] = []

        assert os.path.exists(fpath)
        
        df = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True, dtype=f_dtype)
        df.sort_values(sortbyattr, ascending=True, inplace=True) 

        inrow = df['nrow'].sum()
        nrows_total += inrow
        print('info> model: %s, nrow: %d' % (active_models[i], inrow))
        nuniq_total += df['nuniq'].sum()

        x = df['code'].values
        # nuniq = df['nuniq'].values 
        # nrow = df['nrow'].values

        print('info> code (size: %d):\n%s' % (len(x), x))

        # trace_params = {}
        for j, metric in enumerate(metrics): 
            trace = {}
            trace['x'] = x 
            trace['y'] = df[metric].values 

            trace['color_marker'] = color_markers[active_models[i]][j]

            print('info> metric=%s:\n%s' % (metric, trace['y']))
            params['traces'].append(trace)
            
            # # [test]
            # trace_params['x'] = x 
            # trace_params['y'] = df[metric].values 
            # trace_params['color_marker'] = color_markers[active_models[i]][j]

        params['model'] = active_models[i]

        # if text is not None: params['text'] = text

        params['color_marker'] = kargs.get('color_marker', None)
        params['color_err'] = kargs.get('color_err', None)

        # title 
        params['title_x'] = kargs.get('title_x', 'Training Set Profile for the %s Model' % active_models_std[i])
        params['title_y'] = kargs.get('title_y', 'Number of Unique Patients vs Training Set Size')
        params['plot_type'] = kargs.get('plot_type', 'bar')

        # prefix, level, identifier, suffix, meta
        meta = kargs.get('meta', None)
        fname = '%s-data-hist.pdf' % params['model']
        params['opath'] = os.path.join(rootdir, fname)


        params['axis_range'] = [10, 20000] # axis_range
        # params['plot_type'] = 'scatter'

        # params['trace_params'] = trace_params
        t_plotly(params)

    print('info> total nrow: %d, total n_patients: %d' % (nrows_total, nuniq_total))

    return

def t_plotly(params): # performance evaluation 
    def select_plot_type(): # params: plot_type, values: 'bar', 'scatter', 
        print('plotly> selecting plot type ...')
        plotType = params.get('plot_type', 'bar')
        if plotType.startswith('b'): 
            plotFunc = go.Bar
            # plotType = 'bar'
        elif plotType.startswith('sc'):
            print('plotly> selected scatter plot.')
            plotFunc = go.Scatter 
            # plotType = 'scatter'
        return (plotType, plotFunc)

    # (*) To communicate with Plotly's server, sign in with credentials file
    import plotly.plotly as py
    # (*) Useful Python/Plotly tools
    import plotly.tools as tls
    import plotly.graph_objs as go
    # (*) Graph objects to piece together plots
    # from plotly.graph_objs import *

    print('plotly> setting color scheme ...')
    color_marker = params.get('color_marker', '#E3BA22')
    color_err = params.get('color_err', None)

    plotType, plotFunc = select_plot_type()

    # Make a Bar trace object
    # params: x, y, array, arraymius, color_marker, color_err, title_x, title_y, axis_range
    traces = params.get('traces', [])
    data = []
    for trace_params in traces: 
        color_marker_eff = trace_params.get('color_marker', color_marker)
        if color_marker_eff and color_err: 
            # text, opacity
            trace = plotFunc(
                x=trace_params['x'],  # a list of string as x-coords
                y=trace_params['y'],   # 1d array of numbers as y-coords
                marker=go.Marker(color=color_marker_eff),  # set bar color (hex color model)
                error_y=go.ErrorY(
                    type='data',     # or 'percent', 'sqrt', 'constant'
                    symmetric=False,
                    array=trace_params.get('array', None),
                    arrayminus=trace_params.get('arrayminus', None), 
                    color=color_err,  # set error bar color
                    thickness=0.6
               )
            )
        else: # default color (blue bars and black error bars)
            trace = plotFunc(
                x=trace_params['x'],  # a list of string as x-coords
                y=trace_params['y'],   # 1d array of numbers as y-coords
                marker=go.Marker(color=color_marker_eff), 
                error_y=go.ErrorY(
                    type='data',     # or 'percent', 'sqrt', 'constant'
                    symmetric=False,
                    array=trace_params.get('array', None),
                    arrayminus=trace_params.get('arrayminus', None), 
                )
            )

        # Make Data object
        data.append(trace)
        # data = [trace1, ] # go.Data([trace1])

    titleX = params.get('title_x', None)  # title_x, model
    if titleX is None: titleX = "Class Labels"

    titleY = params.get('title_y', None)
    if titleY is None: titleY = 'Area under the Curve'

    axis_range = params.get('axis_range', None)

    # Make Layout object
    if plotType.startswith('b'): 
        print('info> configuring layout for bar plot ...')
        layout = go.Layout(
            title=titleX,       # set plot title
            showlegend=False,  # remove legend
            font=dict(size=11), # family='Courier New, monospace', color='#7f7f7f'

            xaxis = go.XAxis(
                type = 'category',
                showticklabels=True,
                tickangle=45,
                # tickfont=dict(
                #     family='Old Standard TT, serif',
                #     # size=14,
                #     color='black'
                # )
            ),

            yaxis= go.YAxis(
                title=titleY, # y-axis title
                range=axis_range,               # set range
                zeroline=False,                  # remove thick line at y=0
                gridcolor='white'                # set grid color to white
            ),
           paper_bgcolor='rgb(233,233,233)',  # set paper (outside plot) 
           plot_bgcolor='rgb(233,233,233)',   #   and plot color to grey
       )
    else: # automatic range assignment
        layout = go.Layout(
            title=titleX,       # set plot title
            showlegend=False,  # remove legend
            font=dict(size=11), # family='Courier New, monospace', color='#7f7f7f'

            xaxis = dict(
                type = 'category',
                showticklabels=True,
                tickangle=45,
                # tickfont=dict(
                #     family='Old Standard TT, serif',
                #     # size=14,
                #     color='black'
                # )
            ),

            yaxis= go.YAxis(
                title=titleY, # y-axis title
                range=axis_range,               # set range
                zeroline=False,                  # remove thick line at y=0
                gridcolor='white'                # set grid color to white
            ),

           paper_bgcolor='rgb(233,233,233)',  # set paper (outside plot) 
           plot_bgcolor='rgb(233,233,233)',   #   and plot color to grey
       ) 


    # Make Figure object
    fig = go.Figure(data=data, layout=layout)

    # save file
    fpath = params['opath']
    base, fname = os.path.dirname(fpath), os.path.basename(fpath)
    assert os.path.exists(base)

    # (@) Send to Plotly and show in notebook
    # py.iplot(fig, filename=fname)
    # (@) Send to broswer 
    plot_url = py.plot(fig, filename=fname)
    py.image.save_as({'data': data}, fpath)

    return (fig, data)  

def t_plot_multiple_hist(**kargs): 
    """

    Memo
    ----
    1. works on the notebook, but how to save plot from here? 

    """

    # import plotly.plotly as py
    # import plotly.offline as offline
    # import matplotlib.pyplot as plt 
    import cufflinks as cf  # cufflinks binds plotly to pandas

    # fig = plt.figure(figsize=(8, 8))
    # fig = plt.figure()
    plt.clf()

    cf.set_config_file(offline=False, world_readable=True, theme='pearl')
    
    adict = {'a': np.random.randn(1000) + 1,
                       'b': np.random.randn(1000),
                       'c': np.random.randn(1000) - 1, 
                       'd': np.random.randn(1000) + 20}

    df = pd.DataFrame(adict)
    df.head(2)

    fpath = 'plot/mulitple-histograms.tif'
    # plt.figure()

    # this outputs picture on plotly's server side under 'plot' folder
    # [note] then one can edit it interactively on the website
    df.iplot(kind='histogram', subplots=True, shape=(len(adict), 1), filename=fpath)  # ok. 
    
    # py.image.save_as({'data': adict}, fpath)
    # plt.savefig(fpath)  # not ok 

    return

def plot_multiple_hist(df, cols=None, fpath=None): 
    if fpath is None: 
        basedir = os.getcwd()
        fname = 'multi-histograms.png'
        fpath = os.path.join(basedir, fname)
    else: 
        basedir = os.path.dirname(fpath) 
        assert os.path.exists(basedir), "Invalid (base) path: %s" % basedir

    plt.clf()
    df.hist(color='k', alpha=0.5, bins=50)  
    plt.savefig(fpath)  

    return

def plot_multiple_hist_plotly(df, cols=None, outputdir=None, fname=None): 
    import cufflinks as cf 

    # tInternalRendering = True  # panda's own version of graphic rendering

    if outputdir is None: outputdir = 'plot'  # plotly's local folder that keeps graphics
    cf.set_config_file(offline=True, world_readable=True, theme='pearl')

    if fname is None: fname = 'multi-histograms.png'
    fpath = '%s/%s' % (outputdir, fname)
    # else: 
    #     basedir = os.path.dirname(fpath) 
    #     assert os.path.exists(basedir), "Invalid (base) path: %s" % basedir

    ### plotly
    # [note] adjust the shape
    if cols is None: # plot all columns 
        ncol = df.shape[1]

        if ncol % 2 == 0 and ncol >= 6: 
            df.iplot(kind='histogram', subplots=True, shape=(df.shape[1]/2, 2), filename=fpath)
        else: 
            df.iplot(kind='histogram', subplots=True, shape=(df.shape[1], 1), filename=fpath)
    else: 
        assert len(set(cols)-set(df.columns.values))==0
        df2 = df[cols]
        df2.iplot(kind='histogram', subplots=True, shape=(df2.shape[1], 1), filename=fpath)

    # local, pandas's own rendering of multiple histograms 
    # if tInternalRendering and os.path.exists(): 
    #     plt.clf()
    #     df.hist(color='k', alpha=0.5, bins=50)  
    #     plt.savefig(fpath)  # ok.
 
    return

def t_plot_dataframe(**kargs):
    """

    References
    ----------
    1. pandas visualization 
       https://pandas.pydata.org/pandas-docs/stable/visualization.html


    """
    import matplotlib.pyplot as plt 
    import matplotlib

    matplotlib.style.use('ggplot')
    basedir = os.path.join(os.getcwd(),'test')

    plt.clf()
    ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
    ts = ts.cumsum() # return cumulative sum over requested axis.
    ts.plot() 

    plt.savefig(os.path.join(basedir, 'time_series-1.png'))  # ok. 

    df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=list('ABCD'))
    df = df.cumsum()
    plt.figure(); df.plot();

    # bar plot 
    df2 = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])
    df2.plot.bar();
    df2.plot.bar(stacked=True); # stacked
    df2.plot.barh(stacked=True); # stacked + horizontal 


    # plot one column vs another 
    plt.clf()
    df3 = pd.DataFrame(np.random.randn(1000, 2), columns=['B', 'C']).cumsum()
    df3['A'] = pd.Series(list(range(len(df))))
    df3.plot(x='A', y='B') 
    plt.savefig(os.path.join(basedir, 'time_series-2.png'))  # ok.


    # histogram 
    plt.clf()
    df4 = pd.DataFrame({'a': np.random.randn(1000) + 1, 'b': np.random.randn(1000),
                        'c': np.random.randn(1000) - 1}, columns=['a', 'b', 'c'])
    df4.plot.hist(alpha=0.5)
    plt.savefig(os.path.join(basedir, 'hist-overlapped-demo1.png'))  # ok.

    df4.plot.hist(stacked=True, bins=20) # stacked

    # horizontal + cumulative
    df4['a'].plot.hist(orientation='horizontal', cumulative=True)


    # multiple subplots 
    plt.clf()
    df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=list('ABCD'))
    df = df.cumsum()    
    df.diff().hist(alpha=0.5, bins=50) # color='k' # DataFrame.diff(): 1st discrete difference of object
    plt.savefig(os.path.join(basedir, 'multiple-hist-demo1.png'))  # ok.

    return

def t_heatmap_plotly(): 
    # import plotly.plotly as py
    # import plotly.graph_objs as go

    trace = go.Heatmap(z=[[1, 20, 30, 50, 1], [20, 1, 60, 80, 30], [30, 60, 1, -10, 20]],
                       x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                       y=['Morning', 'Afternoon', 'Evening'])
    data=[trace]
    py.iplot(data, filename='labelled-heatmap')

    return

def t_heatmap(): 
    import heatmap
    import random

    pts = []
    for x in range(400):
        pts.append((random.random(), random.random() ))

    hm = heatmap.Heatmap()
    img = hm.heatmap(pts)
    img.save("classic.png")

    return

def t_histogram(**kargs): 

    ### plot multiple histograms

    # t_plot_multiple_hist(**kargs)

    ### 
    # t_plot_dataframe(**kargs)

    ### Bar plot 
    y = [0.80, 0.80, 0.89, 0.78, 0.73, 0.71, 0.79, 0.82, 0.98, 0.86, 0.82]
    x = ['G1-control', 'G1A1-control',  'Stage 1', 'Stage 2', 'Stage 3a', 'Stage 3b', 
         'Stage 4', 'Stage 5', 'ESRD after transplant', 'ESRD on dialysis', 'Unknown']

    # a wrapper for t_plotly()
    makeBarPlot(x, y, title_x="CKD stages", title_y="Area under the curve (AUC)", color_marker='#C8E3EF', cohort='CKD')

    return

def t_colors(**kargs):
    from itertools import cycle, islice
    import pandas, numpy as np  # I find np.random.randint to be better

    # 
    factor = 'color'
    basedir = os.getcwd() # or sys_config.read('DataExpRoot')  # 'seqmaker/data/CKD/plot'

    # Make the data
    x = [{i:np.random.randint(1,5)} for i in range(10)]
    df = pandas.DataFrame(x)

    # Make a list by cycling through the colors you care about
    # to match the length of your data.
    my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(df)))

    # Specify this list of colors as the `color` option to `plot`.
    df.plot(kind='bar', stacked=True, color=my_colors) 
    plt.savefig(os.path.join(basedir, 'test-%s.tif' % factor ))  #

    return

def runWorkflow():
    plt.style.use('ggplot')  # 'seaborn'
    from itertools import cycle, islice

    adict = {}

    ### Small CKD cohort (n=2833), effective N=2003
    adict['stage'] = ['I', 'II', 'III', 'IV', 'V', 'Ctrl']
    adict['sample_size'] = [89, 630, 422, 84, 778, 830]
   
    df = DataFrame(adict, columns=['stage', 'sample_size'])
    # df.set_index('stage')

    cohort = 'CKD'
    basedir = os.path.join(os.getcwd(), 'data/%s/plot' % cohort)  # or sys_config.read('DataExpRoot')  # 'seqmaker/data/CKD/plot'
    
    ## histogram
    # plt.clf()
    # df.plot.hist(alpha=0.5)
    # plt.savefig(os.path.join(basedir, 'sample_size-hist-%s.tif' % cohort))  # ok.

    ## bar plot
    plt.clf()
    # make a list by cycling through the colors you care about to match the length of your data.
    # my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(df))) # df.shape[0])
    my_colors = [(x/24.0,  x/48.0, 0.05) for x in range(len(df))]

    # Set color transparency (0: transparent; 1: solid)
    a = 0.7
    # my_colors = [plt.cm.Paired(np.arange(len(df)))]
    # my_colors = ['b', 'g', 'r', 'c', 'm', 'y', ] # 'k', 'w'

    ttl = 'Stage versus Sample Size'
    ax = df['sample_size'].plot(kind='bar', title=ttl, 
             legend=False, fontsize=12, edgecolor='w') # figsize=(20, 20), s
    
    ax.set_xlabel("Stage", fontsize=12)
    ax.set_ylabel("Sample Size", fontsize=12)
    ax.set_xticklabels(df['stage'], rotation=0)

    df['sample_size'].plot(color=my_colors, alpha=a)
    plt.savefig(os.path.join(basedir, 'sample_size-bar-%s.tif' % cohort))  #

    # makeBarPlot(x=adict['stage'], y=[89, 630, 422, 84, 778], title_x="CKD stages", title_y="Sample sizes", color_marker='#C8E3EF', cohort='CKD')
      
    ## bar plot version 2
    # Create a colormap
    # fig = plt.figure(figsize=(16,14))

    # fig.set_xlabel("Stage", fontsize=12)
    # fig.set_ylabel("Sample Size", fontsize=12)
    # fig.set_xticklabels(df['stage'], rotation=0)

    # # # Set color transparency (0: transparent; 1: solid)
    # a = 0.7
    # my_colors = [(x/24.0,  x/48.0, 0.05) for x in range(len(df))]

    # # # Plot the 'population' column as horizontal bar plot
    # ttl = 'Stage versus Sample Size'
    # df['sample_size'].plot(kind='bar', title=ttl, legend=True, fontsize=12, color=my_colors, edgecolor='w')
    # plt.savefig(os.path.join(basedir, 'sample_size-bar-%s.tif' % cohort))  #

    return

def test(**kargs): 

    # t_histogram(**kargs)

    # color scheme 
    t_colors()

    return

if __name__ == "__main__": 
    # test()

    runWorkflow()

# if __name__ == "__main__":    
#     pts = []
#     for x in range(400):
#         pts.append((random.random(), random.random() ))

#     print "Processing %d points..." % len(pts)

#     hm = heatmap.Heatmap()
#     img = hm.heatmap(pts)
#     img.save("classic.png")


import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns
import scipy as sp
import pandas as pd
import numpy as np
import sys
import matplotlib as mpl


sys.path.append('/home/kairos/software/')
import toolbag.read_write, toolbag.signal, toolbag.utils

def paired_plot_altair(df, x, y, groupby, tooltip=None, boxplot=False, ylim=[], title=''):
    if tooltip is None:
        tooltip = groupby

    lines = alt.Chart(df).mark_line(size=1).encode(
        x=x,
        y=y,
        color=alt.Color(groupby, legend=None))

    circles = alt.Chart(df).mark_circle().encode(
        x=x,
        y=alt.Y(y, scale=alt.Scale(domain=ylim)),
        color=alt.Color(groupby, legend=None),
        tooltip=tooltip)
    
    if boxplot:
        box = alt.Chart(df).mark_boxplot(size=30, color='gray').encode(
            x=x,
            y=y)
        graph = (box + lines + circles)
    else:
        graph = (lines + circles)
    
    # pairs_ixs = toolbag.utils.find_pairs(df, ['subj', 'stim_ch'], 'cond', only_first=True)
    pairs_ixs = toolbag.utils.find_pairs(df, groupby, x, only_first=True)
    y1 = df.loc[np.array(pairs_ixs)[:,0], y].values
    y2 = df.loc[np.array(pairs_ixs)[:,1], y].values
    T, p = sp.stats.ttest_rel(y2, y1)
    s = f'T={T:.2f}, p={p:.0e}'
    title = s if title=='' else title + '\n' + s
    
    return graph.properties(
        height=200,
        width=200,
        title=title)


def scatterplot(df, col_x, col_y, condition_col=None, condition_order=None):
    if condition_order is None:
        conditions = df[condition_col].unique()
    else:
        conditions = condition_order
    
    sns.scatterplot(x=col_x, y=col_y, data=df, hue=condition_col, hue_order=conditions, marker='o')
    
    for cond in conditions:
        xs = df.loc[df.condition==cond, col_x]
        ys = df.loc[df.condition==cond, col_y]

        mean_x = np.mean(xs)
        mean_y = np.mean(ys)
        error_x = np.array([[np.percentile(xs, 5) - mean_x, np.percentile(xs, 95) - mean_x]])
        error_y = np.array([[np.percentile(ys, 5) - mean_y, np.percentile(ys, 95) - mean_y]])

        plt.errorbar(mean_x, mean_y, xerr=error_x, yerr=error_y, fmt='o')
    
def paired_plot(df, y):
    plt.figure(figsize=(15,4))
    conditions = ['Prop1', 'Keta1', 'Sevo1']
    for j, protocol in enumerate(['Propofol', 'Ketamine', 'Sevoflurane']):
        plt.subplot(1,3,j+1)
        df2 = df[np.logical_and(df['protocol']==protocol, df['condition'].isin(conditions + ['Wake']))].sort_values('condition',ascending=False).reset_index()

        sns.swarmplot(y=y, x='condition', data=df2)
        y1 = []
        y2 = []
        for subject in df2['animal'].unique():
            y_left = df2.loc[np.logical_and(df2['animal'] == subject, df2['condition'] == 'Wake'), y].values
            y_right = df2.loc[np.logical_and(df2['animal'] == subject, df2['condition'] == conditions[j]), y].values                
            if len(y_left)==1 and len(y_right)==1:
                y1.append(y_left[0])
                y2.append(y_right[0])
                plt.plot([0,1], [y_left, y_right], color='grey')
            else:
                print('Warning ({}, {}): {}, {}'.format(subject, conditions[j], y_left, y_right))

        T, p = sp.stats.ttest_rel(y2, y1)

        plt.title(f'T={T:2f}, p={p:.5f}')

    plt.suptitle(y)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.savefig(project_dir/'figs/paired_condition_contrast{}.eps'.format(y))

def paired_plot(df, y, by, compare_col, compare_order=None):    
    
    
    left = 'W'
    right = 'NREM'
    
    y1 = []
    y2 = []
        
    for group, dff in df.groupby(by):
        y_left = dff[dff[compare_col]==left][y].values
        y_right = dff[dff[compare_col]==right][y].values
        if len(y_left)>0 and len(y_right)>0:
            y1.append(y_left[0])
            y2.append(y_right[0])
            plt.plot([0,1], [y_left[0], y_right[0]], 'o', )
            plt.plot([0,1], [y_left[0], y_right[0]], '-', color='grey', linewidth=0.3)
            
def scatterplot(df, col_x, col_y, condition_col=None, condition_order=None):
    if condition_order is None:
        conditions = df[condition_col].unique()
    else:
        conditions = condition_order

    sns.scatterplot(x=col_x, y=col_y, data=df, hue=condition_col, hue_order=conditions, marker='o')
    
    
    for cond in conditions:
        xs = df.loc[df.condition==cond, col_x]
        ys = df.loc[df.condition==cond, col_y]

        mean_x = np.mean(xs)
        mean_y = np.mean(ys)
        error_x = np.array([[np.percentile(xs, 5) - mean_x, np.percentile(xs, 95) - mean_x]])
        error_y = np.array([[np.percentile(ys, 5) - mean_y, np.percentile(ys, 95) - mean_y]])

        plt.errorbar(mean_x, mean_y, xerr=error_x, yerr=error_y, fmt='o')


def plot_tf(power, times, freqs, mask=None, logscale=False, lims=None, symmetric=False):
    '''Plot time-frequency plot.
    Parameters
    ----------
    power : 
    
    '''
    

#     norm = intra_tools.MidpointNormalize(np.min(power), np.max(power), 0)
    fig, ax = plt.subplots(figsize=(12,4))
    
    if lims is None:
        if symmetric:
            vmax = np.nanmax(np.abs(power))
            vmin = -vmax
        else:
            vmax = np.nanmax(power)
            vmin = np.nanmin(power)
    else:
        vmin, vmax = lims
    
    norm = toolbag.utils.MidpointNormalize(vmin, vmax, 0)
        
    cmap = 'RdBu_r'
    
    if mask is None:
        plt.pcolormesh(times, freqs, power, cmap=cmap, norm=norm)
    else:
        plt.pcolormesh(times, freqs, power, cmap='gray', norm=norm)
        plt.pcolormesh(times, freqs, np.ma.masked_array(power, mask), cmap=cmap, norm=norm)
    
    if logscale:
        yticks = np.ceil(np.logspace(np.log10(freqs[0]), np.log10(freqs[-1]), 8)).astype(int)
        ax.set_yscale('log')
        ax.set_yticks(yticks)
    #     ax.set_yticks([1,5,10,25,50,100,200])
        ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    plt.xlabel('Time (ms)'); plt.ylabel('Frequency (Hz)')
    plt.colorbar()
    return fig, ax
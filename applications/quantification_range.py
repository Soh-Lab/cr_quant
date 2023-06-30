import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cr_utils.plotting import save_figure
import matplotlib.colors as mcolors
plt.rcParams.update({'font.size': 22, 'font.family': 'serif',
                     'xtick.labelsize': 15, 'ytick.labelsize': 15})

def process_df_range(df):
    'Update the data file to have mM ranges'
    # Make sure no negative targets:
    df['kyn_M_lower'][df.kyn_M_lower < 0] = 1E-13
    df['kyn_M_upper'][df.kyn_M_upper < 0] = 1E-13
    df['xa_M_lower'][df.xa_M_lower < 0] = 1E-13
    df['xa_M_upper'][df.xa_M_upper < 0] = 1E-13

    df['sample_num'] = ['s' + str(ind) for ind in df.index.to_numpy()]

    # Check if true target conc is within bounds
    df['in_kyn_bounds'] = (df.kyn_M_lower < df.kyn_M) & (df.kyn_M_upper > df.kyn_M)
    df['in_xa_bounds'] = (df.xa_M_lower < df.xa_M) & (df.xa_M_upper > df.xa_M)

    # Get how wide the interval is
    df['size_kyn'] = df.kyn_M_upper - df.kyn_M_lower
    df['size_xa'] = df.xa_M_upper - df.xa_M_lower

    df['size_kyn_log'] = np.log10(df.kyn_M_upper / df.kyn_M_lower)
    df['size_xa_log'] = np.log10(df.xa_M_upper / df.xa_M_lower)

    df['xa_M_log'] = np.log10(df.xa_M)

    df['kyn_M_log'] = np.log10(df.kyn_M)
    df['kyn_mM_lower'] = df.kyn_M_lower * 10 ** 3
    df['kyn_mM_upper'] = df.kyn_M_upper * 10**3
    df['kyn_mM'] = df.kyn_M * 10**3

    df['xa_mM_lower'] = df.xa_M_lower * 10 ** 3
    df['xa_mM_upper'] = df.xa_M_upper * 10**3
    df['xa_mM'] = df.xa_M * 10 ** 3
    df['size_kyn_mM'] = df.size_kyn * 10 ** 3
    df['size_xa_mM'] = df.size_kyn * 10 ** 3
    return df

def plot_bar_CIs(df, x, height, bottom, xlabel, ylabel,plt_log=False,
                 plt_true=True, color='tab:blue', true_val= 0.001,
                 hatch=None, flip_axis=False):
    if not flip_axis:
        plt.bar(data=df, x=x, height=height, bottom=bottom, width=0.15,
                align='edge', log=plt_log, color=color, alpha=0.5, hatch=hatch)
        if plt_true: plt.axhline(y=true_val, color='g', linestyle='-.')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)

    else:
        plt.barh(data=df, y=x, width=height, left=bottom, height=0.15,
                align='edge', log=plt_log, color=color, alpha=0.5, hatch=hatch)
        if plt_true: plt.axvline(x=true_val, color='g', linestyle='-.')
        plt.ylabel(xlabel)
        plt.xlabel(ylabel)

    plt.tight_layout()

def plot_CIs_two_models(df, df_lang, plt_true=False, plt_log=False, lim=None,
             x='xa_M_log', height='size_kyn', bottom='kyn_M_lower',
             xlabel='log(xa)', ylabel='kyn (M)'):
    plot_bar_CIs(df=df, x=x, height=height, bottom=bottom,
                 xlabel=xlabel, ylabel=ylabel, color='tab:blue',
                 hatch=None, plt_true=plt_true, plt_log=plt_log)

    plot_bar_CIs(df=df_lang,  x=x, height=height, bottom=bottom,
                 xlabel=xlabel, ylabel=ylabel, color='tab:orange',  hatch=None,
                 plt_true=plt_true, plt_log=plt_log)
    if lim: plt.ylim(lim)
    plt.tight_layout()
    save_figure('range')

if __name__ == '__main__':
    # load data
    # CR Model
    df = pd.read_csv('CR_13_sample_bounds.csv')
    df = process_df_range(df)
    df_conc_range = df.iloc[2:7]

    # Langmuir Model
    df_lang = pd.read_csv('CR_13_sample_bounds_langmuir.csv')
    df_lang = process_df_range(df_lang)
    df_conc_range_lang = df_lang.iloc[2:7]

    # Plot kyn quant for concentration range of increasing xa
    plt.figure(figsize=[6, 10])
    plot_CIs_two_models(df_conc_range, df_conc_range_lang, plt_true=True)
    plt.show()

    # Plot xa quant for concentration range of increasing xa
    plt.figure()
    plt.plot(df_conc_range['xa_M_log'], df_conc_range['xa_M'], color='g', linestyle='-.')
    plot_CIs_two_models(df_conc_range, df_conc_range_lang, plt_true=False, plt_log=False,
             x='xa_M_log', height='size_xa', bottom='xa_M_lower',
             xlabel='log(xa)', ylabel='xa (M)', lim=[10 ** -6, 10 ** -3.5]
             )
    save_figure('xa_quant_kyn_1mM')
    plt.show()

    # Plot extreme indices
    extreme_indices = [0, 1, 7, 8, 9, 10]
    cur_df = df.iloc[extreme_indices]
    # plt.scatter(x=cur_df.kyn_M, y=cur_df.xa_M_log, color='red', marker='x')
    plot_CIs_two_models(cur_df, df_lang.iloc[extreme_indices], plt_log=True, lim=[10 ** -5, 10 ** -2])
    plt.show()
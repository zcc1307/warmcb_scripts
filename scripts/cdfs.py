import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import pylab
from alg_const import alg_str, alg_color_style, alg_index, make_header
import numpy as np

def order_legends(indices):
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles, indices = zip(*sorted(zip(labels, handles, indices), key=lambda t: t[2]))
    ax.legend(handles, labels)

def save_legend(dir, indices):
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    labels, handles, indices = zip(*sorted(zip(labels, handles, indices), key=lambda t: t[2]))
    #figlegend = pylab.figure(figsize=(26,1))
    #figlegend.legend(handles, labels, 'center', fontsize=26, ncol=8)
    figlegend = pylab.figure(figsize=(17,1.5))
    figlegend.legend(handles, labels, 'center', fontsize=26, ncol=3)
    #figlegend.tight_layout(pad=0)
    figlegend.savefig(dir+'legend.pdf')
    plt.close()

#def problem_str(name_problem):
#    return name_problem[0] + '_' + name_problem[1]

def plot_cdf(alg_name, errs, iw):
    col, sty = alg_color_style(alg_name)

    idx = np.argsort(errs)
    num_errs = len(errs)
    sorted_errs = [errs[idx[i]] for i in range(num_errs)]
    sorted_iw = [iw[idx[i]] for i in range(num_errs)]

    plt.step(sorted_errs, np.cumsum(sorted_iw), label=alg_str(alg_name), color=col, linestyle=sty, linewidth=2.0)

def plot_all_cdfs(alg_results, dir, header, iw=None):
    print('printing cdfs..')
    indices = []
    pylab.figure(figsize=(8,6))

    if iw is None:
        num_errs = len(list(alg_results.values())[0])
        iw = [ 1.0 / num_errs for _ in range(num_errs) ]

    for alg_name, errs in alg_results.items():
        indices.append(alg_index(alg_name))
        plot_cdf(alg_name, errs, iw)

    plt.ylim(0,1)
    plt.grid(True, linestyle='--')
    #params={'legend.fontsize':26,
    #'axes.labelsize': 24, 'axes.titlesize':26, 'xtick.labelsize':20,
    #'ytick.labelsize':20 }
    #plt.rcParams.update(params)
    plt.xlabel('Normalized error',fontsize=34)
    plt.ylabel('Cumulative frequency', fontsize=34)
    plt.title(header, fontsize=20)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.tight_layout(pad=0)

    ax = plt.gca()
    order_legends(indices)
    ax.legend_.set_zorder(-1)
    plt.savefig(dir+'cdf.pdf')
    ax.legend_.remove()
    plt.savefig(dir+'cdf_nolegend.pdf')
    plt.title('')
    plt.xlabel('')
    plt.ylabel('')
    plt.savefig(dir+'cdf_notitle.pdf')
    save_legend(dir, indices)
    plt.close()

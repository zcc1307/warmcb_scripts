import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab
import os
import glob
import pandas as pd
import scipy.stats as stats
from itertools import compress
from math import sqrt
import argparse
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties
from collections import Counter
import random
import math
from alg_const import noise_type_str, alg_str, alg_str_compatible, alg_color_style, alg_index
from run_vw_commands import param_to_str

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)

class model:
    def __init__(self):
        pass

def sum_files(result_path):
    prevdir = os.getcwd()
    os.chdir(result_path)
    dss = sorted(glob.glob('*.sum'))
    os.chdir(prevdir)
    return dss

def parse_sum_file(sum_filename):
    f = open(sum_filename, 'r')
    table = pd.read_table(f, sep='\s+',lineterminator='\n',error_bad_lines=False)

    return table

def get_z_scores(errors_1, errors_2, sizes):
    z_scores = []
    for i in range(len(errors_1)):
        #print i
        z_scores.append( z_score(errors_1[i], errors_2[i], sizes[i]) )
    return z_scores

def z_score(err_1, err_2, size):
    if (abs(err_1) < 1e-6 or abs(err_1) > 1-1e-6) and (abs(err_2) < 1e-6 or abs(err_2) > 1-1e-6):
        return 0
    z = (err_1 - err_2) / sqrt( (err_1*(1 - err_1) + err_2*(1-err_2)) / size )
    return z

def is_significant(z):
    if (stats.norm.cdf(z) < 0.05) or (stats.norm.cdf(z) > 0.95):
        return True
    else:
        return False

def plot_comparison(errors_1, errors_2, sizes):
    #print title
    plt.plot([0,1],[0,1])
    z_scores = get_z_scores(errors_1, errors_2, sizes)

    significance = list(map(is_significant, z_scores))
    results_signi_1 = list(compress(errors_1, significance))
    results_signi_2 = list(compress(errors_2, significance))

    plt.scatter(results_signi_1, results_signi_2, s=18, c='r')

    insignificance = [not b for b in significance]
    results_insigni_1 = list(compress(errors_1, insignificance))
    results_insigni_2 = list(compress(errors_2, insignificance))

    plt.scatter(results_insigni_1, results_insigni_2, s=2, c='k')

    len_errors = len(errors_1)
    wins_1 = [z_scores[i] < 0 and significance[i] for i in range(len_errors) ]
    wins_2 = [z_scores[i] > 0 and significance[i] for i in range(len_errors) ]
    num_wins_1 = wins_1.count(True)
    num_wins_2 = wins_2.count(True)

    return num_wins_1, num_wins_2

def order_legends(indices):
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles, indices = zip(*sorted(zip(labels, handles, indices), key=lambda t: t[2]))
    ax.legend(handles, labels)

def save_legend(mod, indices):
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    labels, handles, indices = zip(*sorted(zip(labels, handles, indices), key=lambda t: t[2]))
    #figlegend = pylab.figure(figsize=(26,1))
    #figlegend.legend(handles, labels, 'center', fontsize=26, ncol=8)
    figlegend = pylab.figure(figsize=(17,1.5))
    figlegend.legend(handles, labels, 'center', fontsize=26, ncol=3)
    figlegend.tight_layout(pad=0)
    figlegend.savefig(mod.problemdir+'legend.pdf')
    plt.close()

#def problem_str(name_problem):
#    return name_problem[0] + '_' + name_problem[1]

def plot_cdf(alg_name, errs):
    col, sty = alg_color_style(alg_name)
    plt.step(np.sort(errs), np.linspace(0, 1, len(errs), endpoint=False), label=alg_str(alg_name), color=col, linestyle=sty, linewidth=2.0)

def plot_all_cdfs(alg_results, mod):
    print('printing cdfs..')
    indices = []
    pylab.figure(figsize=(8,6))

    for alg_name, errs in alg_results.items():
        indices.append(alg_index(alg_name))
        plot_cdf(alg_name, errs)

    if mod.normalize_type == 1:
        plt.xlim(0,1)
    elif mod.normalize_type == 2:
        plt.xlim(-1,1)
    elif mod.normalize_type == 3:
        plt.xlim(0, 1)

    plt.ylim(0,1)
    #params={'legend.fontsize':26,
    #'axes.labelsize': 24, 'axes.titlesize':26, 'xtick.labelsize':20,
    #'ytick.labelsize':20 }
    #plt.rcParams.update(params)
    #plt.xlabel('Normalized error',fontsize=34)
    #plt.ylabel('Cumulative frequency', fontsize=34)
    #plt.title(problem_text(mod.name_problem), fontsize=36)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout(pad=0)

    ax = plt.gca()
    order_legends(indices)
    ax.legend_.set_zorder(-1)
    plt.savefig(mod.problemdir+'cdf.pdf')
    ax.legend_.remove()
    plt.savefig(mod.problemdir+'cdf_nolegend.pdf')
    save_legend(mod, indices)
    plt.close()

def plot_all_lrs(lrs, mod):
    alg_names = lrs.keys()

    for alg in alg_names:
        pylab.figure(figsize=(8,6))
        lrs_alg = lrs[alg]
        names = mod.learning_rates
        values = [lrs_alg.count(n) for n in names]
        plt.barh(range(len(names)),values)
        plt.yticks(range(len(names)),names)
        plt.savefig(mod.problemdir+alg_str_compatible(alg)+'_lr.pdf')
        plt.close()

def plot_all_lambdas(lambdas, mod):
    alg_names = lambdas.keys()

    for alg in alg_names:
        pylab.figure(figsize=(8,6))
        lambdas_alg = lambdas[alg]
        names = sorted(list(set(lambdas_alg)))
        values = [lambdas_alg.count(n) for n in names]
        plt.barh(range(len(names)),values)
        plt.yticks(range(len(names)),names)
        plt.savefig(mod.problemdir+alg_str_compatible(alg)+'_lambdas.pdf')
        plt.close()


def plot_all_pair_comp(alg_results, sizes, mod):
    alg_names = list(alg_results)

    for i in range(len(alg_names)):
        for j in range(len(alg_names)):
            if i < j:
                errs_1 = alg_results[alg_names[i]]
                errs_2 = alg_results[alg_names[j]]

                #print(len(errs_1), len(errs_2), len(sizes))
                num_wins_1, num_wins_2 = plot_comparison(errs_1, errs_2, sizes)
                plt.title( 'total number of comparisons = ' + str(len(errs_1)) + '\n'+
                alg_str(alg_names[i]) + ' wins ' + str(num_wins_1) + ' times, \n' + alg_str(alg_names[j]) + ' wins ' + str(num_wins_2) + ' times')
                plt.savefig(mod.problemdir+alg_str_compatible(alg_names[i])+'_vs_'+alg_str_compatible(alg_names[j])+'.pdf')
                plt.close()


def normalize_score(unnormalized_result, mod):
    if mod.normalize_type == 1:
        l = unnormalized_result['Optimal']
        u = max(unnormalized_result.values())
        normalized = { k : ((v - l) / (u - l + 1e-4)) for k, v in unnormalized_result.items() }
    elif mod.normalize_type == 2:
        l = unnormalized_result['Bandit-Only']
        normalized = { k : ((v - l) / (l + 1e-4)) for k, v in unnormalized_result.items() }
    elif mod.normalize_type == 3:
        normalized = unnormalized_result
    elif mod.normalize_type == 4:
        l = unnormalized_result['Optimal']
        normalized = { k : (v - l) for k, v in unnormalized_result.items() }
    return normalized

def get_best_error(best_error_table, name_dataset):
    name = name_dataset[0]
    best_error_oneline = best_error_table[best_error_table['dataset'] == name]
    best_error = best_error_oneline.loc[best_error_oneline.index[0], 'avg_error']
    return best_error

def get_maj_error(maj_error_table, name_dataset):
    name = name_dataset[0]
    maj_error_oneline = maj_error_table[maj_error_table['data'] == name]
    maj_error = maj_error_oneline.loc[maj_error_oneline.index[0], 'avg_error']
    return maj_error

def get_unnormalized_results(result_table):
    new_unnormalized_results = {}
    new_lr = {}
    new_lambda = {}
    new_size = 0

    #i = 0
    for idx, row in result_table.iterrows():
        #if i == 0:
        #    new_size = row['interaction_multiplier']

        #if row['interaction'] == new_size:
        alg_name = row['algorithm']
        new_unnormalized_results[alg_name] = row['avg_error']
        new_lr[alg_name] = row['learning_rate']
        new_lambda[alg_name] = row['last_lambda']
        #i += 1

    return new_size, new_unnormalized_results, new_lr, new_lambda

def update_result_dict(results_dict, new_result):
    if len(new_result) != len(results_dict):
        print('Warning: length of the new record ( ', len(new_result), ' ) does not match the length of the existing dict ( ', len(results_dict), ' ); perhaps the input data is corrupted.')

    for k, v in new_result.items():
        results_dict[k].append(v)

def plot_all(mod, all_results):
    #Group level 1: corruption mode, corruption prob, warm start - bandit ratio (each group corresponds to one cdf plot)
    problem_title = ['corruption','inter_ws_size_ratio','explore_method']
    grouped_by_problem = all_results.groupby(problem_title)

    for name_problem, group_problem in grouped_by_problem:
        normalized_results = None
        unnormalized_results = None
        sizes = None
        mod.name_problem = name_problem

        #print('in group_problem:', name_problem)
        #print(group_problem)

        #Group level 2: datasets, warm start length (corresponds to each point in cdf)
        #NOTE: warm start is not propagated in sup-only and most-freq, hence we group by warm_start_multiplier
        grouped_by_dataset = group_problem.groupby(['dataset',
                                                    'warm_start_multiplier'])

        for name_dataset, group_dataset in grouped_by_dataset:
            result_table = group_dataset

            #print 'in group_dataset:'
            #print(name_dataset)
            #print(result_table)

            #Record the error rates of all algorithms
            #Group level 3: algorithms
            new_size, new_unnormalized_result, new_lr, new_lambda = get_unnormalized_results(result_table)
            #print(len(new_unnormalized_result))
            if len(new_unnormalized_result) != 7:
                continue
            new_normalized_result = normalize_score(new_unnormalized_result, mod)

            new_normalized_result.pop('Optimal', None)
            new_lr.pop('Optimal', None)
            new_lambda.pop('Optimal', None)
            new_lr.pop('Most-Freq', None)
            new_lambda.pop('Most-Freq', None)

            #first time - generate names of algorithms considered
            if normalized_results is None:
                sizes = []
                unnormalized_results = dict([(k,[]) for k in new_unnormalized_result.keys()])
                normalized_results = dict([(k,[]) for k in new_normalized_result.keys()])
                lrs = dict([(k,[]) for k in new_lr.keys()])
                lambdas = dict([(k,[]) for k in new_lambda.keys()])

            update_result_dict(unnormalized_results, new_unnormalized_result)
            update_result_dict(normalized_results, new_normalized_result)
            update_result_dict(lrs, new_lr)
            update_result_dict(lambdas, new_lambda)
            sizes.append(new_size)

        #print(name_problem)
        #print(name_dataset)
        print(unnormalized_results)
        print(normalized_results)

        mod.problemdir = mod.fulldir+param_to_str(dict(zip(problem_title, name_problem)))+'/'
        if not os.path.exists(mod.problemdir):
            os.makedirs(mod.problemdir)

        if mod.pair_comp_on is True:
            plot_all_pair_comp(unnormalized_results, sizes, mod)
        if mod.cdf_on is True:
            plot_all_cdfs(normalized_results, mod)

        plot_all_lrs(lrs, mod)
        plot_all_lambdas(lambdas, mod)

def save_to_hdf(mod):
    print('saving to hdf..')
    print(mod.all_results)
    store = pd.HDFStore(mod.results_dir+'cache.h5')
    store['result_table'] = mod.all_results
    store.close()

def load_from_hdf(mod):
    print('reading from hdf..')
    store = pd.HDFStore(mod.results_dir+'cache.h5')
    mod.all_results = store['result_table']
    store.close()

def load_from_sum(mod):
    print('reading directory..')
    dss = sum_files(mod.results_dir)
    results_arr = []

    print('reading sum tables..')
    for i in range(len(dss)):
        print('result file name: ', dss[i])
        result = parse_sum_file(mod.results_dir + dss[i])
        results_arr.append(result)

    all_results = pd.concat(results_arr)
    #print(all_results)
    mod.all_results = all_results

def load_cached(mod):
    if os.path.exists(mod.results_dir+'cache.h5'):
        load_from_hdf(mod)
    else:
        load_from_sum(mod)
        save_to_hdf(mod)

def filter_results(mod, all_results):
    if mod.filter == '1':
        pass
    elif mod.filter == '2':
        all_results = all_results[all_results['warm_start'] >= 200]
    elif mod.filter == '3':
        all_results = all_results[all_results['num_classes'] >= 3]
    elif mod.filter == '4':
        all_results = all_results[all_results['num_classes'] <= 2]
    elif mod.filter == '5':
        all_results = all_results[all_results['total_size'] >= 10000]
        all_results = all_results[all_results['num_classes'] >= 3]
    elif mod.filter == '6':
        all_results = all_results[all_results['warm_start'] >= 100]
        all_results = all_results[all_results['learning_rate'] == 0.3]
    elif mod.filter == '7':
        all_results = all_results[all_results['warm_start'] >= 100]
        all_results = all_results[all_results['num_classes'] >= 3]
    elif mod.filter == '8':
        all_results = all_results[all_results['warm_start'] >= 100]

    return all_results

def propag_sup_only(sup_only, other):
    # propagate the exploration method for Sup-Only
    uniq_em = other['explore_method'].unique()
    sup_only_propag = []
    for em in uniq_em:
        sup_only_em = sup_only.copy(deep=True)
        sup_only_em['explore_method'] = em
        sup_only_propag.append(sup_only_em)
    return pd.concat(sup_only_propag)

def cartesian(df1, df2):
    df1_copy = df1.copy()
    df2_copy = df2.copy()
    df1_copy['tmp_key'] = 0
    df2_copy['tmp_key'] = 0
    prod = df1_copy.merge(df2_copy, how='left', on = 'tmp_key')
    prod.drop('tmp_key', 1, inplace=True)
    return prod

def propag_opt_maj(opt_maj, other):
    grouped = other.groupby(['dataset'])
    opt_maj_propag = []
    repl_var = [  'corruption',
                  'warm_start_multiplier',
                  'interaction_multiplier',
                  'inter_ws_size_ratio',
                  'explore_method'
               ]
    for ds, subtable in grouped:
        #print(ds)
        subt = subtable[repl_var]
        uniq_subt = subt.drop_duplicates()
        opt_maj_ds = opt_maj[opt_maj['dataset'] == ds]
        opt_maj_ds = opt_maj_ds.drop(repl_var, axis=1)
        #print("opt_maj_ds = ", opt_maj_ds)
        #print("uniq_subt = ", uniq_subt)
        opt_maj_setting = cartesian(opt_maj_ds, uniq_subt)
        opt_maj_propag.append(opt_maj_setting)
        #print(opt_maj_setting)
        #print(opt_maj_setting.shape)
    return pd.concat(opt_maj_propag)

def propagate(all_res):
    opt_maj_mask = ((all_res['algorithm'] == 'Optimal') | (all_res['algorithm'] == 'Most-Freq'))
    sup_only_mask = (all_res['algorithm'] == 'Sup-Only')
    other_mask = ~(opt_maj_mask | sup_only_mask)

    opt_maj = all_res[opt_maj_mask]
    sup_only = all_res[sup_only_mask]
    other = all_res[other_mask]

    print('propagating Sup-Only results..')
    # propagate for Sup-Only
    prop_sup_only= propag_sup_only(sup_only, other)
    # propagate for Most-Freq and Optimal
    print('propagating Optimal/Most-Freq results..')
    prop_opt_maj = propag_opt_maj(opt_maj, other)
    #all_results = propag_opt_maj(all_results)
    print('propagating complete')

    prop_res = pd.concat([other, prop_opt_maj, prop_sup_only], sort=True)
    return prop_res

'''
def avg_folds(all_res):
    #potential problem: last lambda, after averaging, might not make sense
    #excl = list(filter(lambda item: item != 'fold' and item != 'avg_error', list(all_res)))
    excl = ['corruption', 'warm_start_multiplier', 'interaction_multiplier', 'explore_method', 'dataset', 'algorithm', 'learning_rate']
    return all_res.groupby(excl).mean().reset_index()
'''

def tune_lr(all_res):

    setting_with_lr = ['corruption', 'warm_start_multiplier', 'interaction_multiplier', 'explore_method', 'dataset', 'algorithm', 'learning_rate']

    avg_folds = all_res.groupby(setting_with_lr).mean().reset_index()

    #excl = list(filter(lambda item: item != 'learning_rate' and item != 'avg_error', list(all_res)))
    setting_no_lr = ['corruption', 'warm_start_multiplier', 'interaction_multiplier', 'explore_method', 'dataset', 'algorithm']
    select_lr = avg_folds.iloc[ avg_folds.groupby(setting_no_lr)['avg_error'].idxmin(), : ]
    select_lr = select_lr.loc[:, setting_with_lr]
    #print(select_lr.shape)

    tuned = all_res.join(select_lr.set_index(setting_with_lr), on=setting_with_lr, how='inner')

    return tuned

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='result summary')
    parser.add_argument('--results_dir', default='../../../figs/')
    parser.add_argument('--filter', default='1')
    parser.add_argument('--plot_subdir', default='expt1/')
    parser.add_argument('--cached', action='store_true')
    parser.add_argument('--normalize_type', type=int, default=1)
    parser.add_argument('--pair_comp', action='store_true')
    #1: normalized score;
    #2: bandit only centered score;
    #3: raw score
    #4: normalized score w/o denominator

    args = parser.parse_args()
    mod = model()

    mod.results_dir = args.results_dir
    mod.filter = args.filter
    mod.plot_subdir = args.plot_subdir
    mod.normalize_type = args.normalize_type
    mod.pair_comp_on = args.pair_comp
    mod.cdf_on = True
    #mod.maj_error_dir = '../../../old_figs/figs_all/expt_0509/figs_maj_errors/0of1.sum'
    #mod.best_error_dir = '../../../old_figs/figs_all/expt_0606/0of1.sum'

    mod.fulldir = mod.results_dir + mod.plot_subdir
    if not os.path.exists(mod.fulldir):
        os.makedirs(mod.fulldir)

    if args.cached is True:
        load_cached(mod)
    else:
        load_from_sum(mod)

    all_results = mod.all_results
    #all_results = all_results[all_results['dataset'] == 'ds_vehicle_cs_randcost_54_4.vw.gz']
    #all_results = all_results[all_results['learning_rate'] < 0.004]
    #all_results = avg_folds(all_results)
    all_results = tune_lr(all_results)
    all_results = propagate(all_results)

    #all_results = all_results.loc[:, ['problem_setting', 'explore_method', 'dataset', 'warm_start', 'algorithm', 'learning_rate', 'avg_error']]
    #all_results = all_results[all_results['corrupt_prob_warm_start'] < 0.6]
    #ignore the choices_lambda = 4 row
    #all_results = all_results[(all_results['choices_lambda'] != 4)]
    #all_results = all_results[(all_results['choices_lambda'] != 8)]
    all_results['learning_rate'] = all_results['learning_rate'].astype(float)
    mod.learning_rates = sorted(all_results.learning_rate.unique())
    all_results = filter_results(mod, all_results)

    plot_all(mod, all_results)

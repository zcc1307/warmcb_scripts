import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import os
import glob
import pandas as pd
import argparse
import numpy as np
from collections import OrderedDict
import math
from run_vw_commands import param_to_str
import re
from alg_const import make_header
from cdfs import plot_all_cdfs
from pairwise_comp import plot_all_pair_comp


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
    print(table['fold'].unique())
    if 'fold' not in list(table):
        table = pd.DataFrame()
    return table

def update_result_dict(results_dict, new_result):
    if len(new_result) != len(results_dict):
        print('Warning: length of the new record ( ', len(new_result), ' ) does not match the length of the existing dict ( ', len(results_dict), ' ); perhaps the input data is corrupted.')

    for k, v in new_result.items():
        results_dict[k].append(v)

def plot_agg_ratio(mod, df):
    return plot_bilevel(mod, df, ['corrupt_type_warm_start',
                                  'corrupt_prob_warm_start',
                                  'epsilon'],
                                 ['inter_ws_size_ratio',
                                  'dataset',
                                  'warm_start_multiplier'])

def plot_agg_ratio_eq(mod, df):
    return plot_eq_iw(mod, df, ['corrupt_type_warm_start',
                                'corrupt_prob_warm_start',
                                'epsilon'],
                                ['dataset',
                                 'warm_start_multiplier'])

def plot_agg_all(mod, df):
    return plot_bilevel(mod, df, ['epsilon'],
                                 ['corrupt_type_warm_start',
                                  'corrupt_prob_warm_start',
                                  'inter_ws_size_ratio',
                                  'dataset',
                                  'warm_start_multiplier'])

def plot_agg_all_eq(mod, df):
    return plot_eq_iw(mod, all_results, ['epsilon'],
                                        ['corrupt_type_warm_start',
                                         'corrupt_prob_warm_start',
                                         'dataset',
                                         'warm_start_multiplier'])


def plot_agg_corr_prob(mod, all_results):
    return plot_bilevel(mod, all_results, ['corrupt_type_warm_start',
                                           'inter_ws_size_ratio',
                                           'epsilon'],
                                           ['corr_prob_warm_start',
                                            'dataset',
                                            'warm_start_multiplier'])

def plot_no_agg(mod, all_results):
    return plot_bilevel(mod, all_results, ['corrupt_type_warm_start',
                                           'corrupt_prob_warm_start',
                                           'inter_ws_size_ratio',
                                           'epsilon'],
                                           ['dataset',
                                            'warm_start_multiplier'])


def plot_eq_iw(mod, all_results, enum, agg):
    for expl, group_expl in all_results.groupby(enum):
        norm_scores_all = {}
        ratio_lens = []

        if len(enum) == 1:
           expt_dict = OrderedDict(zip(enum, [expl]))
        else:
           expt_dict = OrderedDict(zip(enum, list(expl)))

        header = make_header(expt_dict)
        dir = mod.fulldir+param_to_str(expt_dict)+'/'
        if not os.path.exists(dir):
            os.makedirs(dir)

        for ratio, group_ratio in group_expl.groupby(['inter_ws_size_ratio']):
           unnorm_scores, norm_scores, lrs, lambdas, sizes = get_scores(group_ratio, agg)
           ratio_lens.append(len(list(norm_scores.values())[0]))
           insert_scores(norm_scores_all, norm_scores)

        num_ratios = len(ratio_lens)
        iw = []
        for i in range(num_ratios):
            iw += [ 1.0 / (ratio_lens[i] * num_ratios) for _ in range(ratio_lens[i]) ]

        plot_all_cdfs(norm_scores_all, dir, header, iw)

def plot_bilevel(mod, all_results, enum, agg):
    grouped_by_expt = all_results.groupby(enum)

    for expt, group_expt in grouped_by_expt:
        print(expt)
        unnorm_scores, norm_scores, sizes = get_scores(group_expt, agg)

        if len(enum) == 1:
            expt_dict = OrderedDict(zip(enum, [expt]))
        else:
            expt_dict = OrderedDict(zip(enum, list(expt)))

        header = make_header(expt_dict)
        dir = mod.fulldir+param_to_str(expt_dict)+'/'
        if not os.path.exists(dir):
            os.makedirs(dir)

        if mod.pair_comp_on is True:
            plot_all_pair_comp(unnorm_scores, sizes, dir, header)
        if mod.cdf_on is True:
            plot_all_cdfs(norm_scores, dir, header)


def insert_scores(scores_all, scores_new, mode):
    for k, v in scores_new.items():
        if k not in scores_all.keys():
            scores_all[k] = []
        scores_all[k].append(v)

def get_scores(results, gr):
    norm_scores = None
    unnorm_scores = None
    sizes = None
    lrs = None
    lambdas = None

    #Group level
    grouped_by = results.groupby(gr)

    for name, group in grouped_by:
        new_size = float(group.loc[group['algorithm'] == 'Bandit-Only', 'interaction'])

        new_norm_score = {}
        new_unnorm_score = {}

        for idx, row in group.iterrows():
            new_norm_score[row['algorithm']] = row['norm_error']
            new_unnorm_score[row['algorithm']] = row['avg_error']

        new_norm_score.pop('Optimal', None)

        #first time - generate names of algorithms considered
        if norm_scores is None:
            sizes = []
            unnorm_scores = dict([(k,[]) for k in new_unnorm_score.keys()])
            norm_scores = dict([(k,[]) for k in new_norm_score.keys()])

        update_result_dict(unnorm_scores, new_unnorm_score)
        update_result_dict(norm_scores, new_norm_score)
        sizes.append(new_size)

    return unnorm_scores, norm_scores, sizes

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

def filter_results(filter, all_results):
    print('apply filters..')
    if filter == '1':
        pass
    elif filter == '2':
        all_results = all_results[all_results['warm_start'] >= 100]

    return all_results

def propag_sup_only(sup_only, other):
    # propagate the exploration method for Sup-Only
    uniq_em = other['epsilon'].unique()
    sup_only_propag = []
    for em in uniq_em:
        sup_only_em = sup_only.copy(deep=True)
        sup_only_em['epsilon'] = em
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
    #NOTE: we need warm start / interaction value to be propagated, in order to
    #apply the filtering correctly

    repl_var = ['warm_start_multiplier',
                'warm_start',
                'inter_ws_size_ratio',
                'corrupt_type_warm_start',
                'corrupt_prob_warm_start',
                'epsilon']

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

    # propagate for Sup-Only
    print('propagating Sup-Only results..')
    prop_sup_only= propag_sup_only(sup_only, other)

    # propagate for Most-Freq and Optimal
    print('propagating Optimal/Most-Freq results..')
    prop_opt_maj = propag_opt_maj(opt_maj, other)
    #all_results = propag_opt_maj(all_results)

    print('propagating complete')

    prop_res = pd.concat([other, prop_opt_maj, prop_sup_only], sort=True)
    return prop_res


def tune_lr(all_res):
    print('tuning learning rates..')
    setting_no_lr = ['dataset',
                     'warm_start_multiplier',
                     'warm_start',
                     'inter_ws_size_ratio',
                     'corrupt_type_warm_start',
                     'corrupt_prob_warm_start',
                     'algorithm',
                     'epsilon']

    setting_lr = setting_no_lr + ['learning_rate']

    avg_folds = all_res.groupby(setting_lr).mean().reset_index()

    select_lr = avg_folds.iloc[ avg_folds.groupby(setting_no_lr)['avg_error'].idxmin(), : ]
    select_lr = select_lr.loc[:, setting_lr]
    #print(select_lr.shape)

    tuned = all_res.join(select_lr.set_index(setting_lr), on=setting_lr, how='inner')

    return tuned

def alg_name(row):
    if row['algorithm'] == 'ARRoW-CB':
        return 'ARRoW-CB' + ',' + str(row['choices_lambda'])
    else:
        return row['algorithm']

def rename_alg(df):
    df['algorithm'] = df.apply(alg_name, axis=1)
    return df


def norm_scores(all_res):
    print('computing normalized scores')

    all_res = all_res.reset_index(drop=True)

    setting = ['dataset',
               'warm_start_multiplier',
               'warm_start',
               'inter_ws_size_ratio',
               'corrupt_type_warm_start',
               'corrupt_prob_warm_start',
               'epsilon']

    grouped = all_res.groupby(setting)

    for s, subg in grouped:
        if subg.shape[0] != 7:
            continue
        else:
            opt_err = float(subg.loc[subg['algorithm'] == 'Optimal', 'avg_error'])
            max_err = subg.loc[:, 'avg_error'].max()
            for idx, row in subg.iterrows():
                all_res.at[idx, 'norm_error'] = ((row['avg_error'] - opt_err) / (max_err - opt_err + 1e-4))

    return all_res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='result summary')
    parser.add_argument('--results_dir', default='../../output/')
    parser.add_argument('--filter', default='1')
    parser.add_argument('--plot_subdir', default='expt1/')
    parser.add_argument('--cached', action='store_true')
    parser.add_argument('--normalize_type', type=int, default=1)
    parser.add_argument('--pair_comp', action='store_true')
    parser.add_argument('--agg_mode', default='all')

    #Normalize_type:
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

    all_results['learning_rate'] = all_results['learning_rate'].astype(float)
    mod.learning_rates = sorted(all_results.learning_rate.unique())

    tuned_results = tune_lr(all_results)
    propag_results = propagate(tuned_results)
    normed = norm_scores(propag_results)
    renamed = rename_alg(normed)

    filt_results = filter_results(mod.filter, normed)

    if args.agg_mode == 'all':
        plot_agg_all(mod, filt_results)
    if args.agg_mode == 'all_eq':
        plot_agg_all_eq(mod, filt_results)
    elif args.agg_mode == 'no':
        plot_no_agg(mod, filt_results)
    elif args.agg_mode == 'agg_corr_prob':
        plot_agg_corr_prob(mod, filt_results)
    elif args.agg_mode == 'agg_ratio':
        plot_agg_ratio(mod, filt_results)
    elif args.agg_mode == 'agg_ratio_eq':
        plot_agg_ratio_eq(mod, filt_results)

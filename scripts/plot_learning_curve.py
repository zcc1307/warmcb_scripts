from alg_comparison import load_cached, tune_lr, propagate, order_legends
from alg_const import alg_str, alg_color_style, alg_index
from run_vw_commands import extract_vw_output, remove_suffix, param_to_str, replace_keys
from vw_commands_const import SIMP_MAP
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from random import random

class model:
    def __init__(self):
        pass

def extract_lc(vw_filename):
    lc_x = []
    lc_y = []
    lambdas = []
    results = extract_vw_output(vw_filename)
    for s in results:
        counter_new, last_lambda, actual_var, ideal_var, \
        avg_loss, last_loss, counter, weight, curr_label, curr_pred, curr_feat = s
        lc_x.append(float(weight))
        lc_y.append(float(avg_loss))
        lambdas.append(float(last_lambda))

    return (lc_x, lc_y, lambdas)

def plot_lcs(lc_alg, plot_name):

    for alg_name, lcs in lc_alg.items():
        for lc in lcs:
            lc_x, lc_y = lc
            if len(lc_x) != 1:
                def_lc_x = lc_x
                break

    plt.figure()

    indices = []
    for alg_name, lcs in lc_alg.items():
        # propagate for Sup-Only, Optimal, Most-Freq
        flag = False
        for lc in lcs:
            lc_x, lc_y = lc
            if len(lc_x) == 1:
                flag = True

        if flag:
            lc_x = def_lc_x
            lc_ys = [[lc[1][0] for x_tick in def_lc_x] for lc in lcs]
        else:
            lc_x = lcs[0][0]
            lc_ys = [lc[1] for lc in lcs]
        reps = len(lcs)

        lc_y_avg = [np.mean(x) + 0.001*random() for x in zip(*lc_ys)]
        lc_y_std = [np.std(x) / np.sqrt(reps) for x in zip(*lc_ys)]
        lc_y_ci = [1.96*y for y in lc_y_std]

        alg_name_latex = alg_str(alg_name)
        alg_col, alg_sty = alg_color_style(alg_name)

        plt.plot(lc_x, lc_y_avg, label=alg_name_latex, color=alg_col, linestyle=alg_sty, linewidth=2.0)
        lc_y_up = [avg + ci for avg, ci in zip(lc_y_avg, lc_y_ci)]
        lc_y_dn = [avg - ci for avg, ci in zip(lc_y_avg, lc_y_ci)]
        plt.fill_between(lc_x, lc_y_up, lc_y_dn, color=alg_col, linestyle=alg_sty, alpha=0.2)
        #plt.errorbar(lc_x, lc_y_avg, yerr=lc_y_std, label=alg_name_latex, color=alg_col, linestyle=alg_sty, linewidth=2.0)
        indices.append(alg_index(alg_name))

    order_legends(indices)
    ax = plt.gca()
    ax.legend_.set_zorder(-1)
    plt.savefig(plot_name + '.pdf')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='result summary')
    parser.add_argument('--results_dir', default='../../../figs/')

    args = parser.parse_args()
    mod = model()
    mod.results_dir = args.results_dir
    load_cached(mod)

    all_res = mod.all_results

    #all_res = avg_folds(all_res)
    print(all_res.shape)
    print(all_res)
    all_res = tune_lr(all_res)
    print(all_res.shape)
    print(all_res)

    #opt_maj_table = all_res[(all_res['algorithm'] == 'Optimal') | (all_res['algorithm'] == 'Most-Freq')]
    #large_ratio_table = all_res[all_res['interaction_multiplier'] > 180.0]
    #all_res = pd.concat([opt_maj_table, large_ratio_table])
    all_res = propagate(all_res)

    group_vars = ['dataset',
                   'warm_start_multiplier',
                   'interaction_multiplier',
                   'corruption',
                   'explore_method']

    grouped = all_res.groupby(group_vars)
    for setting, res in grouped:
        group_dict = OrderedDict(zip(group_vars, setting))
        group_simp = replace_keys(group_dict, SIMP_MAP)
        print('plotting', param_to_str(group_simp), '...')

        lc_alg = OrderedDict()
        lambdas_alg = OrderedDict()

        uniq_alg = res['algorithm'].unique()
        for alg_name in uniq_alg:
            lc_alg[alg_name] = []
            if alg_name.startswith('One-Lambda') or alg_name.startswith('AwesomeBandits'):
                print(alg_name)
                lambdas_alg[alg_name] = []

        for idx, row in res.iterrows():
            alg_name = row['algorithm']
            if alg_name == 'Optimal' or alg_name == 'Most-Freq':
                lc_alg[alg_name].append(([0] , [row['avg_error']]))
            else:
                print(alg_name, row['learning_rate'])
                lc_full = extract_lc(row['vw_output_name'])
                #lambdas = lc_full[2]
                setting_dict = dict(zip(group_vars, setting))
                cutoff_pt = sum([wt <= row['interaction'] for wt in lc_full[0]])
                #grid_pt = int(row['interaction'] / (row['interaction_multiplier'] * 4.0) )
                lc_cutoff = (lc_full[0][:cutoff_pt], lc_full[1][:cutoff_pt])
                lambdas_cutoff = (lc_full[0][:cutoff_pt], lc_full[2][:cutoff_pt])
                #lc_cutoff = (lc_full[0][grid_pt:cutoff_pt:grid_pt], lc_full[1][grid_pt:cutoff_pt:grid_pt])
                lc_alg[alg_name].append(lc_cutoff)

                if alg_name.startswith('One-Lambda') or alg_name.startswith('AwesomeBandits'):
                    lambdas_alg[alg_name].append(lambdas_cutoff)

        #print(lc_alg)
        ds = remove_suffix(group_dict['dataset'])
        group_simp.pop('ds', None)
        plot_name = mod.results_dir + ds + '/' + param_to_str(group_simp)
        plot_lcs(lc_alg, plot_name)

        #print(lambdas_alg)
        plot_name_lambda = plot_name + '_lambdas'
        plot_lcs(lambdas_alg, plot_name_lambda)

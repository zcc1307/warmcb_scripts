from alg_comparison import load_cached, avg_folds, tune_lr, propagate, order_legends
from alg_const import alg_str, alg_color_style, alg_index
from run_vw_commands import extract_vw_output, remove_suffix, param_to_str, replace_keys
from vw_commands_const import SIMP_MAP
import pandas as pd
import argparse
import matplotlib.pyplot as plt

class model:
    def __init__(self):
        pass

def extract_lc(vw_filename):
    lc_x = []
    lc_y = []
    results = extract_vw_output(vw_filename)
    for s in results:
        counter_new, last_lambda, actual_var, ideal_var, \
        avg_loss, last_loss, counter, weight, curr_label, curr_pred, curr_feat = s
        lc_x.append(float(weight))
        lc_y.append(float(avg_loss))

    return (lc_x, lc_y)

def plot_lcs(lcs, plot_name):

    for alg_name, lc in lcs.items():
        lc_x, lc_y = lc
        if len(lc_x) != 1:
            def_lc_x = lc_x
            break

    plt.figure()

    indices = []
    for alg_name, lc in lcs.items():
        lc_x, lc_y = lc
        if len(lc_x) == 1:
            lc_x = def_lc_x
            lc_y = [lc_y[0] for xtick in lc_x]

        alg_name_latex = alg_str(alg_name)
        alg_col, alg_sty = alg_color_style(alg_name)
        plt.errorbar(lc_x, lc_y, label=alg_name_latex, color=alg_col, linestyle=alg_sty, linewidth=2.0)
        indices.append(alg_index(alg_name))
        #yerr=[0]

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
    all_res = tune_lr(all_res)

    opt_maj_table = all_res[(all_res['algorithm'] == 'Optimal') | (all_res['algorithm'] == 'Most-Freq')]
    large_ratio_table = all_res[all_res['interaction_multiplier'] > 180.0]
    all_res = pd.concat([opt_maj_table, large_ratio_table])
    all_res = propagate(all_res)

    group_vars = ['dataset',
                   'warm_start_multiplier',
                   'corruption',
                   'explore_method']

    grouped = all_res.groupby(group_vars)
    for setting, res in grouped:
        lcs = {}
        for idx, row in res.iterrows():
            alg_name = row['algorithm']
            if alg_name == 'Sup-Only' or alg_name == 'Optimal' or alg_name == 'Most-Freq':
                lcs[alg_name] = ([0] , [row['avg_error']])
            else:
                lcs[alg_name] = extract_lc(row['vw_output_name'])

        print(lcs)

        group_dict = dict(zip(group_vars, setting))
        ds = remove_suffix(group_dict['dataset'])
        group_dict.pop('dataset', None)
        group_simp = replace_keys(group_dict, SIMP_MAP)

        plot_name = mod.results_dir + ds + '/' + param_to_str(group_simp)
        plot_lcs(lcs, plot_name)

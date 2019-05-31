from vw_commands_const import SIMP_MAP
from utils import merge_dicts
import math

def param_cartesian(param_set_1, param_set_2):
    prod = []
    for param_1 in param_set_1:
        for param_2 in param_set_2:
            prod.append(merge_dicts(param_1, param_2))
    return prod

def param_cartesian_multi(param_sets):
    prod = [{}]
    for param_set in param_sets:
        prod = param_cartesian(prod, param_set)
    return prod


def dictify(param_name, param_choices):
    result = []
    for param in param_choices:
        dic = {}
        if isinstance(param_name, tuple):
            for i in range(len(param_name)):
                dic[param_name[i]] = param[i]
        else:
            dic[param_name] = param
        result.append(dic)
    print(param_name, result)
    return result

def get_filter(mod):
    fltr_inter_gt = lambda p: (p['corrupt_type_warm_start'] == 1 #filter out repetitive warm start data
                            or abs(p['corrupt_prob_warm_start']) > 1e-4)
    return fltr_inter_gt

def get_params_alg(mod, prm_com, prm_choices_lbd):
    # Algorithm (RoWS-CB with |Lambda|=8, RoWS-CB with |Lambda|=2) parameters construction
    if mod.algs_on:
        # Algorithms for bandit ground truth
        prm_inter_gt = \
        [
             [
                {'algorithm':'ARRoW-CB',
                 'warm_start_update': True,
                 'interaction_update': True,
                 'lambda_scheme': 4}
             ]
        ]
        prm_algs = param_cartesian_multi([prm_com] + [prm_choices_lbd] + prm_inter_gt)
    else:
        prm_algs = []
    return prm_algs


def get_params_opt(mod):
    # Optimal (approximating the best policy) parameter construction
    if mod.optimal_on:
        prm_optimal = \
        [
            {'algorithm': 'Optimal',
             'corrupt_type_warm_start': 1,
             'corrupt_prob_warm_start': 0.0,
             'corrupt_type_interaction': 1,
             'corrupt_prob_interaction': 0.0,
             'fold': 1}
        ]
    else:
        prm_optimal = []
    return prm_optimal

def get_params_maj(mod):
    # Majority (predicting the class of the largest proportion) parameter construction
    if mod.majority_on:
        prm_majority = \
        [
            {'algorithm': 'Most-Freq',
             'corrupt_type_warm_start': 1,
             'corrupt_prob_warm_start': 0.0,
             'corrupt_type_interaction': 1,
             'corrupt_prob_interaction': 0.0,
             'fold': 1}
        ]
    else:
        prm_majority = []
    return prm_majority

def get_params_baseline_sup(mod, prm_com_noeps):
    #Sup-Only
    if mod.baselines_on:
        prm_sup_only_basic = [[]]
        if mod.sup_only_on:
            prm_sup_only_basic[0] += \
                [
                     {'algorithm':'Sup-Only',
                      'warm_start_update': True,
                      'interaction_update': False,
                      'epsilon': 0.0,
                      'choices_lambda':1
                     }
                ]

        prm_baseline_sup = param_cartesian_multi([prm_com_noeps] + prm_sup_only_basic)
    else:
        prm_baseline_sup = []

    return prm_baseline_sup


def get_params_baseline_band(mod, prm_com):
    # Baseline (Sup-only, Bandit-Only, Sim-Bandit) parameters construction
    if mod.baselines_on:
        prm_oth_baseline_basic = [[]]
        #Bandit-Only
        if mod.band_only_on:
            prm_oth_baseline_basic[0] += \
                [
                     {'algorithm':'Bandit-Only',
                      'warm_start_update': False,
                      'interaction_update': True,
                      'choices_lambda':1
                     }
                ]

        #Sim-Bandit
        if mod.sim_bandit_on:
            prm_oth_baseline_basic[0] += \
                [
                    {'algorithm':'Sim-Bandit',
                     'sim_bandit': True,
                     'warm_start_update': True,
                     'interaction_update': True,
                     'choices_lambda':1
                     }
                ]

        prm_baseline_band = param_cartesian_multi([prm_com] + prm_oth_baseline_basic)
    else:
        prm_baseline_band = []

    return prm_baseline_band


def extend_item(prm, items, item_str):
    for item in items:
        if item in prm:
            prm[item_str] += (',' + SIMP_MAP[item] + '=' + str(prm[item]))

def get_all_params(mod):
    # Problem-specific parameters
    prm_cor_type_ws = dictify('corrupt_type_warm_start', mod.choices_cor_type_ws)
    prm_cor_prob_ws = dictify('corrupt_prob_warm_start', mod.choices_cor_prob_ws)
    prm_ws_multiplier = dictify('warm_start_multiplier', mod.ws_multipliers)
    prm_lrs = dictify('learning_rate', mod.learning_rates)
    prm_fold = dictify('fold', mod.folds)

    # Algorithm-specific parameters
    prm_cb_type = dictify('cb_type', mod.choices_cb_type)
    prm_dataset = dictify('dataset', mod.dss)
    prm_choices_lbd = dictify('choices_lambda', mod.choices_choices_lambda)
    prm_choices_eps = dictify('epsilon', mod.choices_epsilon)
    prm_adf_on = dictify('adf_on', mod.choices_adf)
    #prm_cs_on = dictify('cs_on', [mod.cs_on])
    prm_loss_enc = dictify(('loss0', 'loss1'), mod.choices_loss_enc)

    # Common parameters
    # Corruption parameters
    prm_cor = param_cartesian_multi([prm_cor_type_ws,
                                     prm_cor_prob_ws])
    fltr_inter_gt = get_filter(mod)
    prm_cor = filter(lambda p: fltr_inter_gt(p), prm_cor)

    prm_com_noeps = param_cartesian_multi([prm_cor,
                                           prm_ws_multiplier,
                                           prm_lrs,
                                           prm_cb_type,
                                           prm_fold,
                                           prm_adf_on,
                                           prm_loss_enc])
                                           #prm_cs_on,
    prm_com = param_cartesian_multi([prm_com_noeps, prm_choices_eps])

    # Optimal
    prm_opt = get_params_opt(mod)
    # Majority
    prm_maj = get_params_maj(mod)
    # Baseline: Sup-only
    prm_baseline_sup = get_params_baseline_sup(mod, prm_com_noeps)
    # Baselines: Bandit-Only, Sim-Bandit
    prm_baseline_band = get_params_baseline_band(mod, prm_com)
    # Algorithms: RoWS-CB with |Lambda|=8, RoWS-CB with |Lambda|=2
    prm_alg = get_params_alg(mod, prm_com, prm_choices_lbd)

    # Concatentate dataset names to in all four groups
    prm_all = param_cartesian_multi([prm_dataset,
                                     prm_baseline_sup +
                                     prm_baseline_band +
                                     prm_alg +
                                     prm_opt +
                                     prm_maj])

    prm_all = sorted(prm_all, key=lambda d: (d['dataset'],
                                             d['corrupt_type_warm_start'],
                                             d['corrupt_prob_warm_start']))
    #for prm in prm_all:
    #    prm = extend_prm(prm)

    print('The total number of VW commands to run is: ', len(prm_all))

    for prm in prm_all:
        print(prm)

    return prm_all

from vw_commands_const import SIMP_MAP

def merge_two_dicts(x, y):
	#print 'x = ', x
	#print 'y = ', y
	z = x.copy()   # start with x's keys and values
	z.update(y)    # modifies z with y's keys and values & returns None
	return z

def param_cartesian(param_set_1, param_set_2):
	prod = []
	for param_1 in param_set_1:
		for param_2 in param_set_2:
			prod.append(merge_two_dicts(param_1, param_2))
	return prod

def param_cartesian_multi(param_sets):
	#print param_sets
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


def get_all_params(mod):
    # Problem parameters
    prm_cor_type_ws = dictify('corrupt_type_warm_start', mod.choices_cor_type_ws)
    prm_cor_prob_ws = dictify('corrupt_prob_warm_start', mod.choices_cor_prob_ws)
    prm_cor_type_inter = dictify('corrupt_type_interaction', mod.choices_cor_type_inter)
    prm_cor_prob_inter = dictify('corrupt_prob_interaction', mod.choices_cor_prob_inter)
    prm_ws_multiplier = dictify('warm_start_multiplier', mod.ws_multipliers)
    prm_lrs = dictify('learning_rate', mod.learning_rates)
    # could potentially induce a bug if the maj and best does not have this parameter
    prm_fold = dictify('fold', mod.folds)
    # Algorithm parameters
    prm_cb_type = dictify('cb_type', mod.choices_cb_type)
    prm_dataset = dictify('dataset', mod.dss)
    prm_choices_lbd = dictify('choices_lambda', mod.choices_choices_lambda)
    prm_choices_eps = dictify('epsilon', mod.choices_epsilon) + dictify('eps_t', mod.choices_eps_t)
    prm_adf_on = dictify('adf_on', mod.choices_adf)
    prm_cs_on = dictify('cs_on', mod.choices_cs)
    prm_loss_enc = dictify(('loss0', 'loss1'), mod.choices_loss_enc)

    # Common parameters

    # Corruption parameters
    prm_cor = param_cartesian_multi(
    [prm_cor_type_ws,
     prm_cor_prob_ws,
     prm_cor_type_inter,
     prm_cor_prob_inter])
    fltr_inter_gt, fltr_ws_gt = get_filters(mod)
    prm_cor = filter(lambda p: (fltr_ws_gt(p) or fltr_inter_gt(p)), prm_cor)

    prm_com_noeps = param_cartesian_multi(
    [prm_cor,
     prm_ws_multiplier,
     prm_lrs,
     prm_cb_type,
     prm_fold,
     prm_adf_on,
     prm_cs_on,
     prm_loss_enc])

    prm_com = param_cartesian(prm_com_noeps, prm_choices_eps)
    prm_com_ws_gt = filter(fltr_ws_gt, prm_com)
    prm_com_inter_gt = filter(fltr_inter_gt, prm_com)

    prm_baseline = get_params_baseline(mod, prm_com, prm_com_noeps)
    prm_alg = get_params_alg(mod, prm_com_ws_gt, prm_com_inter_gt, prm_choices_lbd)
    prm_opt = get_params_opt(mod)
    prm_maj = get_params_maj(mod)

    #for p in params_common:
    #	print p
    #for p in params_baseline:
    #	print p
    #print len(prm_com_ws_gt), len(prm_algs_ws_gt)
    #print len(prm_com_inter_gt), len(prm_algs_inter_gt)
    #print len(prm_com)
    #print len(prm_baseline)
    #print len(prm_algs)
    #raw_input('..')

    # Common factor in all 3 groups: dataset
    prm_all = param_cartesian_multi(
    [prm_dataset,
     prm_baseline + prm_alg + prm_opt + prm_maj])

    prm_all = sorted(prm_all,
    					key=lambda d: (d['dataset'],
    					               d['corrupt_type_warm_start'],
    								   d['corrupt_prob_warm_start'],
    								   d['corrupt_type_interaction'],
    								   d['corrupt_prob_interaction'])
    				 )
    print('The total number of VW commands to run is: ', len(prm_all))
    for row in prm_all:
    	print(row)
    return prm_all

def get_filters(mod):
	if mod.inter_gt_on:
		fltr_inter_gt = lambda p: ((p['corrupt_type_interaction'] == 1 #noiseless for interaction data
								and abs(p['corrupt_prob_interaction']) < 1e-4)
								and
			                    (p['corrupt_type_warm_start'] == 1 #filter out repetitive warm start data
								or abs(p['corrupt_prob_warm_start']) > 1e-4))
	else:
		fltr_inter_gt = lambda p: False

	if mod.ws_gt_on:
		fltr_ws_gt = lambda p: ((p['corrupt_type_warm_start'] == 1 #noiseless for warm start data
							and abs(p['corrupt_prob_warm_start']) < 1e-4)
							and
		                    (p['corrupt_type_interaction'] == 1 #filter out repetitive interaction data
							or abs(p['corrupt_prob_interaction']) > 1e-4))
	else:
		fltr_ws_gt = lambda p: False

	return (fltr_inter_gt, fltr_ws_gt)

def get_params_alg(mod, prm_com_ws_gt, prm_com_inter_gt, prm_choices_lbd):
	# Algorithm parameters construction
	if mod.algs_on:
		# Algorithms for supervised validation
		if mod.ws_gt_on:
			prm_ws_gt = \
			[
				 [
			  	 	{'algorithm':'AwesomeBandits',
					 'warm_start_update': True,
					 'interaction_update': True,
					 'warm_start_type': 1,
					 'lambda_scheme': 2,
					 'weighting_scheme': 2}
				 ],
				 [
				 	{'validation_method':2},
					{'validation_method':3}
				 ]
		    ]
		else:
			prm_ws_gt = [[]]

		if mod.inter_gt_on:
			prm_inter_gt = \
			[
				 [
			  	 	{'algorithm':'AwesomeBandits',
					 'warm_start_update': True,
					 'interaction_update': True,
					 'warm_start_type': 1,
					 'weighting_scheme': 1,
                     'validation_method': 1,
 					 # for time-varying epsilon
					 #'eps_t':0.1,
					 'lambda_scheme': 2,
					 # for fixed epsilon
					 #'epsilon': 0.05,
					 #'lambda_scheme': 4
					 }
				 ],
			]
		else:
			prm_inter_gt = [[]]

		prm_algs_ws_gt = param_cartesian_multi([prm_com_ws_gt] + [prm_choices_lbd] + prm_ws_gt)
		prm_algs_inter_gt = param_cartesian_multi([prm_com_inter_gt] + [prm_choices_lbd] + prm_inter_gt)
		prm_algs = prm_algs_ws_gt + prm_algs_inter_gt
	else:
		prm_algs = []
	return prm_algs


def get_params_opt(mod):
	# Optimal baselines parameter construction
	if mod.optimal_on:
		prm_optimal = \
		[
			{'algorithm': 'Optimal',
			 'fold': 1,
			 'corrupt_type_warm_start':1,
			 'corrupt_prob_warm_start':0.0,
			 'corrupt_type_interaction':1,
			 'corrupt_prob_interaction':0.0}
	    ]
	else:
		prm_optimal = []
	return prm_optimal

def get_params_maj(mod):
	if mod.majority_on:
		prm_majority = \
		[
			{'algorithm': 'Most-Freq',
			 'fold': 1,
			 'corrupt_type_warm_start':1,
			 'corrupt_prob_warm_start':0.0,
			 'corrupt_type_interaction':1,
			 'corrupt_prob_interaction':0.0}
		]
	else:
		prm_majority = []
	return prm_majority


def get_params_baseline(mod, prm_com, prm_com_noeps):
	# Baseline parameters construction
	if mod.baselines_on:
		prm_sup_only_basic = [[]]
		if mod.sup_only_on:
			prm_sup_only_basic[0] += \
				[
					#Sup-Only
					#TODO: make sure the epsilon=0 setting propagates
			 		{'algorithm':'Sup-Only',
					 'warm_start_type': 1,
					 'warm_start_update': True,
					 'interaction_update': False,
					 'epsilon': 0.0
					 }
				]

		prm_oth_baseline_basic = [[]]
		if mod.band_only_on:
			prm_oth_baseline_basic[0] += \
				[
					#Band-Only
	 		 		{'algorithm':'Bandit-Only',
					 'warm_start_type': 1,
	 				 'warm_start_update': False,
	 				 'interaction_update': True
					 }
				]

		if mod.sim_bandit_on:
			prm_oth_baseline_basic[0] += \
				[
					#Sim-Bandit
					{'algorithm':'Sim-Bandit',
					 'warm_start_type': 2,
					 'warm_start_update': True,
				     'interaction_update': True,
					 'lambda_scheme': 1
					 }
				]

		#Sim-Bandit with only warm-start update
		#(ideally, we need to set epsilon != 0 for the ws stage and epsilon = 0
		#for the interaction stage, and it seems that we need to change warm_cb.cc:
		#if interaction_update = False then we should use csoaa predict for interaction stage
		#{'algorithm':'Sim-Bandit-Freeze',
        #'warm_start_type': 2,
		# 'warm_start_update': True,
		# 'interaction_update': False}

		prm_baseline_const = \
		[
			[
				{'weighting_scheme':1,
				 'choices_lambda':1}
			]
		]

		prm_baseline = param_cartesian_multi([prm_com_noeps] + prm_baseline_const + prm_sup_only_basic) \
		+ param_cartesian_multi([prm_com] + prm_baseline_const + prm_oth_baseline_basic)
	else:
		prm_baseline = []

	return prm_baseline

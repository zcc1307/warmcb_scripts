TMPLTimport subprocess
from itertools import product
import os
import math
import argparse
import time
import glob
import re
from collections import OrderedDict


RESULT_TMPLT = \
	[
	('fold', 'fd', 0),
	('data', 'dt', ''),
	('dataset', 'ds', ''),
	('num_classes','nc', 0),
	('total_size', 'ts', 0),
	('majority_size','ms', 0),
	('corrupt_type_warm_start', 'ctws', 0),
	('corrupt_prob_warm_start', 'cpws', 0.0),
	('corrupt_type_interaction', 'cti', 0),
	('corrupt_prob_interaction', 'cpi', 0.0),
	('warm_start_multiplier','wsm',1),
	('warm_start', 'ws', 0),
	('warm_start_type', 'wst', 0),
	('interaction', 'bs', 0),
	('inter_ws_size_ratio', 'iwsr', 0),
	('algorithm', 'alg', ''),
	('adf_on', 'ao', True),
	('epsilon', 'eps', 0.0),
	('loss0', 'l0', 0.0),
	('loss1', 'l1', 0.0),
	('cb_type', 'cbt', 'mtr'),
	('validation_method', 'vm', 0),
	('weighting_scheme', 'wts', 0),
	('lambda_scheme', 'ls', 0),
	('choices_lambda', 'cl', 0),
	('warm_start_update', 'wsu', True),
	('interaction_update', 'iu', True),
	('learning_rate', 'lr', 0.0),
	('avg_error', 'ae', 0.0),
	('actual_variance', 'av', 0.0),
	('ideal_variance', 'iv', 0.0),
	('last_lambda', 'll', 0.0),
	]
	#('optimal_approx', 'oa', False),
	#('majority_approx', 'ma', False),
SUMMARY_TMPLT = \
	[
	'fold',
	'data',
	'num_classes',
	'total_size',
	'majority_size',
	'corrupt_type_warm_start',
	'corrupt_prob_warm_start',
	'corrupt_type_interaction',
	'corrupt_prob_interaction',
	'warm_start',
	'interaction',
	'inter_ws_size_ratio',
	'algorithm',
	'adf_on',
	'epsilon',
	'loss0',
	'loss1',
	'validation_method',
	'weighting_scheme',
	'lambda_scheme',
	'choices_lambda',
	'learning_rate',
	'avg_error',
	'actual_variance',
	'ideal_variance',
	'last_lambda'
	]

VW_OUTFILE_NAME_TMPLT = \
	['dataset',
	 'fold',
	 'lambda_scheme',
	 'validation_method',
	 'warm_start_multiplier',
	 'corrupt_prob_interaction',
	 'corrupt_prob_warm_start',
	 'corrupt_type_interaction',
	 'corrupt_type_warm_start',
 	 'warm_start_update',
 	 'interaction_update',
	 'warm_start_type',
	 'choices_lambda',
	 'weighting_scheme',
	 'cb_type',
	 'learning_rate',
	 'adf_on',
	 'epsilon',
	 'loss0',
	 'loss1',
	 'algorithm']
 	 #'optimal_approx',
 	 #'majority_approx',

VW_RUN_TMPLT_OPT = \
 	[('data',''),
	 ('progress',2.0),
	 ('passes',0),
	 ('oaa',0),
	 ('cache_file','')]

VW_RUN_TMPLT_MAJ = \
	 [('data',''),
	  ('progress',2.0),
	  ('warm_cb',0),
	  ('warm_start',0),
	  ('interaction',0)]

VW_RUN_TMPLT_WARMCB = \
 	[('data',''),
	 ('warm_cb',0),
	 ('cb_type','mtr'),
	 ('warm_start',0),
	 ('interaction',0),
	 ('corrupt_type_interaction',0),
	 ('corrupt_prob_interaction',0.0),
	 ('corrupt_type_warm_start',0),
	 ('corrupt_prob_warm_start',0.0),
	 ('warm_start_update',True),
	 ('interaction_update',True),
	 ('choices_lambda',0),
	 ('lambda_scheme',1),
	 ('warm_start_type',1),
	 ('overwrite_label',1),
	 ('validation_method',1),
	 ('weighting_scheme',1),
	 ('learning_rate',0.5),
	 ('epsilon', 0.05),
	 ('loss0', 0),
	 ('loss1', 0),
	 ('progress',2.0)]

#VW_PROGRESS_PATTERN = \
# '\d+\s\d+\.\d+\n' +
# '\d+\.\d+\s+\d+\.\d+\s+\d+\s+\d+\.\d+\s+[a-zA-Z0-9]+\s+[a-zA-Z0-9]+\s+\d+.*'
#float_pat = '\d+\.\d+'
#int_pat = '\d+'
foi_pat = '\d+(\.\d+)?'
label_pat = '[a-zA-Z0-9]+'
gen_pat = '[a-zA-Z0-9\.]+'
VW_PROGRESS_PATTERN = '('+gen_pat+'\s+'+gen_pat+'\s+'+gen_pat+'\s+'+gen_pat+'\n'+ \
			'\d+\.\d+\s+\d+\.\d+\s+\d+\s+\d+\.\d+\s+[a-zA-Z0-9]+\s+[a-zA-Z0-9]+\s+\d+\n)'

#VW_PROGRESS_PATTERN = '(\n'+foi_pat+'\s+'+foi_pat+'\s+'+foi_pat+'\s+'+foi_pat+'\n'+ \
                  #foi_pat+'\s+'+foi_pat+'\s+'+foi_pat+'\s+'+foi_pat+'\s+'+label_pat+'\s+'+label_pat+'\s+'+foi_pat+'\n)'

VW_RESULT_TMPLT = \
	{
	'interaction': 0,
	'inter_ws_size_ratio': 0,
	'avg_error': 0.0,
	'actual_variance': 0.0,
	'ideal_variance': 0.0,
	'last_lambda':0.0
	}


class model:
	def __init__(self):
		# Setting up argument-independent learning parameters in the constructor
		self.baselines_on = True
		self.sup_only_on = True
		self.band_only_on = True
		self.sim_bandit_on = False

		self.algs_on = True
		self.optimal_on = False
		self.majority_on = False

		self.ws_gt_on = True
		self.inter_gt_on = False

		#self.num_checkpoints = 400
		self.num_checkpoints = 200

		# use fractions instead of absolute numbers
		#self.ws_multipliers = [pow(2,i) for i in range(4)]
		self.ws_multipliers = [pow(2,i) for i in range(1)]

		self.choices_cb_type = ['mtr']
		#mod.choices_choices_lambda = [2,4,8]
		self.choices_choices_lambda = [2,8]

		self.choices_cor_type_ws = [1,2,3]
		self.choices_cor_prob_ws = [0.0,0.5,1.0]
		#self.choices_cor_type_ws = [1]
		#self.choices_cor_prob_ws = [0.0]

		self.choices_cor_type_inter = [1]
		self.choices_cor_prob_inter = [0.0]
		#self.choices_cor_prob_inter = [0.0, 0.125, 0.25, 0.5]

		self.choices_loss_enc = [(0, 1)]
		#self.choices_cor_type_inter = [1,2]
		#self.choices_cor_prob_inter = [0.0,0.5]

		self.choices_epsilon = [0.05]
		#self.choices_epsilon = [0.0125, 0.025, 0.05, 0.1]
		#self.epsilon_on = True
		#self.lr_template = [0.1, 0.03, 0.3, 0.01, 1.0, 0.003, 3.0, 0.001, 10.0, 0.0003, 30.0, 0.0001, 100.0]
		self.choices_adf = [True]
		#self.critical_size_ratios = [368 * pow(2, -i) for i in range(8) ]
		self.critical_size_ratios = [184 * pow(2, -i) for i in range(8) ]

def gen_lr(n):
	m = math.floor(n / 4.0)
	if n % 4 == 0:
		return 0.1 * pow(10, m)
	if n % 4 == 1:
		return 0.03 * pow(10, -m)
	if n % 4 == 2:
		return 0.3 * pow(10, m)
	if n % 4 == 3:
		return 0.01 * pow(10, -m)

def analyze_vw_out_maj_opt(mod):
	vw_result = VW_RESULT_TMPLT.copy()
	if mod.param['algorithm'] == 'Optimal':
		# this condition is for computing the optimal error
		vw_result['avg_error'] = avg_error(mod)
	else:
		# this condition is for computing the majority error
		err =  1 - float(mod.param['majority_size']) / mod.param['total_size']
		vw_result['avg_error'] = float('%0.5f' % err)
	return vw_result

def analyze_vw_out(mod):
	vw_run_results = []

	if mod.param['algorithm'] == 'Most-Freq' or mod.param['algorithm'] == 'Optimal':
		vw_run_results.append(analyze_vw_out_maj_opt(mod))
		return vw_run_results

	f = open(mod.vw_output_filename, 'r')
	vw_output_text = f.read()
	rgx = re.compile(VW_PROGRESS_PATTERN, flags=re.M)
	matched = rgx.findall(vw_output_text)

	for mat in matched:
		line = mat
		#print mat
		s = line.split()
		if len(s) >= 12:
			s = s[:11]

		counter_new, last_lambda, actual_var, ideal_var, \
		avg_loss, last_loss, counter, weight, curr_label, curr_pred, curr_feat = s
		inter_effective = int(float(weight))

		for ratio in mod.critical_size_ratios:
			if inter_effective >= (1 - 1e-7) * mod.param['warm_start'] * ratio and \
			inter_effective <= (1 + 1e-7) * mod.param['warm_start'] * ratio:
				vw_result = VW_RESULT_TMPLT.copy()
				vw_result['interaction'] = inter_effective
				vw_result['inter_ws_size_ratio'] = ratio
				vw_result['avg_error'] = float(avg_loss)
				vw_result['actual_variance'] = float(actual_var)
				vw_result['ideal_variance'] = float(ideal_var)
				vw_result['last_lambda'] = float(last_lambda)
				vw_run_results.append(vw_result)
	f.close()

	#if len(vw_run_results) >= 1:
	#	print mod.param['warm_start']
	#	print vw_run_results
	#raw_input('..')
	return vw_run_results


def gen_vw_command(mod):
	mod.vw_options = format_setting(mod.vw_template, mod.param)
	vw_options_list = []
	for k, v in mod.vw_options.iteritems():
		vw_options_list.append('--'+str(k))
		vw_options_list.append(str(v))
	cmd = intersperse([mod.vw_path]+vw_options_list, ' ')
	return cmd

def gen_vw_options(mod):
	if mod.param['algorithm'] == 'Optimal':
		# Fully supervised on full dataset
		mod.vw_template = OrderedDict(VW_RUN_TMPLT_OPT)
		mod.param['passes'] = 5
		mod.param['oaa'] = mod.param['num_classes']
		mod.param['cache_file'] = mod.param['data'] + '.cache'
	elif mod.param['algorithm'] == 'Most-Freq':
		# Compute majority error; basically we would like to skip vw running as fast as possible
		mod.vw_template = OrderedDict(VW_RUN_TMPLT_MAJ)
		mod.param['warm_cb'] = mod.param['num_classes']
		mod.param['warm_start'] = 0
		mod.param['interaction'] = 0
	else:
		# General CB
		mod.vw_template = OrderedDict(VW_RUN_TMPLT_WARMCB)
		mod.param['warm_start'] = mod.param['warm_start_multiplier'] * mod.param['progress']
		mod.param['interaction'] = mod.param['total_size'] - mod.param['warm_start']
		mod.param['warm_cb'] = mod.param['num_classes']
		mod.param['overwrite_label'] = mod.param['majority_class']

		if mod.param['adf_on'] is True:
			mod.param['cb_explore_adf'] = ' '
			mod.vw_template['cb_explore_adf'] = ' '
		else:
			mod.param['cb_explore'] = mod.param['num_classes']
			mod.vw_template['cb_explore'] = 0


def execute_vw(mod):
	gen_vw_options(mod)
	cmd = gen_vw_command(mod)
	print cmd
	f = open(mod.vw_output_filename, 'w')
	f.write(cmd+'\n')
	process = subprocess.Popen(cmd, shell=True, stdout=f, stderr=f)
	process.wait()
	f.close()

def intersperse(l, ch):
	s = ''
	for item in l:
		s += str(item)
		s += ch
	return s

def param_to_str(param):
	param_list = [ str(k)+'='+str(v) for k,v in param.iteritems() ]
	return intersperse(param_list, ',')

def replace_keys(dic, simplified_keymap):
	dic_new = OrderedDict()
	for k, v in dic.iteritems():
		dic_new[simplified_keymap[k]] = v
	return dic_new

def get_vw_out_name(mod):
	# step 1: use the above as a template to filter out irrelevant parameters
	param_formatted = format_setting(mod.vw_out_tmplt, mod.param)
	# step 2: replace the key names with the simplified names
	param_simplified = replace_keys(param_formatted, mod.simp_map)
	return param_to_str(param_simplified)

def run_single_expt(mod):
	mod.param['data'] = mod.ds_path + str(mod.param['fold']) + '/' + mod.param['dataset']
	mod.param['total_size'] = get_num_lines(mod.param['data'])
	mod.param['num_classes'] = get_num_classes(mod.param['data'])
	mod.param['majority_size'], mod.param['majority_class'] = get_majority_class(mod.param['data'])
	mod.param['progress'] = int(math.ceil(float(mod.param['total_size']) / float(mod.num_checkpoints)))
	mod.vw_output_dir = mod.results_path + remove_suffix(mod.param['data']) + '/'
	mod.vw_output_filename = mod.vw_output_dir + get_vw_out_name(mod) + '.txt'

	execute_vw(mod)
	vw_run_results = collect_stats(mod)
	for vw_result in vw_run_results:
		result_combined = merge_two_dicts(mod.param, vw_result)
		result_formatted = format_setting(mod.sum_tmplt, result_combined)
		write_result(mod, result_formatted)

# The following function is a "template filling" function
# Given a template, we use the setting dict to fill it as much as possible
def format_setting(template, setting):
	formatted = template.copy()
	for k, v in setting.iteritems():
		if k in template.keys():
			formatted[k] = v
	return formatted

def write_row(mod, row):
	summary_file = open(mod.summary_file_name, 'a')
	summary_file.write( intersperse(row, '\t') + '\n')
	summary_file.close()

def write_result(mod, result):
	write_row(mod, result.values())

def write_summary_header(mod):
	write_row(mod, mod.result_template.keys())

def ds_files(ds_path):
	prevdir = os.getcwd()
	os.chdir(ds_path)
	dss = sorted(glob.glob('*.vw.gz'))
	#dss = [ds_path+ds for ds in dss]
	os.chdir(prevdir)
	return dss

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
	print param_name, result
	return result


def params_per_task(mod):
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
	prm_choices_eps = dictify('epsilon', mod.choices_epsilon)
	prm_adf_on = dictify('adf_on', mod.choices_adf)
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
	print 'The total number of VW commands to run is: ', len(prm_all)
	for row in prm_all:
		print row
	return get_params_task(prm_all)

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
					 'lambda_scheme': 4,
					 'weighting_scheme': 1}
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
	 				 'interaction_update': True}
				]

		if mod.sim_bandit_on:
			prm_oth_baseline_basic[0] += \
				[
					#Sim-Bandit
					{'algorithm':'Sim-Bandit',
					 'warm_start_type': 2,
					 'warm_start_update': True,
				     'interaction_update': True,
					 'lambda_scheme': 1}
				]

		#Sim-Bandit with only warm-start update
		#(ideally, we need to set epsilon != 0 for the ws stage and epsilon = 0
		#for the interaction stage, and it seems that we need to change warm_cb.cc:
		#if interaction_update = False then we should use csoaa predict for interaction stage
		#{'warm_start_type': 2,
		# 'warm_start_update': True,
		# 'interaction_update': False}

		prm_baseline_const = \
		[
			[
				{'weighting_scheme':1,
				 'adf_on':True,
				 'lambda_scheme':3,
				 'choices_lambda':1}
			]
		]

		prm_baseline = param_cartesian_multi([prm_com_noeps] + prm_baseline_const + prm_sup_only_basic) \
		+ param_cartesian_multi([prm_com] + prm_baseline_const + prm_oth_baseline_basic)
	else:
		prm_baseline = []

	return prm_baseline


def get_params_task(params_all):
	params_task = []
	for i in range(len(params_all)):
		if (i % mod.num_tasks == mod.task_id):
			params_task.append(params_all[i])
	return params_task

def get_num_lines(dataset_name):
	num_lines = subprocess.check_output(('zcat ' + dataset_name + ' | wc -l'), shell=True)
	return int(num_lines)

def get_num_classes(ds):
	# could be a bug for including the prefix..
	did, n_actions = os.path.basename(ds).split('.')[0].split('_')[1:]
	did, n_actions = int(did), int(n_actions)
	return n_actions

def get_majority_class(dataset_name):
	maj_class_str = subprocess.check_output(('zcat '+ dataset_name +' | cut -d \' \' -f 1 | sort | uniq -c | sort -r -n | head -1 | xargs '), shell=True)
	maj_size, maj_class = maj_class_str.split()
	return int(maj_size), int(maj_class)

def avg_error(mod):
	return vw_output_extract(mod, 'average loss')

def actual_var(mod):
	return vw_output_extract(mod, 'Measured average variance')

def ideal_var(mod):
	return vw_output_extract(mod, 'Ideal average variance')

def vw_output_extract(mod, pattern):
	#print mod.vw_output_filename
	vw_output = open(mod.vw_output_filename, 'r')
	vw_output_text = vw_output.read()
	#print vw_output_text
	#rgx_pattern = '^'+pattern+' = (.*)(|\sh)\n.*$'
	#print rgx_pattern
	rgx_pattern = '.*'+pattern+' = ([\d]*.[\d]*)( h|)\n.*'
	rgx = re.compile(rgx_pattern, flags=re.M)

	errs = rgx.findall(vw_output_text)
	if not errs:
		avge = 0
	else:
		#print errs
		avge = float(errs[0][0])

	vw_output.close()
	return avge

def complete_header(simp_header):
	simplified_keymap = OrderedDict([ (item[1], item[0]) for item in RESULT_TMPLT ])
	return simplified_keymap[simp_header]


def main_loop(mod):
	mod.summary_file_name = mod.results_path+str(mod.task_id)+'of'+str(mod.num_tasks)+'.sum'
	mod.full_tmplt = OrderedDict([ (item[0], item[2]) for item in RESULT_TMPLT ])
	mod.simp_map = OrderedDict([ (item[0], item[1]) for item in RESULT_TMPLT ])
	mod.sum_tmplt = OrderedDict([ (item, mod.full_tmplt[item]) for item in SUMMARY_TMPLT ])
	mod.vwout_tmplt = OrderedDict([ (item, mod.full_tmplt[item]) for item in VW_OUTFILE_NAME_TMPLT ])
	write_summary_header(mod)
	for mod.param in mod.config_task:
		run_single_expt(mod)

def create_dir(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)
		import stat
		os.chmod(dir, os.stat(dir).st_mode | stat.S_IWOTH)

def remove_suffix(filename):
	return os.path.basename(filename).split('.')[0]

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='vw job')
	parser.add_argument('task_id', type=int, help='task ID, between 0 and num_tasks - 1')
	parser.add_argument('num_tasks', type=int)
	parser.add_argument('--results_dir', default='../../../figs/')
	parser.add_argument('--ds_dir', default='../../../vwshuffled/')
	parser.add_argument('--num_learning_rates', type=int, default=1)
	parser.add_argument('--num_datasets', type=int, default=-1)
	parser.add_argument('--num_folds', type=int, default=1)

	args = parser.parse_args()
	flag_dir = args.results_dir + 'flag/'

	mod = model()
	mod.num_tasks = args.num_tasks
	mod.task_id = args.task_id
	mod.vw_path = '../../vowpal_wabbit/vowpalwabbit/vw'
	mod.ds_path = args.ds_dir
	mod.results_path = args.results_dir
	print 'reading dataset files..'
	#TODO: this line specifically for multiple folds
	#Need a systematic way to detect subfolder names
	mod.dss = ds_files(mod.ds_path + '1/')

	print len(mod.dss)

	if args.num_datasets == -1 or args.num_datasets > len(mod.dss):
		pass
	else:
		mod.dss = mod.dss[:args.num_datasets]

	#print mod.dss

	if args.task_id == 0:
		# Compile vw in one of the subfolders
		#process = subprocess.Popen('make -C .. clean; make -C ..', shell=True, stdout=f, stderr=f)
		#subprocess.check_call(cmd, shell=True)
		#process.wait()

		# To avoid race condition of writing to the same file at the same time
		create_dir(args.results_dir)

		# This is specifically designed for teamscratch, as accessing a folder
		# with a huge number of result files can be super slow. Hence, we create a
		# subfolder for each dataset to alleviate this.
		for ds in mod.dss:
			ds_no_suffix = remove_suffix(ds)
			create_dir(args.results_dir + ds_no_suffix + '/')

		create_dir(flag_dir)
	else:
		# may still have the potential of race condition on those subfolders (if
		# we have a lot of datasets to run and the datasets are small)
		while not os.path.exists(flag_dir):
			time.sleep(1)

	if args.num_learning_rates <= 0:
		#mod.learning_rates = [gen_lr(0)]
		mod.learning_rates = [0.5]
	else:
		mod.learning_rates = [gen_lr(i) for i in range(args.num_learning_rates)]
	#mod.folds = range(1,11)
	mod.folds = range(1, args.num_folds+1)

	#mod.dss = ["ds_223_63.vw.gz"]
	#mod.dss = mod.dss[:5]

	print 'generating tasks..'
	# here, we are generating the task specific parameter settings
	# by first generate all parameter setting and pick every num_tasks of them
	mod.config_task = params_per_task(mod)
	print 'task ' + str(mod.task_id) + ' of ' + str(mod.num_tasks) + ':'
	print len(mod.config_task)

	#print mod.ds_task
	# we only need to vary the warm start fraction, and there is no need to vary the bandit fraction,
	# as each run of vw automatically accumulates the bandit dataset
	main_loop(mod)

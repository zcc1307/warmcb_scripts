import subprocess
from itertools import product
import os
import math
import argparse
import time
import glob
import re
from collections import OrderedDict
from params_gen import get_all_params, merge_two_dicts
from vw_commands_const import VW_RUN_TMPLT_OPT, VW_RUN_TMPLT_MAJ, VW_RUN_TMPLT_WARMCB, VW_PROGRESS_PATTERN, VW_RESULT_TMPLT, FULL_TMPLT, SIMP_MAP, SUM_TMPLT, VW_OUT_TMPLT

class model:
	def __init__(self):
		# Setting up argument-independent learning parameters in the constructor
		self.baselines_on = True
		self.sup_only_on = True
		self.band_only_on = True
		self.sim_bandit_on = True

		self.algs_on = True
		self.optimal_on = True
		self.majority_on = True

		self.ws_gt_on = False
		self.inter_gt_on = True

		#self.num_checkpoints = 400
		self.num_checkpoints = 200

		# use fractions instead of absolute numbers
		#self.ws_multipliers = [pow(2,i) for i in range(4)]
		self.ws_multipliers = [pow(2,i) for i in range(4)]

		self.choices_cb_type = ['mtr']
		#mod.choices_choices_lambda = [2,4,8]
		self.choices_choices_lambda = [2,8]

		self.choices_cor_type_ws = [1,2,3]
		self.choices_cor_prob_ws = [0.0,0.25,0.5,1.0]
		#self.choices_cor_type_ws = [1]
		#self.choices_cor_prob_ws = [0.0]

		self.choices_cor_type_inter = [1]
		self.choices_cor_prob_inter = [0.0]
		#self.choices_cor_prob_inter = [0.0, 0.125, 0.25, 0.5]

		self.choices_loss_enc = [(0, 1)]
		#self.choices_cor_type_inter = [1,2]
		#self.choices_cor_prob_inter = [0.0,0.5]

		#self.choices_epsilon = [0.05]
		self.choices_epsilon = []
		self.choices_eps_t = [0.1]
		#self.choices_epsilon = [0.0125, 0.025, 0.05, 0.1]
		#self.epsilon_on = True
		#self.lr_template = [0.1, 0.03, 0.3, 0.01, 1.0, 0.003, 3.0, 0.001, 10.0, 0.0003, 30.0, 0.0001, 100.0]
		self.choices_adf = [True]
		#self.critical_size_ratios = [368 * pow(2, -i) for i in range(8) ]
		self.critical_size_ratios = [184 * pow(2, -i) for i in range(7) ]

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
	return vw_run_results


def gen_vw_command(mod):
	mod.vw_options = format_setting(mod.vw_template, mod.param)
	vw_options_list = []
	for k, v in mod.vw_options.items():
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

		if 'eps_t' in mod.param:
			mod.vw_template['eps_t'] = 1.0
		else:
			mod.vw_template['epsilon'] = 0.0


def execute_vw(mod):
	gen_vw_options(mod)
	cmd = gen_vw_command(mod)
	print(cmd)
	f = open(mod.vw_output_filename, 'w')
	f.write(cmd+'\n')
	f.close()
	f = open(mod.vw_output_filename, 'a')
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
	param_list = [ str(k)+'='+str(v) for k,v in param.items() ]
	return intersperse(param_list, ',')

def replace_keys(dic, simplified_keymap):
	dic_new = OrderedDict()
	for k, v in dic.items():
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
	vw_run_results = analyze_vw_out(mod)
	for vw_result in vw_run_results:
		result_combined = merge_two_dicts(mod.param, vw_result)
		result_formatted = format_setting(mod.sum_tmplt, result_combined)
		write_result(mod, result_formatted)

# The following function is a "template filling" function
# Given a template, we use the setting dict to fill it as much as possible
def format_setting(template, setting):
	formatted = template.copy()
	for k, v in setting.items():
		if k in template.keys():
			formatted[k] = v
	return formatted

def write_row(mod, row, mode):
	summary_file = open(mod.summary_file_name, mode)
	if mode == 'a':
		summary_file.truncate()
	summary_file.write( intersperse(row, '\t') + '\n')
	summary_file.close()

def write_result(mod, result):
	write_row(mod, result.values(), 'a')

def write_summary_header(mod):
	write_row(mod, mod.sum_tmplt.keys(), 'w')

def ds_files(ds_path):
	prevdir = os.getcwd()
	os.chdir(ds_path)
	dss = sorted(glob.glob('*.vw.gz'))
	#dss = [ds_path+ds for ds in dss]
	os.chdir(prevdir)
	return dss

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

#def complete_header(simp_header):
#	simplified_keymap = OrderedDict([ (item[1], item[0]) for item in RESULT_TMPLT ])
#	return simplified_keymap[simp_header]


def main_loop(mod):
	mod.summary_file_name = mod.results_path+str(mod.task_id)+'of'+str(mod.num_tasks)+'.sum'
	mod.full_tmplt = FULL_TMPLT
	mod.simp_map = SIMP_MAP
	mod.sum_tmplt = SUM_TMPLT
	mod.vw_out_tmplt = VW_OUT_TMPLT
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
	print('reading dataset files..')
	#TODO: this line specifically for multiple folds
	#Need a systematic way to detect subfolder names
	mod.dss = ds_files(mod.ds_path + '1/')

	print(len(mod.dss))

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

	print('generating tasks..')
	# here, we are generating the task specific parameter settings
	# by first generate all parameter setting and pick every num_tasks of them
	all_params = get_all_params(mod)
	mod.config_task = get_params_task(all_params)
	print('task ', str(mod.task_id), ' of ', str(mod.num_tasks), ':')
	print(len(mod.config_task))

	#print mod.ds_task
	# we only need to vary the warm start fraction, and there is no need to vary the bandit fraction,
	# as each run of vw automatically accumulates the bandit dataset
	main_loop(mod)

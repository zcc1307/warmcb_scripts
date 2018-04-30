import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import pylab
from itertools import product
import os
import math
import argparse
import time
import glob
import re


class model:
	def __init__(self):
		self.no_bandit = False
		self.no_supervised = False

def collect_stats(mod):

	vw_output_filename = mod.vw_output_filename
	# using progress parameter
	# num_rows = mod.bandit / mod.progress
	#print vw_output_filename

	#avg_error_value = avg_error(mod)
	mod.actual_var = actual_var(mod)
	mod.ideal_var = ideal_var(mod)

	avg_loss = []
	last_loss = []
	wt = []
	end_table = False

	f = open(vw_output_filename, 'r')
	#linenumber = 0
	i = 0
	for line in f:
		vw_progress_pattern = '\d+\.\d+\s+\d+\.\d+\s+\d+\s+\d+\.\d+\s+[a-zA-Z0-9]+\s+[a-zA-Z0-9]+\s+\d+'
		matchobj = re.match(vw_progress_pattern, line)

		if matchobj:
			avg_loss_str, last_loss_str, counter_str, weight_str, curr_label_str, \
			curr_pred_str, curr_feat_str = line.split()

			avg_loss.append(float(avg_loss_str))
			last_loss.append(float(last_loss_str))
			wt.append(float(weight_str))

			mod.avg_loss = float(avg_loss_str)
			mod.bandit = float(weight_str)

			for mod.ratio in mod.critical_size_ratios:
				if mod.bandit >= 0.99 * mod.warm_start * mod.ratio and \
				mod.bandit <= 1.01 * mod.warm_start * mod.ratio:
					record_result(mod)


		#linenumber += 1

	f.close()

	#if len(avg_loss) == 0:
	#	avg_loss = [0]
	#	last_loss = [0]
	#	wt = [0]
	#return avg_loss, last_loss, wt

def record_result(mod):
	problem_params_trailer = [mod.bandit, mod.ratio]
	config_name = disperse(mod.problem_params + problem_params_trailer + mod.alg_params, ' ')

	list_results = [mod.avg_loss, mod.actual_var, mod.ideal_var]
	result = disperse(list_results, ' ')

	summary_file = open(mod.summary_file_name, 'a')
	summary_file.write(config_name + ' ' + result + '\n')
	summary_file.close()


def execute_vw(mod):

	alg_option = ' '
	if mod.adf_on:
		alg_option += ' --cb_explore_adf '
	else:
		alg_option += ' --cb_explore ' + str(mod.num_classes) + ' '

	if mod.cover_on:
		alg_option += ' --cover 5 --psi 0.01 --nounif '
		#mod.cb_type = 'dr'
	if mod.epsilon_on:
		alg_option += ' --epsilon ' + str(mod.epsilon) + ' '
	if mod.no_bandit:
		alg_option += ' --no_bandit '
	if mod.no_supervised:
		alg_option += ' --no_supervised '
	#if mod.no_exploration:
	#	alg_option += ' --epsilon 0.0 '
	#if mod.cb_type == 'mtr':
	#	mod.adf_on = True;

	cmd_vw = mod.vw_path + ' --cbify ' + str(mod.num_classes) + ' --cb_type ' + str(mod.cb_type) + ' --warm_start ' + str(mod.warm_start) + ' --bandit ' + str(mod.bandit) + ' --choices_lambda ' + str(mod.choices_lambda) + alg_option + ' --progress ' + str(mod.progress) \
	 + ' -d ' + mod.ds_path + mod.dataset \
	 + ' --corrupt_type_supervised ' + str(mod.corrupt_type_supervised) \
	 + ' --corrupt_prob_supervised ' + str(mod.corrupt_prob_supervised) \
	 + ' --corrupt_type_bandit ' + str(mod.corrupt_type_bandit) \
	 + ' --corrupt_prob_bandit ' + str(mod.corrupt_prob_bandit) \
	 + ' --validation_method ' + str(mod.validation_method) \
	 + ' --weighting_scheme ' + str(mod.weighting_scheme) \
	 + ' --lambda_scheme ' + str(mod.lambda_scheme)

	cmd = cmd_vw
	print cmd

	f = open(mod.vw_output_filename, 'w')
	process = subprocess.Popen(cmd, shell=True, stdout=f, stderr=f)
	#subprocess.check_call(cmd, shell=True)
	process.wait()
	f.close()

'''
def plot_errors(mod):
	#avg_loss, last_loss, wt =
	if mod.plot_flat:
		# for supervised only, we simply plot a horizontal line using the last point
		len_avg_loss = len(avg_loss)
		avg_loss = avg_loss[len_avg_loss-1]
		avg_loss = [avg_loss for i in range(len_avg_loss)]

	#line = plt.plot(wt, avg_loss, mod.plot_color, label=(mod.plot_label))
	avg_error_value = avg_error(mod)
	actual_var_value = actual_var(mod)
	ideal_var_value = ideal_var(mod)

	return avg_error_value, actual_var_value, ideal_var_value
'''

def disperse(l, ch):
	s = ''
	for item in l:
		s += str(item)
		s += ch
	return s


def gen_comparison_graph(mod):

	mod.num_lines = get_num_lines(mod.ds_path+mod.dataset)
	mod.progress = int(math.ceil(float(mod.num_lines) / float(mod.num_checkpoints)))
	mod.warm_start = mod.warm_start_multiplier * mod.progress
	mod.bandit = mod.num_lines - mod.warm_start
	mod.num_classes = get_num_classes(mod.dataset)

	mod.problem_params = [mod.dataset, mod.num_classes, mod.num_lines, \
	mod.corrupt_type_supervised, mod.corrupt_prob_supervised, \
	mod.corrupt_type_bandit, mod.corrupt_prob_bandit, \
	mod.warm_start]

	mod.alg_params = [ mod.cb_type, \
	mod.validation_method, mod.weighting_scheme, \
	mod.lambda_scheme, mod.choices_lambda, \
	mod.no_supervised, mod.no_bandit]

	mod.vw_output_filename = mod.results_path + disperse(mod.problem_params+mod.alg_params, '_') + '.txt'

	#plot_errors(mod)
	execute_vw(mod)
	collect_stats(mod)

	print('')

def ds_files(ds_path):
	prevdir = os.getcwd()
	os.chdir(ds_path)
	dss = sorted(glob.glob('*.vw.gz'))
	os.chdir(prevdir)
	return dss


def get_num_classes(ds):
	did, n_actions = os.path.basename(ds).split('.')[0].split('_')[1:]
	did, n_actions = int(did), int(n_actions)
	return n_actions


def ds_per_task(mod):
	# put dataset name to the last coordinate so that the task workloads tend to be
	# allocated equally
	config_baselines_raw = list(product(mod.choices_corrupt_type_supervised, mod.choices_corrupt_prob_supervised, mod.choices_cb_types, mod.dss, mod.warm_start_multipliers, [1], [False, True], [False, True]))

	config_baselines = filter(lambda (x1, x2, x3, x4, x5, x6, x7, x8): x7 == True or x8 == True, config_baselines_raw)


	config_algs = list(product(mod.choices_corrupt_type_supervised, mod.choices_corrupt_prob_supervised, mod.choices_cb_types, mod.dss, mod.warm_start_multipliers, mod.choices_choices_lambda, [False], [False]))

 	config_all = config_baselines + config_algs
	config_task = []
	print len(config_all)
	for i in range(len(config_all)):
		if (i % mod.num_tasks == mod.task_id):
			config_task.append(config_all[i])
			print config_all[i]

	return config_task

def get_num_lines(dataset_name):
	ps = subprocess.Popen(('zcat', dataset_name), stdout=subprocess.PIPE)
	output = subprocess.check_output(('wc', '-l'), stdin=ps.stdout)
	ps.wait()
	return int(output)

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
	rgx = re.compile('^'+pattern+' = (.*)$', flags=re.M)

	errs = rgx.findall(vw_output_text)
	if not errs:
		avge = 0
	else:
		avge = float(errs[0])

	vw_output.close()
	return avge


def main_loop(mod):
	mod.summary_file_name = mod.results_path+str(mod.task_id)+'of'+str(mod.num_tasks)+'.sum'
	summary_file = open(mod.summary_file_name, 'w')

	list_header = ['dataset', 'num_classes', 'total_size', \
	'corrupt_type_supervised', 'corrupt_prob_supervised', \
	'corrupt_type_bandit', 'corrupt_prob_bandit', \
	'warm_start_size', 'bandit_size', 'bandit_supervised_size_ratio', \
	'cb_type', 'validation_method', 'weighting_scheme', \
	'lambda_scheme', 'choices_lambda', \
	'no_supervised', 'no_bandit', \
	'avg_error', 'actual_variance', \
	'ideal_variance']

	summary_header = disperse(list_header, ' ')

	summary_file.write(summary_header+'\n')
	summary_file.close()

	for mod.corrupt_type_supervised, mod.corrupt_prob_supervised, \
	mod.cb_type, mod.dataset, mod.warm_start_multiplier, \
	mod.choices_lambda, \
	mod.no_supervised, mod.no_bandit in mod.config_task:
		gen_comparison_graph(mod)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='vw job')
	parser.add_argument('task_id', type=int, help='task ID, between 0 and num_tasks - 1')
	parser.add_argument('num_tasks', type=int)
	parser.add_argument('--results_dir', default='../../../figs/')
	parser.add_argument('--warm_start_fraction', type=float)
	parser.add_argument('--corrupt_prob_supervised', type=float)
	parser.add_argument('--corrupt_prob_bandit',type=float)


	args = parser.parse_args()
	if args.task_id == 0:
		if not os.path.exists(args.results_dir):
			os.makedirs(args.results_dir)
			import stat
			os.chmod(args.results_dir, os.stat(args.results_dir).st_mode | stat.S_IWOTH)
	else:
		while not os.path.exists(args.results_dir):
			time.sleep(1)

	mod = model()
	mod.num_tasks = args.num_tasks
	mod.task_id = args.task_id

	mod.ds_path = '../../../vwshuffled/'
	mod.vw_path = '../vowpalwabbit/vw'
	mod.results_path = args.results_dir

	#DIR_PATTERN = '../results/cbresults_{}/'

	mod.num_checkpoints = 100
	#mod.warm_start = 50
	#mod.bandit = 4096
	#mod.num_classes = 10
	#mod.cb_type = 'mtr'  #'ips'
    #mod.choices_lambda = 10
	#mod.progress = 25
	mod.adf_on = True

	# use fractions instead of absolute numbers
	mod.warm_start_multipliers = [pow(2, i) for i in range(6)]
	#mod.choices_warm_start_frac = [0.01 * pow(2, i) for i in range(1)]
	#mod.choices_warm_start_frac = [0.01, 0.03, 0.1, 0.3]
	#mod.choices_warm_start_frac = [0.03]
	#mod.choices_warm_start_frac = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]

	#mod.choices_warm_start = [0.01 * pow(2, i) for i in range(5)]
	#mod.choices_bandit = [0.01 * pow(2, i) for i in range(5)]

	#mod.choices_warm_start = [pow(2,i) for i in range(11)] #put it here in order to plot 2d mesh
	# we are implicitly iterating over the bandit sample sizes
	#choices_fprob1 = [0.1, 0.2, 0.3]
	#choices_fprob2 = [0.1, 0.2, 0.3]
	#choices_cb_types = ['mtr', 'ips']
	#mod.choices_cb_types = ['mtr', 'ips']
	mod.choices_cb_types = ['mtr']
	mod.choices_no_supervised = [False, True]
	mod.choices_no_bandit = [False, True]
	mod.choices_choices_lambda = [2*i for i in range(1,5)]
	#mod.choices_choices_lambda = [i for i in range(1,3)]
	#mod.choices_choices_lambda = [i for i in range(1,2)]
	#mod.choices_choices_lambda = [1, 3, 5, 7]
	#[i for i in range(10,11)]
	#mod.corrupt_type_supervised = 2
	#mod.corrupt_prob_supervised = 0.3
	mod.choices_corrupt_type_supervised = [1,2]
	#mod.choices_corrupt_type_supervised = [2]
	#mod.corrupt_prob_supervised = 0.3
	mod.choices_corrupt_prob_supervised = [0.0,0.3]
	#mod.choices_corrupt_prob_supervised = [0.3]

	mod.corrupt_type_bandit = 1
	mod.corrupt_prob_bandit = 0

	mod.validation_method = 1
	mod.epsilon = 0.05

	mod.choices_lambda = 2
	mod.weighting_scheme = 1
	mod.lambda_scheme = 3
	mod.no_bandit = False
	mod.no_supervised = False
	mod.no_exploration = False
	mod.cover_on = False
	mod.epsilon_on = True
	mod.plot_color = 'r'
	mod.plot_flat = False
	mod.critical_size_ratios = [pow(2,i) for i in range(-5, 7)]

	#for correctness test
	#mod.choices_warm_start = [20]
	#choices_fprob1 = [0.1]
	#choices_fprob2 = [0.1]

	mod.dss = ds_files(mod.ds_path)
	#mod.dss = ["ds_223_63.vw.gz"]
	#mod.dss = mod.dss[:5]

	# here, we are generating the task specific parameter settings
	# by first generate all parameter setting and pick every num_tasks of them
	mod.config_task = ds_per_task(mod)
	print 'task ' + str(mod.task_id) + ' of ' + str(mod.num_tasks) + ':'

	#print mod.ds_task

	# we only need to vary the warm start fraction, and there is no need to vary the bandit fraction,
	# as each run of vw automatically accumulates the bandit dataset
	main_loop(mod)

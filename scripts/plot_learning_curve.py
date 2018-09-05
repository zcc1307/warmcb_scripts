import matplotlib
matplotlib.use('Agg')
from alg_const import alg_str, alg_color_style, alg_index, noise_type_str
import glob
from alg_comparison import order_legends, save_legend
from run_vw_commands import avg_error, complete_header, VW_PROGRESS_PATTERN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import re

class model:
	def __init__(self):
		pass
'''
def collect_learning_curve(mod):
	f = open(mod.vw_output_filename, 'r')

	avg_losses = []
	weights = []

	i = 0
	for line in f:
		vw_progress_pattern = '\d+\.\d+\s+\d+\.\d+\s+\d+\s+\d+\.\d+\s+[a-zA-Z0-9]+\s+[a-zA-Z0-9]+\s+\d+.*'
		matchobj = re.match(vw_progress_pattern, line)

		if matchobj:
			s = line.split()
			if len(s) >= 8:
				s = s[:7]
			avg_loss_str, last_loss_str, counter_str, weight_str, curr_label_str, \
			curr_pred_str, curr_feat_str = s

			if mod.error_type == 1:
				avg_loss = float(avg_loss_str)
			else:
				avg_loss = float(last_loss_str)

			weight = int(float(weight_str))

			avg_losses.append(avg_loss)
			weights.append(weight)

	if mod.alg_name == 'Sup-Only':
		# for supervised only, we simply plot a horizontal line using the last point
		len_all = len(avg_losses)
		avg_losses = [avg_losses[len_all-1] for i in range(len_all) ]

	f.close()
	return weights, avg_losses
'''

def collect_learning_curve(mod):
	f = open(mod.vw_output_filename, 'r')

	avg_losses = []
	weights = []
	lambdas = []

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

		if mod.error_type == 1:
			avg_loss = float(avg_loss)
		else:
			avg_loss = float(last_loss)

		weight = int(float(weight))
		last_lambda = float(last_lambda)

		avg_losses.append(avg_loss)
		weights.append(weight)
		lambdas.append(last_lambda)

	if mod.alg_name == 'Sup-Only':
		# for supervised only, we simply plot a horizontal line using the last point
		len_all = len(avg_losses)
		avg_losses = [avg_losses[len_all-1] for i in range(len_all) ]

	f.close()
	return weights, avg_losses, lambdas

def problem_str(problem_param_names, problem_param_values):
	s = ''
	l_names = list(problem_param_names)
	l_values = list(problem_param_values)
	for param_name, param_value in zip(l_names, l_values):
		s = s + str(param_name) + '=' + str(param_value) + ','
	return s

def parse_all_filenames(mod):
	d = {'filename':[], 'avg_error':[]}

	all_filenames = glob.glob('*.txt')
	for filename in all_filenames:
		filename_splitted = filename.split(',')
		d['filename'].append(filename)
		for item in filename_splitted:
			item_splitted = item.split('=')
			if len(item_splitted) == 2:
				param_name, param_value = item_splitted
				param_name = complete_header(param_name)
				if param_name not in d.keys():
					d[param_name] = []
				d[param_name].append(param_value)

		mod.vw_output_filename=filename
		d['avg_error'].append(avg_error(mod))

	results = pd.DataFrame(d)
	print results
	results[['warm_start_type','choices_lambda']] = results[['warm_start_type','choices_lambda']].astype(int)

	results[['warm_start_update','interaction_update']] = results[['warm_start_update','interaction_update']]=='True'

	results[['validation_method']] = results[['validation_method']].astype(int)

	print results

	#ignore the no update row:
	results = results[(results['warm_start_update'] == True) | (results['interaction_update'] == True)]
	#ignore the choice_lambda = 4 row
	results = results[(results['choices_lambda'] != 4)]

	return results

def problem_text(problem_param_values):
	s=''
	warm_start_fraction = 0.005 * float(problem_param_values[0])
	#s += '#Warm start = ' + str(warm_start_fraction) + 'n, '
	#s += 'Warm start fraction = ' + str(warm_start_fraction*100) + '%, '
	if abs(float(problem_param_values[2])) < 1e-6:
		s += 'noiseless'
	else:
		s += noise_type_str(int(problem_param_values[1])) + ', '
		s += 'p=' + str(problem_param_values[2])

	return s

def generate_figs_per_ds(mod):
	os.chdir(mod.ds)
	results = parse_all_filenames(mod)

	problem_param_names = ['warm_start_multiplier', 'corrupt_type_warm_start','corrupt_prob_warm_start']
	grouped_by_problem_setting = results.groupby(problem_param_names)

	for name_problem, group_problem in grouped_by_problem_setting:

		mod.problem_name = problem_str(problem_param_names, name_problem)
		mod.problem_text = problem_text(name_problem)
		print group_problem.shape[0]

		alg_param_names = ['warm_start_type', 'choices_lambda', 'warm_start_update', 'interaction_update', 'validation_method']

		group_problem_subcol = group_problem[['warm_start_type', 'choices_lambda', 'warm_start_update', 'interaction_update', 'validation_method']].drop_duplicates()
		num_algs = group_problem_subcol.shape[0]
		# a hack here - ensure that all graphs generated have 7 algorithms.

		grouped_by_algorithm = group_problem.groupby(alg_param_names)

		plt.close("all")
		fig = plt.figure()
		#plt.title(mod.problem_text)
		indices = []

		for name_algorithm, group_problem_algorithm in grouped_by_algorithm:
			# pick the best average learning rate here
			# first compute the mean of all learning rates:
			grouped_by_lr = group_problem_algorithm.groupby('learning_rate')
			lr_means = grouped_by_lr.mean()
			lr_means = lr_means.reset_index()
			idx_min = lr_means['avg_error'].idxmin()
			best_lr = lr_means.iloc[idx_min]['learning_rate']

			group_problem_algorithm = group_problem_algorithm[group_problem_algorithm['learning_rate'] == best_lr]

			weights = None

			print name_algorithm
			mod.alg_name = alg_str(name_algorithm)
			mod.alg_col, mod.alg_sty = alg_color_style(name_algorithm)
			print mod.alg_name, mod.alg_col, mod.alg_sty
			print best_lr, mod.ds

			#if mod.alg_name == 'Class-1' or mod.alg_name == 'AwesomeBandits with $|\Lambda|$=4':
			#	continue

			avg_losses_all = []
			lambdas_all = []

			for idx, row in group_problem_algorithm.iterrows():
				mod.vw_output_filename = row['filename']
				temp_weights, avg_losses, lambdas = collect_learning_curve(mod)
				if weights is None:
					weights = temp_weights
				avg_losses_all.append(avg_losses)
				lambdas_all.append(lambdas)

			folds = len(avg_losses_all)
			lambdas_mean = [np.mean(x) for x in zip(*lambdas_all)]
			avg_losses_mean = [np.mean(x) for x in zip(*avg_losses_all)]
			avg_losses_stderr = [np.std(x)  / np.sqrt(folds) for x in zip(*avg_losses_all)]
			len_x = len(avg_losses_mean)
			avg_losses_ci = [1.96 * avg_losses_stderr[i] for i in range(len_x)]

			# This is because we have 200 checkpoints and we would like to
			# limit the number of examples to be 92%
			num_examples_checkpoint = weights[1] - weights[0]
			#num_bandit_cutoff = 184 * num_examples_checkpoint

			if mod.plot_ticks_only is True:
				ticks = range(0,len_x,len_x/20)
				avg_losses_mean = [avg_losses_mean[i] for i in ticks]
				avg_losses_ci = [avg_losses_ci[i] for i in ticks]
				weights = [weights[i] for i in ticks]
				len_x = len(avg_losses_mean)

			if mod.plot_ci is False:
				avg_losses_ci = [0 for i in range(len_x)]

			if mod.plot_log_scale is True:
				plt.gca().set_xscale('log')

			print len(weights)
			#print weights
			#print avg_losses_mean

			if mod.learning_curve_type == 1:
				#plt.plot(weights, avg_losses_mean, label=mod.alg_name, color=mod.alg_col, linestyle=mod.alg_sty, linewidth=2.0)
				plt.errorbar(weights, lambdas_mean, yerr=avg_losses_ci, label=mod.alg_name, color=mod.alg_col, linestyle=mod.alg_sty, linewidth=2.0)
				#plt.errorbar(weights, avg_losses_mean, yerr=avg_losses_ci, label=mod.alg_name, color=mod.alg_col, linestyle=mod.alg_sty, linewidth=2.0)
			else:
				plt.plot(weights, avg_losses_mean, label=mod.alg_name, color=mod.alg_col, linestyle=mod.alg_sty, linewidth=2.0)
				avg_losses_up = [avg_losses_mean[i] + avg_losses_ci[i] for i in range(len_x)]
				avg_losses_down = [avg_losses_mean[i] - avg_losses_ci[i] for i in range(len_x)]
				plt.fill_between(weights, avg_losses_down, avg_losses_up, color=mod.alg_col, linestyle=mod.alg_sty, alpha=0.2)

			#plt.xlim(0, num_bandit_cutoff)
			indices.append(alg_index(name_algorithm))

		print 'plotting for '+mod.problem_text+'...'
		params={'legend.fontsize':15}
		plt.rcParams.update(params)
		order_legends(indices)
		#plt.xlabel('#Bandit examples',fontsize=20)
		#plt.ylabel('Average error', fontsize=20)
		#plt.title(mod.problem_text, fontsize=20)
		plt.locator_params(axis='y',nbins=5)
		plt.xticks(fontsize=20)
		plt.yticks(fontsize=20)
		plt.tight_layout(h_pad=1.0)
		ax = plt.gca()
		ax.legend_.remove()
		#plt.ylim(0.8,1.0)
		plt.savefig(mod.problem_name+',learning_curve_type='+str(mod.learning_curve_type)+',error_type='+str(mod.error_type)+'_lambdas.pdf')
		mod.problemdir='./'
		save_legend(mod, indices)
		plt.clf()
		plt.close(fig)


def go_over_dirs(mod):
	prevdir = os.getcwd()
	os.chdir(mod.results_dir)
	dss = sorted(glob.glob('*/'))
	os.chdir(prevdir)

	for ds in dss:
		if ds.startswith('ds'):
			mod.ds = ds
			os.chdir(mod.results_dir)
			generate_figs_per_ds(mod)
			os.chdir(prevdir)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='learning curve plots')
	parser.add_argument('--results_dir', default='../../../figs/')
	args = parser.parse_args()

	mod = model()
	mod.results_dir = args.results_dir
	mod.learning_curve_type = 1
	mod.error_type = 1
	mod.plot_ticks_only = False
	mod.plot_ci = False
	mod.plot_log_scale = True
	go_over_dirs(mod)

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
from alg_const import noise_type_str, alg_info, alg_str, alg_str_compatible, alg_color_style, alg_index

pd.set_option('display.max_columns', 500)

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
	sorted_z_scores = sorted(enumerate(z_scores), key=lambda x:x[1])

	significance = map(is_significant, z_scores)
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

def problem_str(name_problem):
	return 'eps='+str(name_problem[5]) \
			+'_sct='+str(name_problem[0]) \
			+'_scp='+str(name_problem[1]) \
			+'_bct='+str(name_problem[2]) \
			+'_bcp='+str(name_problem[3]) \
			+'_ratio='+str(name_problem[4]) \

def problem_text(name_problem):
	s=''
	s += 'Ratio = ' + str(name_problem[2]) + ', '
	if abs(name_problem[1]) < 1e-6:
		s += 'noiseless'
	else:
		s += noise_type_str(name_problem[0]) + ', '
		s += 'p = ' + str(name_problem[1])
	return s


def plot_cdf(alg_name, errs):
	col, sty = alg_color_style(alg_name)
	plt.step(np.sort(errs), np.linspace(0, 1, len(errs), endpoint=False), label=alg_str(alg_name), color=col, linestyle=sty, linewidth=2.0)

def plot_all_cdfs(alg_results, mod):
	print 'printing cdfs..'
	indices = []
	pylab.figure(figsize=(8,6))

	for alg_name, errs in alg_results.iteritems():
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
	plt.clf()

def plot_all_lrs(lrs, mod):
	alg_names = lrs.keys()

	for i in range(len(alg_names)):
		pylab.figure(figsize=(8,6))
		lrs_alg = lrs[alg_names[i]]
		names = mod.learning_rates
		values = [lrs_alg.count(n) for n in names]
		plt.barh(range(len(names)),values)
		plt.yticks(range(len(names)),names)
		plt.savefig(mod.problemdir+alg_str_compatible(alg_names[i])+'_lr.pdf')
		plt.clf()

def plot_all_lambdas(lambdas, mod):
	alg_names = lambdas.keys()

	for i in range(len(alg_names)):
		pylab.figure(figsize=(8,6))
		lambdas_alg = lambdas[alg_names[i]]
		names = sorted(list(set(lambdas_alg)))
		values = [lambdas_alg.count(n) for n in names]
		plt.barh(range(len(names)),values)
		plt.yticks(range(len(names)),names)
		plt.savefig(mod.problemdir+alg_str_compatible(alg_names[i])+'_lambdas.pdf')
		plt.clf()


def plot_all_pair_comp(alg_results, sizes, mod):
	alg_names = alg_results.keys()

	for i in range(len(alg_names)):
		for j in range(len(alg_names)):
			if i < j:
				errs_1 = alg_results[alg_names[i]]
				errs_2 = alg_results[alg_names[j]]

				print len(errs_1), len(errs_2), len(sizes)
				num_wins_1, num_wins_2 = plot_comparison(errs_1, errs_2, sizes)
				plt.title( 'total number of comparisons = ' + str(len(errs_1)) + '\n'+
				alg_str(alg_names[i]) + ' wins ' + str(num_wins_1) + ' times, \n' + alg_str(alg_names[j]) + ' wins ' + str(num_wins_2) + ' times')
				plt.savefig(mod.problemdir+alg_str_compatible(alg_names[i])+'_vs_'+alg_str_compatible(alg_names[j])+'.pdf')
				plt.clf()


def normalize_score(unnormalized_result, mod):
	if mod.normalize_type == 1:
		l = get_best_error(mod.best_error_table, mod.name_dataset)
		u = max(unnormalized_result.values())
		return { k : ((v - l) / (u - l + 1e-4)) for k, v in unnormalized_result.iteritems() }
	elif mod.normalize_type == 2:
		l = unnormalized_result[(1, 1, True, False)]
		return { k : ((v - l) / (l + 1e-4)) for k, v in unnormalized_result.iteritems() }
	elif mod.normalize_type == 3:
		return unnormalized_result
	elif mod.normalize_type == 4:
		l = get_best_error(mod.best_error_table, mod.name_dataset)
		return { k : (v - l) for k, v in unnormalized_result.iteritems() }

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

	i = 0
	for idx, row in result_table.iterrows():
		if i == 0:
			new_size = row['interaction']

		if row['interaction'] == new_size:
			alg_name = (row['warm_start_type'],
			 			row['choices_lambda'],
			 			row['warm_start_update'],
			 			row['interaction_update'],
			 			row['validation_method'])
			new_unnormalized_results[alg_name] = row['avg_error']
			new_lr[alg_name] = row['learning_rate']
			new_lambda[alg_name] = row['last_lambda']
		i += 1

	return new_size, new_unnormalized_results, new_lr, new_lambda

def update_result_dict(results_dict, new_result):
	if len(new_result) != len(results_dict):
		print 'Warning: length of the new record ( ', len(new_result), ' ) does not match the length of the existing dict ( ', len(results_dict), ' ); perhaps the input data is corrupted.'

	for k, v in new_result.iteritems():
		results_dict[k].append(v)


def plot_all(mod, all_results):
	#Group level 1: corruption mode, corruption prob, warm start - bandit ratio (each group corresponds to one cdf plot)
	grouped_by_problem = all_results.groupby(['corrupt_type_warm_start',
											  'corrupt_prob_warm_start',
											  'corrupt_type_interaction',
											  'corrupt_prob_interaction',
											  'inter_ws_size_ratio',
											  'epsilon'])

	#Group level 2: datasets, warm start length (corresponds to each point in cdf)
	for name_problem, group_problem in grouped_by_problem:
		normalized_results = None
		unnormalized_results = None
		sizes = None
		mod.name_problem = name_problem

		#print 'in group_problem:'
		#print name_problem
		#print group_problem[(group_problem['warm_start_update'] == True) & (group_problem['interaction_update'] == False) ].shape
		#raw_input('...')

		grouped_by_dataset = group_problem.groupby(['dataset',
													'warm_start'])

		#Group level 3: algorithms
		for name_dataset, group_dataset in grouped_by_dataset:
			result_table = group_dataset

			#print 'in group_dataset:'
			#print name_dataset
			#print group_dataset[(group_dataset['warm_start_update'] == True) & (group_dataset['interaction_update'] == False) ].shape
			#print group_dataset[(group_dataset['warm_start_update'] == True) & (group_dataset['interaction_update'] == False) ]
			#raw_input('...')

		 	group_dataset = group_dataset.reset_index(drop=True)
			grouped_by_algorithm = group_dataset.groupby(['warm_start_type',
			                                              'choices_lambda',
														  'warm_start_update',
														  'interaction_update',
														  'validation_method'])
			mod.name_dataset = name_dataset

			#The 'learning_rate' would be the only free degree here now. Taking the
			#min aggregation will give us the algorithms we are evaluating.
			#In the future this should be changed if we run multiple folds: we
			#should average among folds before choosing the min
			idx = []

			for name_alg, group_alg in grouped_by_algorithm:
				min_error = group_alg['avg_error'].min()
				min_error_rows = group_alg[group_alg['avg_error'] == min_error]
				num_min_error_rows = min_error_rows.shape[0]
				local_idx = random.randint(0, num_min_error_rows-1)
				global_idx = min_error_rows.index[local_idx]
				idx.append(global_idx)

			result_table = group_dataset.ix[idx, :]

			#print result_table
			#raw_input('...')

			#Record the error rates of all algorithms
			new_size, new_unnormalized_result, new_lr, new_lambda = get_unnormalized_results(result_table)

			new_unnormalized_result[(0, 0, False, False, 1)] = get_maj_error(mod.maj_error_table, mod.name_dataset)
			new_lr[(0, 0, False, False, 1)] = 0.0
			new_lambda[(0, 0, False, False, 1)] = 0.0
			new_normalized_result = normalize_score(new_unnormalized_result, mod)

			#if len(new_lr) != 3:
			#	continue

			#first time - generate names of algorithms considered
			if normalized_results is None:
				sizes = []
				unnormalized_results = dict([(k,[]) for k in new_unnormalized_result.keys()])
				normalized_results = dict([(k,[]) for k in new_unnormalized_result.keys()])
				lrs = dict([(k,[]) for k in new_unnormalized_result.keys()])
				lambdas = dict([(k,[]) for k in new_unnormalized_result.keys()])

			update_result_dict(unnormalized_results, new_unnormalized_result)
			update_result_dict(normalized_results, new_normalized_result)
			update_result_dict(lrs, new_lr)
			update_result_dict(lambdas, new_lambda)
			sizes.append(new_size)

		print normalized_results

		mod.problemdir = mod.fulldir+problem_str(mod.name_problem)+'/'
		if not os.path.exists(mod.problemdir):
			os.makedirs(mod.problemdir)

		if mod.pair_comp_on is True:
			plot_all_pair_comp(unnormalized_results, sizes, mod)
		if mod.cdf_on is True:
			plot_all_cdfs(normalized_results, mod)

		plot_all_lrs(lrs, mod)
		plot_all_lambdas(lambdas, mod)

def save_to_hdf(mod):
	print 'saving to hdf..'
	store = pd.HDFStore(mod.results_dir+'cache.h5')
	store['result_table'] = mod.all_results
	store.close()

def load_from_hdf(mod):
	print 'reading from hdf..'
	store = pd.HDFStore(mod.results_dir+'cache.h5')
	mod.all_results = store['result_table']
	store.close()

def load_from_sum(mod):
	print 'reading directory..'
	dss = sum_files(mod.results_dir)
	results_arr = []

	print 'reading sum tables..'
	for i in range(len(dss)):
		print 'result file name: ', dss[i]
		result = parse_sum_file(mod.results_dir + dss[i])
		results_arr.append(result)

	all_results = pd.concat(results_arr)
	print all_results
	mod.all_results = all_results

def filter_results(modm, all_results):
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

	return all_results


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='result summary')
	parser.add_argument('--results_dir', default='../../../figs/')
	parser.add_argument('--filter', default='1')
	parser.add_argument('--plot_subdir', default='expt1/')
	parser.add_argument('--cached', action='store_true')
	parser.add_argument('--normalize_type', type=int, default=1)
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
	mod.pair_comp_on = False
	mod.cdf_on = True
	mod.maj_error_dir = '../../../old_figs/figs_all/expt_0509/figs_maj_errors/0of1.sum'
	mod.best_error_dir = '../../../old_figs/figs_all/expt_0606/0of1.sum'

	mod.fulldir = mod.results_dir + mod.plot_subdir
	if not os.path.exists(mod.fulldir):
		os.makedirs(mod.fulldir)

	if args.cached is True:
		if os.path.exists(mod.results_dir+'cache.h5'):
			load_from_hdf(mod)
		else:
			load_from_sum(mod)
			save_to_hdf(mod)
	else:
		load_from_sum(mod)

	all_results = mod.all_results
	#ignore the choices_lambda = 4 row
	all_results = all_results[(all_results['choices_lambda'] != 4)]
	all_results = all_results[(all_results['choices_lambda'] != 8)]

	all_results['epsilon'] = all_results['epsilon'].astype(float)
	uniq_eps = all_results['epsilon'].unique()
	sup_only_results = all_results[(all_results['warm_start_update'] == True) & (all_results['interaction_update'] == False) ]
	sup_only_augmented = []
	#print uniq_eps
	for eps in uniq_eps:
		sup_only_eps = sup_only_results.copy(deep=True)
		sup_only_eps['epsilon'] = eps
		#sup_only_eps = sup_only_results['epsilon'].apply(lambda x: eps)
		#print sup_only_eps[(sup_only_eps['epsilon'] == 0.0125) & (sup_only_eps['warm_start_update'] == True) & (sup_only_eps['interaction_update'] == False)].shape

		sup_only_augmented.append(sup_only_eps)
		#print eps
		#raw_input('..')

	#(1.0, 0.0, 1.0, 0.0, 2.875, 0.0125)
	#for tab in sup_only_augmented:
	#	print tab
	#	raw_input('..')

	sup_only_all_eps = pd.concat(sup_only_augmented)
	#print sup_only_all_eps[(sup_only_all_eps['epsilon'] == 0.0125) & (sup_only_all_eps['warm_start_update'] == True) & (sup_only_all_eps['interaction_update'] == False)].shape
	all_results = pd.concat([all_results, sup_only_all_eps])
	#all_results.append(sup_only_all_eps)
	all_results = all_results[(all_results['epsilon'] != 0.0)]


	#all_results = all_results[(all_results['warm_start_update'] == True) & (all_results['interaction_update'] == True)]
	# Some of the summary files have broken records (incomplete rows)
	all_results['learning_rate'] = all_results['learning_rate'].astype(float)
	#print all_results[all_results.apply(lambda row: math.isnan(row['learning_rate']), axis=1)]
	#raw_input('..')

	#all_results['warm_start_type'] = all_results['warm_start_type'].astype(int)
	#all_results['choices_lambda'] = all_results['choices_lambda'].astype(int)
	#all_results['warm_start_update'] = all_results['warm_start_update'].astype(bool)
	#all_results['interaction_update'] = all_results['interaction_update'].astype(bool)
	#all_results['validation_method'] = all_results['validation_method'].astype(int)


	#print all_results[(all_results['epsilon'] == 0.0125) & (all_results['warm_start_update'] == True) & (all_results['interaction_update'] == False)].shape
	#raw_input('..')

	mod.maj_error_table = parse_sum_file(mod.maj_error_dir)
	mod.maj_error_table = mod.maj_error_table[mod.maj_error_table['majority_approx']]
	mod.best_error_table = parse_sum_file(mod.best_error_dir)
	mod.best_error_table = mod.best_error_table[mod.best_error_table['optimal_approx']]


	mod.learning_rates = sorted(all_results.learning_rate.unique())


	all_results = filter_results(mod, all_results)

	plot_all(mod, all_results)

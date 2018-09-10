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
	'dataset',
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
	'adf_on',
	'algorithm',
	'choices_lambda',
	'validation_method',
	'weighting_scheme',
	'lambda_scheme',
	'epsilon',
	'loss0',
	'loss1',
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

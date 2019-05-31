import subprocess
from itertools import product
import os
import math
import argparse
import time
import glob
import re
from collections import OrderedDict
from params_gen import get_all_params
from vw_commands_const import VW_RUN_TMPLT_OPT, VW_RUN_TMPLT_MAJ, VW_RUN_TMPLT_WARMCB, \
  VW_PROGRESS_PATTERN, VW_RESULT_TMPLT, FULL_TMPLT, SIMP_MAP, SUM_TMPLT, VW_OUT_TMPLT
from utils import intersperse, format_setting, replace_keys, merge_dicts, param_to_str
import gzip
import random
from parse_res import write_row, write_result, analyze_vw_out

class model:
    def __init__(self):
        # Setting up argument-independent learning parameters in the constructor
        self.baselines_on = True
        self.sup_only_on = True
        self.band_only_on = True
        self.sim_bandit_on = True
        self.one_lambda_on = False

        self.algs_on = True
        self.optimal_on = True
        self.majority_on = True

        #self.ws_gt_on = False
        #self.inter_gt_on = True

        self.num_checkpoints = 200

        # use fractions instead of absolute numbers
        self.ws_multipliers = [pow(2,i) for i in range(4)]

        self.choices_cb_type = ['mtr']
        self.choices_choices_lambda = [2,8]

        #self.choices_cor_type_ws = [1,2,3]
        #self.choices_cor_prob_ws = [0.0,0.25,0.5,1.0]
        self.choices_cor_type_ws = [1,2]
        self.choices_cor_prob_ws = [0.0,0.25]
        #self.choices_cor_type_ws = [1]
        #self.choices_cor_prob_ws = [0.0]

        self.choices_cor_type_inter = [1]
        self.choices_cor_prob_inter = [0.0]
        #self.choices_cor_type_inter = [1,2]
        #self.choices_cor_prob_inter = [0.0,0.5]

        self.choices_loss_enc = [(0, 1)]
        #self.choices_epsilon = [0.05]
        self.choices_epsilon = [0.00625, 0.0125]
        #self.choices_epsilon = [0.00625, 0.0125, 0.025, 0.05, 0.1]
        self.choices_eps_t = []

        self.choices_adf = [True]
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

def gen_lrs(num_lrs):
    if num_lrs <= 0:
        lrs = [0.5]
    else:
        lrs = [gen_lr(i) for i in range(num_lrs)]

    return lrs

def gen_vw_command(mod):
    mod.vw_options = format_setting(mod.vw_run_tmplt, mod.param)
    vw_options_list = []
    for k, v in mod.vw_options.items():
        vw_options_list.append('--'+str(k))
        vw_options_list.append(str(v))
    cmd = intersperse([mod.vw_path]+vw_options_list, ' ')
    return cmd

def gen_vw_options(mod):
    if mod.param['algorithm'] == 'Optimal':
        # Cost-sensitive one-versus-all learning on full dataset
        mod.vw_run_tmplt = OrderedDict(VW_RUN_TMPLT_OPT)
        mod.param['passes'] = 5
        mod.param['cache_file'] = mod.param['data'] + '.cache'
        mod.param['oaa'] = mod.param['num_classes']

    elif mod.param['algorithm'] == 'Most-Freq':
        # Skip vw running
        pass
    else:
        # Contextual bandits simulation
        mod.vw_run_tmplt = OrderedDict(VW_RUN_TMPLT_WARMCB)
        mod.param['warm_start'] = mod.param['warm_start_multiplier'] * mod.param['grid_size']
        mod.param['interaction'] = mod.param['total_size'] - mod.param['warm_start']
        mod.param['warm_cb'] = mod.param['num_classes']
        mod.param['overwrite_label'] = mod.param['majority_class']

        if mod.param['adf_on'] is True:
            mod.param['cb_explore_adf'] = ' '
            mod.vw_run_tmplt['cb_explore_adf'] = ' '
        else:
            mod.param['cb_explore'] = mod.param['num_classes']
            mod.vw_run_tmplt['cb_explore'] = 0

        if 'sim_bandit' in mod.param.keys():
            mod.param['sim_bandit'] = ' '
            mod.vw_run_tmplt['sim_bandit'] = ' '

        if mod.param['warm_start_update'] is True:
            mod.param['warm_start_update'] = ' '
            mod.vw_run_tmplt['warm_start_update'] = ' '

        if mod.param['interaction_update'] is True:
            mod.param['interaction_update'] = ' '
            mod.vw_run_tmplt['interaction_update'] = ' '

def execute_vw(mod):
    print('writing to:', mod.vw_output_filename)

    if mod.param['algorithm'] == 'Most-Freq':
        maj_err = get_maj_error_mc(mod.param['data'])
        f = open(mod.vw_output_filename, 'w')
        f.write('average loss = '+str(maj_err)+'\n')
        f.close()
    else:
        gen_vw_options(mod)
        cmd = gen_vw_command(mod)
        print(mod.param['algorithm'])
        print(cmd, '\n')
        f = open(mod.vw_output_filename, 'w')
        f.write(cmd+'\n')
        f.close()
        f = open(mod.vw_output_filename, 'a')
        process = subprocess.Popen(cmd, shell=True, stdout=f, stderr=f)
        process.wait()
        f.close()

def get_vw_out_filename(mod):
    # step 1: fill in vw output template
    param_formatted = format_setting(mod.vw_out_tmplt, mod.param)
    # step 2: replace the key names with the simplified names
    param_simplified = replace_keys(param_formatted, mod.simp_map)
    return param_to_str(param_simplified)

def run_single_expt(mod):
    mod.param['data'] = mod.ds_path + str(mod.param['fold']) + '/' + mod.param['dataset']
    mod.param['total_size'] = get_num_lines(mod.param['data'])
    mod.param['num_classes'] = get_num_classes(mod.param['data'])
    mod.param['majority_class'] = get_maj_class_mc(mod.param['data'])

    mod.param['grid_size'] = int(math.ceil(float(mod.param['total_size']) / float(mod.num_checkpoints)))
    mod.param['progress'] = mod.param['grid_size']
    mod.vw_output_dir = mod.results_path + rm_suffix(mod.param['data']) + '/'
    mod.vw_output_filename = mod.vw_output_dir + get_vw_out_filename(mod) + '.txt'

    mod.param['vw_output_name'] = mod.vw_output_filename

    execute_vw(mod)

    #if mod.remove_vw_out:
    #    os.remove(mod.vw_output_filename)

def shuffle(ds_name, dir, fold):
    f = gzip.open(dir + '/' + ds_name, 'rt')
    data = [(random.random(), line) for line in f]
    data.sort()

    f_shuf = gzip.open(dir + '/' + str(fold) + '/' + ds_name, 'wt')
    for _, line in data:
        f_shuf.write(line)

def ds_files(ds_path, num_ds):
    prevdir = os.getcwd()
    os.chdir(ds_path)
    dss = sorted(glob.glob('*.vw.gz'))
    os.chdir(prevdir)

    if num_ds == -1 or num_ds > len(dss):
        pass
    else:
        dss = dss[:num_ds]

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

def get_class_count(dataset_name):
    count_label = {}
    f = gzip.open(dataset_name, 'r')
    for line in f:
        line_label = line.decode("utf-8").split('|')
        label = line_label[0].split()[0]
        if label not in count_label:
            count_label[label] = 0
        count_label[label] += 1
    return count_label

def get_num_classes(dataset_name):
    count_label = get_class_count(dataset_name)
    return len(count_label)

def get_maj_class_mc(dataset_name):
    count_label = get_class_count(dataset_name)
    return max(count_label, key=count_label.get)

def get_maj_error_mc(dataset_name):
    count_label = get_class_count(dataset_name)
    return 1 - (float(max(count_label.values())) / float(sum(count_label.values())))

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        import stat
        os.chmod(dir, os.stat(dir).st_mode | stat.S_IWOTH)

def rm_suffix(filename):
    return os.path.basename(filename).split('.')[0]

def collect_result(mod):
    vw_run_results = analyze_vw_out(mod.param, mod.vw_output_filename)
    for vw_result in vw_run_results:
        result_combined = merge_dicts(mod.param, vw_result)
        result_formatted = format_setting(SUM_TMPLT, result_combined)
        write_result(mod.summary_file_name, result_formatted)

def main_loop(mod):
    mod.summary_file_name = mod.results_path+str(mod.task_id)+'of'+str(mod.num_tasks)+'.sum'
    write_row(mod.summary_file_name, SUM_TMPLT.keys(), 'w')

    mod.full_tmplt = FULL_TMPLT
    mod.simp_map = SIMP_MAP
    mod.sum_tmplt = SUM_TMPLT
    mod.vw_out_tmplt = VW_OUT_TMPLT
    for mod.param in mod.config_task:
        run_single_expt(mod)
        collect_result(mod)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vw job')
    parser.add_argument('task_id', type=int, help='task ID, between 0 and num_tasks - 1')
    parser.add_argument('num_tasks', type=int)
    parser.add_argument('--results_dir', default='../../output/')
    parser.add_argument('--ds_dir', default='../../data/')
    parser.add_argument('--num_learning_rates', type=int, default=1)
    parser.add_argument('--num_datasets', type=int, default=-1)
    parser.add_argument('--num_folds', type=int, default=1)
    parser.add_argument('--remove_vw_out', type=int, default=0)

    args = parser.parse_args()
    flag_dir = args.results_dir + 'flag/'

    mod = model()
    mod.num_tasks = args.num_tasks
    mod.task_id = args.task_id
    mod.vw_path = '../../vowpal_wabbit/build/vowpalwabbit/vw'
    mod.ds_path = args.ds_dir
    mod.results_path = args.results_dir
    mod.remove_vw_out = (args.remove_vw_out != 0)

    print('reading dataset files..')
    #This line handles multiple folds, where we assume that
    #folder 1/ includes all dataset files, and subsequent folders
    #includes the same set of filenames with different shufflings
    mod.dss = ds_files(mod.ds_path, args.num_datasets)
    mod.learning_rates = gen_lrs(args.num_learning_rates)
    mod.folds = range(1, args.num_folds+1)

    if args.task_id == 0:
        # To avoid race condition, use subtask 0 to create all folders
        create_dir(args.results_dir)

        # Create a subfolder for each dataset for storing the VW outputs.
        # To avoid having too many files in the same folder
        for ds in mod.dss:
            ds_no_suffix = rm_suffix(ds)
            create_dir(args.results_dir + ds_no_suffix + '/')


        for fold in mod.folds:
            create_dir(mod.ds_path + '/' + str(fold))
            for ds in mod.dss:
                shuffle(ds, mod.ds_path, fold)

        # Create a flag directory to mark the success of creating all folders.
        create_dir(flag_dir)
    else:
        while not os.path.exists(flag_dir):
            time.sleep(1)

    print(len(mod.dss))
    # Task-specific parameter settings (for running the tasks in parallel)
    # First generate all parameter settings
    # Then pick every num_tasks of them
    print('generating tasks..')

    all_params = get_all_params(mod)
    mod.config_task = get_params_task(all_params)
    print('task ', str(mod.task_id), ' of ', str(mod.num_tasks), ':')
    print(len(mod.config_task))

    main_loop(mod)

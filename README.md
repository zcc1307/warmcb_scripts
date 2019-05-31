# Warm starting contextual bandits: robustly combining supervised and bandit feedback - source code

## Prerequisites:
- Vowpal Wabbit prerequisites (see https://github.com/VowpalWabbit/vowpal_wabbit "Prerequisite software" for details).
- Python >= 3.6.5
- matplotlib >= 2.2.2
- seaborn >= 0.9.0

## Includes:

warmcb_scripts/scripts/: scripts for running the scripts for generating the CDFs

## Running instructions:

**Step 1**: download and compile Vowpal Wabbit (follow the instructions in https://github.com/VowpalWabbit/vowpal_wabbit/), ensuring that the vowpal_wabbit/ directory is at the same level as warmcb_scripts/

**Step 2**: Create a folder data/ at the same level as warmcb_scripts/, download all datasets evaluated in the paper from openml.org, by executing the following in warmcb_scripts/scripts/

`python oml_to_vw.py 0 2000`

The script will automatically download all the openml dataset and transform them into Vowpal Wabbit Format
in the data/ folder (with cache file created in /data/omlcache/)

**Step 3**: In folder warmcb_scripts/scripts, run python scripts for generating the results (written to figs/):

`python run_vw_commands.py 0 1 --num_learning_rates 9`

The will generate a file named 0of1.sum, which is a table that summarizes the output of VW in different
experimental settings.

Remark: we can parallelize by running "python run_vw_commands.py task_num n_tasks", for task_num = 0,1,..,n_tasks-1.
For example:
`python run_vw_commands.py 0 3 --num_learning_rates 9`
`python run_vw_commands.py 1 3 --num_learning_rates 9`
`python run_vw_commands.py 2 3 --num_learning_rates 9`
The script with split the workload to each of the executions. This will generate three files 0of3.sum, 1of3.sum and 2of3.sum,
each of which records the result of each subtask.

**Step 4**: plot the aggregated graphs. In warmcb_scripts/scripts/, use the following

4.1 For the full aggregated plots, i.e. grouped according to epsilon only:

`python alg_comparison.py --cached --filter 2 --plot_subdir all_eq/ --agg_mode all_eq`

Results can be found in figs/all_eq/ folder.

4.2 For the plots that aggregates over warm start ratios, i.e. grouped according to (epsilon, corruption):

`python alg_comparison.py --cached --filter 2 --plot_subdir agg_ratio_eq/ --agg_mode agg_ratio_eq`

Results can be found in figs/agg_ratio_eq/ folder.

4.3 For the plots of individual noise condition and warm start ratio, i.e. grouped according to (epsilon, corruption, warm start ratio):

`python alg_comparison.py --cached --filter 2 --plot_subdir agg_no/ --agg_mode no`

Results can be found in figs/agg_no/ folder.

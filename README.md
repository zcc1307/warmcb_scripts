# Warm starting contextual bandits: robustly combining supervised and bandit feedback - source code

## Prerequisites:
- Vowpal Wabbit prerequisites (see https://github.com/VowpalWabbit/vowpal_wabbit "Prerequisite software" for details).
- Python >= 3.6.5
- Matplotlib >= 2.2.2

## Includes:

warmcb_scripts/scripts/: scripts for running the scripts for generating the CDFs

## Running instructions:

**Step 1**: download and compile VW, ensuring that the vowpal_wabbit/ directory is in the same level as /warmcb_scripts

**Step 2**: download all datasets evaluated in the paper from openml.org
`cd warmcb_scripts/scripts/; python2 oml_to_vw.py 0 2000`
Copy all files just downloaded (in vwdatasets/) to vwshuffled/1/

**Step 3**: run python scripts for generating the results (written to figs/):
`cd warmcb_scripts/scripts/; python run_vw_commands.py 0 1 --num_learning_rates 9`

(Can parallelize by running "python run_vw_commands.py task_num n_tasks", for task_num = 0,1,..,n_tasks-1.)

**Step 4**: plot the aggregated graphs:
cd warmcb_scripts/scripts/

For the full aggregated plots (results can be found in figs/all_eq/ folder):
`python alg_comparison.py --results_dir ../../../figs/ --cached --filter 8 --plot_subdir all_eq/ --agg_mode all_eq`

For the plots that aggregates over warm start ratios (results can be found in figs/agg_ratio_eq/ folder):
`python alg_comparison.py --results_dir ../../../figs/ --cached --filter 8 --plot_subdir agg_ratio_eq/ --agg_mode agg_ratio_eq`

For the plots for individual noise condition and warm start ratios (results can be found in figs/agg_no/ folder):
`python alg_comparison.py --results_dir ../../../figs/ --cached --filter 8 --plot_subdir agg_no/ --agg_mode no`


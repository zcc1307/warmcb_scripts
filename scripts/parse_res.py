from vw_commands_const import VW_OUTFILE_NAME_TMPLT, COMP_MAP, SUM_TMPLT, \
                              VW_PROGRESS_PATTERN, VW_RESULT_TMPLT
from utils import intersperse, format_setting, replace_keys, merge_dicts
import glob
import argparse
import re
from collections import OrderedDict

def avg_error(fn):
    return vw_output_extract(fn, 'average loss')

def vw_output_extract(fn, pattern):
    vw_output = open(fn, 'r')
    vw_output_text = vw_output.read()

    rgx_pattern = pattern + ' = ([\d]*.[\d]*)( h|)\n'
    rgx = re.compile(rgx_pattern, flags=re.M)

    errs = rgx.findall(vw_output_text)
    if not errs:
        avge = 0
    else:
        avge = float(errs[0][0])

    vw_output.close()
    return avge

def vw_out_lns(fn):
    results = []
    f = open(fn, 'r')
    vw_output_text = f.read()
    rgx = re.compile(VW_PROGRESS_PATTERN, flags=re.M)
    matched = rgx.findall(vw_output_text)

    for mat in matched:
        if len(mat) >= 8:
            print('warning: parsing vw file encountered unexpected output. The line is: ', line)
            mat = mat[:7]
        results.append(mat)
    f.close()
    return results

def analyze_vw_out(param, fn):
    vw_run_results = []
    if 'algorithm' not in param.keys():
        import pdb; pdb.set_trace()

    if param['algorithm'] == 'Most-Freq' or param['algorithm'] == 'Optimal':
        vw_result = VW_RESULT_TMPLT.copy()
        vw_result['avg_error'] = avg_error(fn)
        vw_run_results.append(vw_result)
        return vw_run_results

    vw_results = vw_out_lns(fn)

    for s in vw_results:
        avg_loss, last_loss, counter, weight, curr_label, curr_pred, curr_feat = s
        counter = int(counter)
        weight = float(weight)
        inter = int(weight)
        ws = counter - weight
        wsm = int(param['warm_start_multiplier'])
        grid_size = ws / wsm

        for im in [23.0, 46.0, 92.0, 184.0]:
            if inter >= (1 - 1e-7) * grid_size * im and \
               inter <= (1 + 1e-7) * grid_size * im:
                vw_result = VW_RESULT_TMPLT.copy()
                vw_result['interaction_multiplier'] = im
                vw_result['interaction'] = inter
                vw_result['inter_ws_size_ratio'] = im / wsm
                vw_result['avg_error'] = float(avg_loss)
                vw_result['warm_start'] = ws
                vw_run_results.append(vw_result)
    return vw_run_results

def write_row(fn, row, mode):
    summary_file = open(fn, mode)
    if mode == 'a':
        summary_file.truncate()
    summary_file.write( intersperse(row, '\t') + '\n')
    summary_file.close()

def write_result(sum_fn, result):
    write_row(sum_fn, result.values(), 'a')

def write_summary_header(sum_fn):
    write_row(sum_fn, SUM_TMPLT.keys(), 'w')

def parse_fn(fn):
    t = fn.split('/')[-1]
    p = t.split(',')
    param = OrderedDict()

    for part in p:
        matched = bracket_rgx.findall(part)
        if len(matched) != 0:
            k, v = matched[0]
            param[k] = v

    return param


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vw job')
    parser.add_argument('--results_dir', default='../../../figs/')

    args = parser.parse_args()
    files = sorted(glob.glob(args.results_dir+'/*/*.txt'))

    bracket_pat = '(.*)={(.*)}'
    bracket_rgx = re.compile(bracket_pat, flags=re.M)

    sum_fn = args.results_dir + 'summary.sum'

    write_row(sum_fn, SUM_TMPLT.keys(), 'w')

    for fn in files:
        param = parse_fn(fn)
        param = replace_keys(param, COMP_MAP)
        vw_run_results = analyze_vw_out(param, fn)

        for vw_result in vw_run_results:
            result_combined = merge_dicts(param, vw_result)
            result_formatted = format_setting(SUM_TMPLT, result_combined)
            write_result(sum_fn, result_formatted)

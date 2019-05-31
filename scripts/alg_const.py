import seaborn as sns
import random
import re

ALG_NAMES_SUM = \
    ['Most-Freq',
    'Sim-Bandit',
    'Bandit-Only',
    'Sup-Only',
    'ARRoW-CB,2',
    'ARRoW-CB,8',
    'Optimal',
    'unknown']

ALG_NAMES_LATEX = \
    ['Majority',
    'Sim-Bandit',
    'Bandit-Only',
    'Sup-Only',
    'ARRoW-CB with $|\Lambda|$=2',
    'ARRoW-CB with $|\Lambda|$=8',
    'Optimal',
    'unknown']

palette = sns.color_palette('colorblind', 10)
colors = palette.as_hex()
#['black', 'magenta', 'lime', 'green', 'blue', 'darkorange','darksalmon', 'red', 'cyan']

ALG_COLORS = \
    [colors[9],
    colors[4],
    colors[0],
    colors[2],
    colors[3],
    colors[3],
    'black',
    'black' ]

ALG_STYLES = \
    ['solid',
    'solid',
    'solid',
    'solid',
    'dashed',
    'dashdot',
    'solid',
    'solid']

ALG_COLORS_STYLES = list(zip(ALG_COLORS, ALG_STYLES))

ALG_ORDERS = \
    [5,
    4,
    3,
    2,
    1,
    0,
    6,
    7]

def rand_color():
    r = lambda: random.randint(0,255)
    return ('#%02X%02X%02X' % (r(),r(),r()))

def get_color(alg):
    l = float(alg.split('=')[1])
    r = int(255*( (1-l) ))
    g = int(255*( (1-l) + 0.44 * l ))
    b = int(255*( 0.69 * l ))
    #r = 50 + int(185 * l)
    #g = 50 + int(185 * l)
    #b = 0
    #50 + int(185 * l)
    return ('#%02X%02X%02X' % (r,g,b))

def get_order(alg):
    return 10.0 - float(alg.split('=')[1])

def switch(input, cases, results, default_type):
    for case, result in zip(cases, results):
        if input == case:
            return result
    if default_type == 'keep':
        return input
    elif default_type == 'color':
        return (get_color(input), 'solid')
    elif default_type == 'order':
        import pdb; pdb.set_trace()
        return get_order(input)
    else:
        return results[len(results)-1]

def corr_type_res(t):
    return switch(t, [1,2,3], ['UAR','CYC','MAJ'], 'unknown')

def alg_str(alg_name):
    return switch(alg_name, ALG_NAMES_SUM, ALG_NAMES_LATEX, 'keep')

def alg_color_style(alg_name):
    return switch(alg_name, ALG_NAMES_SUM, ALG_COLORS_STYLES, 'color')

def alg_index(alg_name):
    return switch(alg_name, ALG_NAMES_SUM, ALG_ORDERS, 'order')

def corr_prob_res(p):
    return "p = {}".format(p)

def expl_res(e):
    return "$\epsilon$ = {}".format(e)

def ratio_res(r):
    return "ratio = {}".format(r)

def make_header(setting):
    s = ""
    counter = 0
    st = setting.copy()

    if 'corrupt_prob_warm_start' in st and 'corrupt_type_warm_start' in st:
        if float(st['corrupt_prob_warm_start']) < 1e-4:
            st.pop('corrupt_prob_warm_start', None)
            st.pop('corrupt_type_warm_start', None)
            s += "Noiseless, "

    for k, v in st.items():
        if k == 'epsilon':
            s += expl_res(v)
        elif k == 'corrupt_prob_warm_start':
            s += corr_prob_res(v)
        elif k == 'inter_ws_size_ratio':
            s += ratio_res(v)
        elif k == 'corrupt_type_warm_start':
            s += corr_type_res(v)

        counter += 1
        if counter != len(st):
            s += ", "

    return s

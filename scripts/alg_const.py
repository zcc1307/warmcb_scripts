import seaborn as sns
import random

ALG_NAMES_SUM = \
    ['Most-Freq',
    'Sim-Bandit',
    'Class-1',
    'Bandit-Only',
    'Sup-Only',
    'Sim-Bandit-Freeze',
    'AwesomeBandits,vm=1,wts=1,cl=2',# for interaction ground truth
    'AwesomeBandits,vm=1,wts=1,cl=4',
    'AwesomeBandits,vm=1,wts=1,cl=8',
    'AwesomeBandits,vm=1,wts=1,cl=16',
    'AwesomeBandits,vm=2,wts=2,cl=2', # for warm start ground truth
    'AwesomeBandits,vm=2,wts=2,cl=4',
    'AwesomeBandits,vm=2,wts=2,cl=8',
    'AwesomeBandits,vm=2,wts=2,cl=16',
    'AwesomeBandits,vm=3,wts=2,cl=2',
    'AwesomeBandits,vm=3,wts=2,cl=4',
    'AwesomeBandits,vm=3,wts=2,cl=8',
    'AwesomeBandits,vm=3,wts=2,cl=16',
    'Optimal',
    'unknown'
    ]

ALG_NAMES_LATEX = \
    ['Most-Freq',
    'Sim-Bandit',
    'Class-1',
    'Bandit-Only',
    'Sup-Only',
    'Sim-Bandit-Freeze',
    'MinimaxBandits',
    'AwesomeBandits with $|\Lambda|$=4',
    'AwesomeBandits with $|\Lambda|$=8',
    'AwesomeBandits with $|\Lambda|$=16',
    'MinimaxBandits, no-split validation',
    'AwesomeBandits with $|\Lambda|$=4, no-split validation',
    'AwesomeBandits with $|\Lambda|$=8, no-split validation',
    'AwesomeBandits with $|\Lambda|$=16, no-split validation',
    'MinimaxBandits, split validation',
    'AwesomeBandits with $|\Lambda|$=4, split validation',
    'AwesomeBandits with $|\Lambda|$=8, split validation',
    'AwesomeBandits with $|\Lambda|$=16, split validation',
    'Optimal',
    'unknown']

ALG_NAMES_COMPATIBLE = \
    ['Most-Freq',
    'Sim-Bandit',
    'Class-1',
    'Bandit-Only',
    'Sup-Only',
    'Sim-Bandit-Freeze',
    'Choices_lambda=2, validation_method=1',
    'Choices_lambda=4, validation_method=1',
    'Choices_lambda=8, validation_method=1',
    'Choices_lambda=16, validation_method=1',
    'Choices_lambda=2, validation_method=2',
    'Choices_lambda=4, validation_method=2',
    'Choices_lambda=8, validation_method=2',
    'Choices_lambda=16, validation_method=2',
    'Choices_lambda=2, validation_method=3',
    'Choices_lambda=4, validation_method=3',
    'Choices_lambda=8, validation_method=3',
    'Choices_lambda=16, validation_method=3',
    'Optimal',
    'unknown']

palette = sns.color_palette('colorblind')
colors = palette.as_hex()
#['black', 'magenta', 'lime', 'green', 'blue', 'darkorange','darksalmon', 'red', 'cyan']

ALG_COLORS = \
    [
    colors[5],
    colors[3],
    'black',
    colors[0],
    colors[1],
    'black',
    colors[2],
    colors[2],
    colors[2],
    colors[2],
    colors[2],
    colors[2],
    colors[2],
    colors[2],
    colors[4],
    colors[4],
    colors[4],
    colors[4],
    'black',
    'black' ]

ALG_STYLES = \
    [
    'solid',
    'solid',
    'solid',
    'solid',
    'solid',
    'dashed',
    'dashed',
    'dotted',
    'dashdot',
    'solid',
    'dashed',
    'dotted',
    'dashdot',
    'solid',
    'dashed',
    'dotted',
    'dashdot',
    'solid',
    'solid',
    'solid']

ALG_COLORS_STYLES = list(zip(ALG_COLORS, ALG_STYLES))

ALG_ORDERS = \
    [
    7.0,
    6.0,
    8.0,
    5.0,
    4.0,
    8.5,
    1.9,
    1.0,
    1.2,
    1.5,
    1.9,
    1.0,
    1.2,
    1.5,
    2.9,
    2.0,
    2.2,
    2.5,
    8.9,
    9.0]

def rand_color():
    r = lambda: random.randint(0,255)
    return ('#%02X%02X%02X' % (r(),r(),r()))

def get_color(alg):
    l = float(alg.split('=')[1])
    r = 50 + int(185 * l)
    g = 50 + int(185 * l)
    b = 0
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
        return get_order(input)
    else:
        return results[len(results)-1]

def noise_type_str(noise_type):
    return switch(noise_type, [1,2,3], ['UAR','CYC','MAJ'], 'unknown')

def alg_str(alg_name):
    return switch(alg_name, ALG_NAMES_SUM, ALG_NAMES_LATEX, 'keep')

def alg_str_compatible(alg_name):
    return switch(alg_name, ALG_NAMES_SUM, ALG_NAMES_COMPATIBLE, 'keep')

def alg_color_style(alg_name):
    return switch(alg_name, ALG_NAMES_SUM, ALG_COLORS_STYLES, 'color')

def alg_index(alg_name):
    return switch(alg_name, ALG_NAMES_SUM, ALG_ORDERS, 'order')

from collections import OrderedDict

def intersperse(l, ch):
    s = ''
    for item in l:
        s += str(item)
        s += ch
    return s

# Template filling
# Given a template, we use the setting dict to fill it as much as possible
def format_setting(template, setting):
    formatted = template.copy()
    for k, v in setting.items():
        if k in template.keys():
            formatted[k] = v
    return formatted

def replace_keys(dic, keymap):
    dic_new = OrderedDict()
    for k, v in dic.items():
        dic_new[keymap[k]] = v
    return dic_new

def merge_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

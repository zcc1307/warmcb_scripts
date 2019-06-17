'''
Scripts for downloading openml datasets
adapted from Alberto Bietti's script, originally at
https://github.com/albietz/cb_bakeoff/blob/master/oml_to_vw.py
'''

import argparse
import gzip
import openml
import os
import scipy.sparse as sp
import numpy as np

VW_DS_DIR = '../../data/'

def augment_nom_ft(X, i):
    X_col = X[:,i].astype('str')
    uniq = np.unique(X_col)
    uniq_elts = len(uniq)
    x_len = np.shape(X)[0]
    aug = np.zeros((x_len, uniq_elts))
    for i in range(uniq_elts):
        aug[:,i] = (X_col == uniq[i])
    return aug

def augment_nom_fts(X, fts):
    X_aug = []
    for i in fts:
        X_aug.append(augment_nom_ft(X, i))

    num_cols = np.shape(X)[1]
    msk_remain = [ True for i in range(num_cols) ]
    for i in fts:
        msk_remain[i] = False

    X_remain = X[:, msk_remain]

    X_aug.append(X_remain)
    return np.concatenate(X_aug, axis=1)

def gen_cost_mat(k):
    cost_mat = np.random.rand(k,k)
    for i in range(k):
        for j in range(k):
            if i == j:
                cost_mat[i,j] = 0

    return cost_mat

def gen_cs_labels(Y, cost_mat):
    y_len = np.shape(Y)[0]
    k = np.shape(cost_mat)[0]
    cs_labels = np.zeros((y_len, k))

    for i in range(y_len):
        for j in range(k):
            p = cost_mat[Y[i], j]
            cs_labels[i,j] = np.random.binomial(1, p)

    return cs_labels


def save_vw_dataset(X, y, dname, ds_dir, cs, tag=None):
    if cs:
        n_classes = np.shape(y)[1]
    else:
        n_classes = int(y.max() + 1)

    if tag is None:
        fname = 'ds_{}_{}.vw.gz'.format(dname, n_classes)
    else:
        fname = 'ds_{}_{}_{}.vw.gz'.format(dname, n_classes, tag)

    sparse = sp.isspmatrix_csr(X)

    with gzip.open(os.path.join(ds_dir, fname), 'wt') as f:
            for i in range(X.shape[0]):
                if cs:
                    for j in range(n_classes):
                        f.write('{}:{} '.format(j+1, y[i,j]))
                else:
                    f.write('{} '.format(y[i]+1))

                if sparse:
                    f.write(' | {}\n'.format(' '.join(
                    '{}:{:.6f}'.format(j, val) for j, val in zip(X[i].indices, X[i].data))))
                else:
                    f.write(' | {}\n'.format(' '.join(
                        '{}:{:.6f}'.format(j, val) for j, val in enumerate(X[i]) if val != 0)))

def shuffle(X, y):
    n = np.shape(X)[0]
    perm = np.random.permutation(n)
    X_shuf = X[perm, :]
    y_shuf = y[perm]
    return X_shuf, y_shuf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='openML to vw converter')
    parser.add_argument('min_did', type=int, default=0, help='minimum dataset id to process')
    parser.add_argument('max_did', type=int, default=None, help='maximum dataset id to process')
    parser.add_argument('--cs', action='store_true', help='cost sensitive label')
    parser.add_argument('--rand_cost', action='store_true', help='random cost matrices in the cost sensitive setting')

    args = parser.parse_args()
    print(args.min_did, ' to ', args.max_did)

    cs = args.cs
    rand_cost = args.rand_cost

    if not os.path.exists(VW_DS_DIR):
        os.makedirs(VW_DS_DIR)

    openml.config.apikey = '3411e20aff621cc890bf403f104ac4bc'
    openml.config.set_cache_directory(VW_DS_DIR+'/omlcache')

    print('loaded openML')

    dids = \
    [ 3, 6, 8, 10, 11, 12, 14, 16, 18, 20, 21, 22, 23, 26, 28, 30, 31, 32, 36, 37, 39, 40, 41,\
      43, 44, 46, 48, 50, 53, 54, 59, 60, 61, 62, 150, 151, 153, 154, 155, 156, 157, 158, 159,\
      160, 161, 162, 180, 181, 182, 183, 184, 187, 189,197, 209, 223, 227, 273, 275, 276, 277,\
      278, 279, 285, 287, 292, 293, 294, 298, 300, 307, 310, 312, 313, 329, 333, 334, 335, 336,\
      337, 338, 339, 343, 346, 351, 354, 357, 375, 377, 383, 384, 385, 386, 387, 388, 389, 390,\
      391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 444, 446, 448, 450, 457, 458, 459,\
      461, 462, 463, 464, 465, 467, 468, 469, 472, 475, 476, 477, 478, 479, 480, 554, 679, 682,\
      683, 685, 694, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727,\
      728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 740, 741, 742, 743, 744, 745, 746, 747,\
      748, 749, 750, 751, 752, 753, 754, 755, 756, 758, 759, 761, 762, 763, 764, 765, 766, 767,\
      768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 782, 783, 784, 785, 787,\
      788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 799, 800, 801, 803, 804, 805, 806, 807,\
      808, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827,\
      828, 829, 830, 832, 833, 834, 835, 836, 837, 838, 841, 843, 845, 846, 847, 848, 849, 850,\
      851, 853, 855, 857, 859, 860, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873,\
      874, 875, 876, 877, 878, 879, 880, 881, 882, 884, 885, 886, 888, 891, 892, 893, 894, 895,\
      896, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916,\
      917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 931, 932, 933, 934, 935,\
      936, 937, 938, 941, 942, 943, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956,\
      958, 959, 962, 964, 965, 969, 970, 971, 973, 974, 976, 977, 978, 979, 980, 983, 987, 988,\
      991, 994, 995, 996, 997, 1004, 1005, 1006, 1009, 1011, 1012, 1013, 1014, 1015, 1016, 1019,\
      1020, 1021, 1022, 1025, 1026, 1036, 1038, 1040, 1041, 1043, 1044, 1045, 1046, 1048, 1049,\
      1050, 1054, 1055, 1056, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069,\
      1071, 1073, 1075, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087,\
      1088, 1100, 1104, 1106, 1107, 1110, 1113, 1115, 1116, 1117, 1120, 1121, 1122, 1123,\
      1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1135, 1136, 1137, 1138,\
      1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152,\
      1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166,\
      1169, 1216, 1217, 1218, 1233, 1235, 1236, 1237, 1238, 1241, 1242, 1412, 1413, 1441, 1442,\
      1443, 1444, 1449, 1451, 1453, 1454, 1455, 1457, 1459, 1460, 1464, 1467, 1470, 1471, 1472,\
      1473, 1475, 1481, 1482, 1483, 1486, 1487, 1488, 1489, 1496, 1498]


    '''
    names_datasets =  [
        ('letter', 6),
        ('optdigits', 28),
        ('page-blocks', 30),
        ('pendigits', 32),
        ('satimage', 182),
        ('vehicle', 54),
        ('yeast', 181),
        ('adult', 1590)
        ]
    '''

    # min_did = 1200
    # max_did = 1500
    # sorted(dids)
    #for dname, did in names_datasets:
    for did in dids:
        dname = ''
        if did < args.min_did:
            continue
        if args.max_did is not None and did >= args.max_did:
            break
        print('processing did', did)
        try:
            ds = openml.datasets.get_dataset(did)
            X, y = ds.get_data(target=ds.default_target_attribute)
            nom_fts = ds.get_features_by_type('nominal')
            nom_fts = filter(lambda x: x < np.shape(X)[1], nom_fts)
            X = augment_nom_fts(X, nom_fts)
            X, y = shuffle(X, y)
        except Exception as e:
            print(e)
            continue

        if cs:
            tag = 'cs'
            n_classes = y.max() + 1
            if rand_cost:
                cost_mat = gen_cost_mat(n_classes)
                tag += '_randcost'
            else:
                cost_mat = np.ones((n_classes, n_classes)) - np.eye(n_classes)
                tag += '_zeroone'
            print(cost_mat)
            cs_labels = gen_cs_labels(y, cost_mat)
            print(X, cs_labels)

            save_vw_dataset(X, cs_labels, dname, VW_DS_DIR, True, tag)
        else:
            print(X, y)
            save_vw_dataset(X, y, did, VW_DS_DIR, False)

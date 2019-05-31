import scipy.stats as stats
from math import sqrt

def plot_all_pair_comp(alg_results, sizes, dir, header):
    alg_names = list(alg_results)

    for i in range(len(alg_names)):
        for j in range(len(alg_names)):
            if i < j:
                errs_1 = alg_results[alg_names[i]]
                errs_2 = alg_results[alg_names[j]]

                #print(len(errs_1), len(errs_2), len(sizes))
                num_wins_1, num_wins_2 = plot_comparison(errs_1, errs_2, sizes)
                plt.title( 'total number of comparisons = ' + str(len(errs_1)) + '\n'+
                alg_str(alg_names[i]) + ' wins ' + str(num_wins_1) + ' times, \n' + alg_str(alg_names[j]) + ' wins ' + str(num_wins_2) + ' times')
                plt.savefig(dir+alg_str_compatible(alg_names[i])+'_vs_'+alg_str_compatible(alg_names[j])+'.pdf')
                plt.close()


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

    significance = list(map(is_significant, z_scores))
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

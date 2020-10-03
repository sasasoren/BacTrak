import numpy as np
import vectorwise_func2 as vf2
from scipy.spatial import distance, Delaunay
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from skimage.measure import label
import multiprocessing
import istarmap
from collections import Counter


# img_num = 0
# relation_window_size = 45
# cell_num = 1
# growth_rate = 1.3
# epoch = 600
# T = 100
# dT = 99/100


def label_changer(lab, svec):
    """
    Change the label of the cells base on the length of the cells ascending
    :param lab: labeled image
    :param svec: array of length of the cells
    :return: new labeled image
    """
    new_lab = np.zeros(np.shape(lab)).astype(np.uint16)
    sordered = np.sort(svec)
    cvec = svec.copy()
    for i in range(len(sordered)):
        x = np.where(cvec == sordered[i])[0][0]
        new_lab[lab == x + 1] = i + 1
        cvec[x] = 0
    return new_lab


def mu_1(cent0, cent1):
    # This function will return norm 2 of cent0-cent1
    w_i = np.linalg.norm((cent0 - cent1))
    return w_i


def mu_2(v_i, v_j, g):
    """
    compute \mu_2 in the function for cell i and j
    :param v_i: vector v_i from I_0
    :param v_j: vector v_j from I_1
    :param g: growth rate
    :return: measure the growth rate
    """
    mu = np.square(np.log(np.linalg.norm(v_j) / np.linalg.norm(v_i * g)))

    return mu


def mu_3(v_i, v_j):
    """
    compute \mu_3 in the function for cell i and j
    :param v_i: vector v_i from I_0
    :param v_j: vector v_j from I_1
    :return: measure the rotation change
    """
    mu = np.min((np.abs(np.arccos(np.dot(v_i, v_j) / (np.linalg.norm(v_j) * np.linalg.norm(v_i)))),
                 np.abs(np.arccos(np.dot(-v_i, v_j) / (np.linalg.norm(v_j) * np.linalg.norm(v_i))))))

    if mu != mu:
        mu = 0

    return mu


def measure_finder(cent0, cent1, v_0, v_1, g, relate):
    """
    This function gets values and returns dictionary of the measures between cells of I_0 and I_1
    :param cent0: centroid I_9
    :param cent1: centroid I_1
    :param v_0: vectors I_0
    :param v_1: vectors I_1
    :param g: growth rate
    :return: dictionary of the measures
    """
    measure_dic = {}
    for num_ in range(1, len(cent0) + 1):
        temp_dic = {}
        for k in relate[num_]:
            temp_dic[k] = [mu_1(cent0[num_ - 1], cent1[k - 1]),
                           mu_2(v_0[num_ - 1], v_1[k - 1], g),
                           mu_3(v_0[num_ - 1], v_1[k - 1])]

        measure_dic[num_] = temp_dic.copy()

    return measure_dic


def set_for_probability_maker(m_dic, nall_=True, nall_val=2):
    """
    This function returns set of length that we will use to find the probability of each assignment
    :param m_dic: dictionary of the measures
    :param nall_: not all means if we wanted to consider a portion of the sets as output
    :param nall_val: length of the portion of the output
    :return: set for each measure as input to make probability distribution
    """
    m1_set = []
    m2_set = []
    m3_set = []
    for key1, val1 in m_dic.items():
        if nall_:
            temp_1 = []
            temp_2 = []
            temp_3 = []
            for key2, val2 in val1.items():
                temp_1.append(val2[0])
                temp_2.append(val2[1])
                temp_3.append(val2[2])
            for s in np.sort(temp_1)[:nall_val]:
                m1_set.append(s)
            for s in np.sort(temp_2)[:nall_val]:
                m2_set.append(s)
            for s in np.sort(temp_3)[:nall_val]:
                m3_set.append(s)

        else:
            for key2, val2 in val1.items():
                m1_set.append(val2[0])
                m2_set.append(val2[1])
                m3_set.append(val2[2])

    return [np.array(m1_set), np.array(m2_set), np.array(m3_set)]


def probability_finder(mcell, mtotal, not_zero=True, big_num=10**9):
    """
    this function return probability vector related to each pair of cells from I_0 to I_1
    :param mcell: measure cell vector for the cell i and j which we want to find the probability
    :param mtotal: all of the values for measure of the cell we want to find the probability based on
    :return: probability vector for the intended cell
    """
    prob_cel = []
    prob_total = 1
    for m in range(len(mcell)):
        if mcell[m] > np.sort(mtotal[m])[-1]:
            if not_zero:
                prob_cel.append(1/big_num)
                prob_total *= 1/big_num

            else:
                prob_cel.append(0)
                prob_total *= 0
        else:
            min_p = np.where(np.sort(mtotal[m]) == np.max(mtotal[m][mtotal[m] <= mcell[m]]))[0][0] + 1
            max_p = np.where(np.sort(mtotal[m]) == np.min(mtotal[m][mtotal[m] >= mcell[m]]))[0][-1] + 1

            cell_p = (min_p + max_p) / 2
            temp_p = 1 - (cell_p / len(mtotal[m]))
            if temp_p == 0:
                prob_cel.append(1 / len(mtotal[m]))

                prob_total *= 1 / len(mtotal[m])
            else:
                prob_cel.append(1 - (cell_p / len(mtotal[m])))

                prob_total *= 1 - (cell_p / len(mtotal[m]))

    return prob_cel, prob_total


def single_pwn(cn_i, cn_j, cent0, cent1, m_dic, mtotal, relate, one_comb, neig0, neig1, not_zero=True, big_num=10**9):
    """
    here we compute probability for cell number f(cn_i) = cn_j just for one combination of outputs
    of f(cn_i) = cn_j
    :param cn_i: cell number in I_0, integer
    :param cn_j: cell number in I_1 which we find P(f(cn_i)), integer
    :param m_dic: dictionary of all measures, dictionary of dictionaries output of measure finder
    :param mtotal: list of measure selected measures to find probability. output of set_for_probability_maker
    :param relate: relational dictionary which consist of cell numbers for Winkle window. dictionary
    :param one_comb: one combination for neighbors of the cell cn_i
    :return: Probability for one combination of f(cn_i) = cn_j
    """
    prob = probability_finder(m_dic[cn_i][cn_j], mtotal)[1]

    for y in one_comb:
        # Here we test if the neighbors will stay in the second image as neighbors
        num_same_nei = 0
        ln0 = list(np.intersect1d(neig0[cn_i], neig0[y[0]]))
        ln1 = list(np.intersect1d(neig1[cn_j], neig1[y[1]]))
        test_pd = list(itertools.product(ln0, ln1))

        base_vec0 = cent0[cn_i - 1] - cent0[y[0] - 1]
        base_vec1 = cent1[cn_j - 1] - cent1[y[1] - 1]

        vlogic = True
        # here we test if they flip
        for k in test_pd:
            if k in one_comb:
                num_same_nei += 1

                temp_vec0 = cent0[cn_i - 1] - cent0[k[0] - 1]
                temp_vec1 = cent1[cn_j - 1] - cent1[k[1] - 1]

                if np.cross(base_vec0, temp_vec0) * np.cross(base_vec1, temp_vec1) < 0:
                    vlogic = False

        if y[1] in relate[y[0]] and num_same_nei == len(ln0) and vlogic:
            prob *= probability_finder(m_dic[y[0]][y[1]], mtotal, not_zero, big_num)[1]

        else:
            if not_zero:
                prob *= 1/big_num

            else:
                prob *= 0

    return prob / (len(one_comb) + 1)


def prob_with_neighbors(cn_i, cn_j, cent0, cent1, m_dic, mtotal, nei0, nei1, relate, not_zero=True, big_num=10**9):
    """
    here we compute probability for cell number f(cn_i) = cn_j all the combination
    of f(cn_i) = cn_j
    :param cn_i: cell number in I_0, integer
    :param cn_j: cell number in I_1 which we find P(f(cn_i)), integer
    :param m_dic: dictionary of all measures, dictionary of dictionaries output of measure finder
    :param mtotal: list of measure selected measures to find probability. output of set_for_probability_maker
    :param nei0: neighbors in I_0: dictionary
    :param nei1: neighbors in I_1: dictionary
    :param relate: relational dictionary which consist of cell numbers for Winkle window. dictionary
    :return: list of possible probability for the cell cn_i
    """
    targ = [(cn_i, cn_j)]

    possible_list = [list(zip(x, nei1[cn_j])) for x in
                     itertools.permutations(nei0[cn_i], int(np.min((len(nei1[cn_j]), len(nei0[cn_i])))))]

    prob_list_acell = list(map(lambda x: [single_pwn(cn_i, cn_j, cent0, cent1, m_dic, mtotal, relate,
                                                     x, nei0, nei1, not_zero, big_num), x, [cn_i, cn_j]],
                               possible_list))

    min_prob = np.where(np.array(prob_list_acell, dtype=object)[:, 0] == np.max(np.array(prob_list_acell,
                                                                                         dtype=object)[:, 0]))[0]

    prob_rc = np.random.choice(min_prob)

    for it in prob_list_acell[prob_rc][1]:
        targ.append(it)

    return [prob_list_acell[prob_rc][0], targ]


def generate_colors(n):
    """
    find n different color
    :param n: number of the colors, integer
    :return: list of different colors
    """

    # hex_values = []
    r, g, b = 255, 0, 0
    rgb_values = [(r, g, b)]
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for _ in range(1, n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        # r_hex = hex(r)[2:]
        # g_hex = hex(g)[2:]
        # b_hex = hex(b)[2:]
        # hex_values.append('#' + r_hex + g_hex + b_hex)
        rgb_values.append((r, g, b))
    return rgb_values  # hex_values


def generate_colors2(n):
    """
    find n different color
    :param n: number of the colors, integer
    :return: list of different colors
    """

    # hex_values = []
    rgb_values = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for _ in range(n):
        r += step
        g += random.randint(0, 8) * step
        b += random.randint(1, 10) * step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        # r_hex = hex(r)[2:]
        # g_hex = hex(g)[2:]
        # b_hex = hex(b)[2:]
        # hex_values.append('#' + r_hex + g_hex + b_hex)
        rgb_values.append((r, g, b))
    return rgb_values  # hex_values


def static_generate_colors(n):
    """
    find n different color
    :param n: number of the colors, integer
    :return: list of different colors
    """

    # hex_values = []
    rgb_values = []
    r = 75
    g = 140
    b = 210
    step = 256 / n
    for _ in range(n):
        r += step
        g += 43 * step
        b += 11 * step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        # r_hex = hex(r)[2:]
        # g_hex = hex(g)[2:]
        # b_hex = hex(b)[2:]
        # hex_values.append('#' + r_hex + g_hex + b_hex)
        rgb_values.append((r, g, b))
    return rgb_values  # hex_values


def cell_painting(c_list, l0, l1):

    i1 = np.zeros((np.shape(l1)[0], np.shape(l1)[1], 3)).astype(np.uint8)
    i1[l1 > 0] = [255, 255, 255]
    i0 = np.zeros((np.shape(l0)[0], np.shape(l0)[1], 3)).astype(np.uint8)
    i0[l0 > 0] = [255, 255, 255]
    color_list = generate_colors(len(c_list))
    for c in range(len(c_list)):
        i0[l0 == c_list[c][0]] = color_list[c]
        i1[l1 == c_list[c][1]] = color_list[c]

    return i0, i1


def color_all_cells(tar, l0, l1):
    """
    This function will give you colored cells based on the prediction
    mapping tar, where the origin and the predict have the same color
    :param tar: prediction mapping
    :param l0: labeled image0
    :param l1: labeled image1
    :return: colored image0 and image1
    """
    i1 = np.zeros((np.shape(l1)[0], np.shape(l1)[1], 3)).astype(np.uint8)
    i1[l1 > 0] = [255, 255, 255]
    i0 = np.zeros((np.shape(l0)[0], np.shape(l0)[1], 3)).astype(np.uint8)
    i0[l0 > 0] = [255, 255, 255]
    color_list = static_generate_colors(len(tar))
    for k in range(1, len(tar) + 1):
        i0[l0 == k] = color_list[k - 1]
        i1[l1 == tar[k]] = color_list[k - 1]

    return i0, i1


def predict_overlap(base_func, tar_func, same_set=True):
    '''
    This function counts number of same segment chosen by targer function. if just two cell
    have the same target function value it will return 1 more than that it will grow w.r.t
    number of
    :param base_func:
    :param tar_func:
    :param same_set:
    :return: how many same segments have same target value
    '''

    counter_ = 0
    for key, cells in base_func.items():
        for c in cells:
            flag = False
            temp_dic = tar_func.copy()
            if same_set:
                del temp_dic[key]
            for cells2 in temp_dic.values():
                for c2 in cells2:
                    if c == c2:
                        flag = True

            if flag is True:
                counter_ += 1

    return counter_


def pred_not_neighbors(cell0, tar, neigh0, neigh1):
    """
    looks at cells and count how many neighbors of the cells will not stay as neighbors based on
    the prediction dictionary tar
    :param cell0: cell number in I_0
    :param tar: target function
    :param neigh0: neighbor dictionary in I_0
    :param neigh1: neighbor dictionary in I_1
    :return: ratio of the cell the won't stay as neighbor in the second image
    """
    cell0_tar = tar[cell0][0]
    counter_ = 0
    for i in neigh0[cell0]:
        cell_tar = tar[i][0]
        if cell_tar not in neigh1[cell0_tar]:
        # if cell_tar not in [y for x in neigh1[cell0_tar] for y in neigh1[x]] + neigh1[cell0_tar]:
            counter_ += 1

    return counter_ / len(neigh0[cell0]), counter_


def total_not_neighbors(tar_f, neigh0, neigh1):

    return np.sum(list(map(lambda x: pred_not_neighbors(x, tar_f,
                                                        neigh0,
                                                        neigh1)[0],
                           list(tar_f.keys()))))


def log_prob(tar_dic, lprob_t):
    pr = 0
    for k in tar_dic.keys():
        pr += lprob_t[k - 1, tar_dic[k][0] - 1]
    return -pr


def three_vec_flip(cn, c_nei, tar, cent0, cent1):
    """
    This function look at the order of the vector from center cn to two other center in c_nei is the order is
    the same it will return 0 otherwise if they flipped it will return 1
    :param cn: center of the two vectors/ one integer
    :param c_nei: destination of two vector/ list of two integer
    :param tar: target function/ dictionary
    :param cent0: centroid of segments in image 0/ list of centroid
    :param cent1: cebtroid of segments in image 1/ list of centroid
    :return: 0 if vectors don't flip, 1 if they flipped
    """
    base_vec0 = cent0[cn - 1] - cent0[c_nei[0] - 1]
    base_vec1 = cent1[tar[cn][0] - 1] - cent1[tar[c_nei[0]][0] - 1]

    temp_vec0 = cent0[cn - 1] - cent0[c_nei[1] - 1]
    temp_vec1 = cent1[tar[cn][0] - 1] - cent1[tar[c_nei[1]][0] - 1]

    if np.linalg.norm(temp_vec1) * np.linalg.norm(base_vec1) == 0:
        return 0

    if np.linalg.norm(temp_vec1) * np.linalg.norm(base_vec1) == 0:
        return 0

    deg0 = np.dot(base_vec0, temp_vec0) / (np.linalg.norm(temp_vec0) * np.linalg.norm(base_vec0))
    deg1 = np.dot(base_vec1, temp_vec1) / (np.linalg.norm(temp_vec1) * np.linalg.norm(base_vec1))

    if deg0 < -.8:
            return 0

    if deg1 < -.8:
        return 0

    if np.cross(base_vec0, temp_vec0) * np.cross(base_vec1, temp_vec1) < 0:
        return 1
    else:
        return 0


def flip_cost(tar, cent0, cent1, neigh0):
    """
    this function counts all of the flips of the vectors
    :param tar: target dictionary
    :param cent0:centroid for segments in image0
    :param cent1:centroid for segments in image1
    :param neigh0:neighbors in image0
    :return: number of flipped vectors
    """
    flip_num = 0
    for k in tar.keys():
        if len(neigh0[k]) > 1:
            comb = itertools.combinations(neigh0[k], 2)
            flip_num += np.sum(list(map(lambda x: three_vec_flip(k, x, tar, cent0, cent1), comb)))

    return flip_num


def total_cost(lam, tar_f, neigh0, neigh1, lprob_t, cent0, cent1, avg_nei, cnum=None, tar_n=None):
    """
    Cost function for the method
    :param lam:
    :param tar_f:
    :param neigh0:
    :param neigh1:
    :param lprob_t:
    :param cent0:
    :param cent1:
    :param avg_nei:
    :param cnum: cell number in I_0 which supposed to replace in target funtion and find the value of new
    cost function
    :param tar_n: replace in I_1
    :return:
    """
    if isinstance(cnum, int):
        tar = tar_f.copy()
        tar[cnum] = tar_n
    elif cnum:
        for n, cn in cnum:
            tar = tar_f.copy()
            tar[cn] = tar_n[n]
    else:
        tar = tar_f.copy()

    n0 = len(tar_f)

    alpha1 = log_prob(tar, lprob_t)

    alpha2 = predict_overlap(tar, tar, same_set=True)

    alpha3 = total_not_neighbors(tar, neigh0, neigh1)

    alpha4 = flip_cost(tar, cent0, cent1, neigh0)

    return (alpha1 / n0) + (lam[0] * alpha2 / n0) +\
           (lam[1] * alpha3 / (2 * n0)) + (lam[2] * alpha4 / (4 * n0 * avg_nei)),\
           alpha1, alpha2, alpha3, alpha4


def parameter_cost(tar_f, neigh0, neigh1, lprob_t, cent0, cent1, avg_nei, cnum=None, tar_n=None):
    """
    returns only parameters for the cost function
    :param tar_f:
    :param neigh0:
    :param neigh1:
    :param lprob_t:
    :param cent0:
    :param cent1:
    :param avg_nei:
    :param cnum: cell number in I_0 which supposed to replace in target funtion and find the value of new
    cost function
    :param tar_n: replace in I_1
    :return:
    """
    if cnum:
        tar = tar_f.copy()
        tar[cnum] = tar_n
    else:
        tar = tar_f.copy()

    n0 = len(tar_f)

    alpha1 = log_prob(tar, lprob_t) / n0

    alpha2 = 2 * predict_overlap(tar, tar, same_set=True) / (n0 * (n0 - 1))

    alpha3 = total_not_neighbors(tar, neigh0, neigh1) / (2 * n0)

    alpha4 = flip_cost(tar, cent0, cent1, neigh0) / (4 * n0 * avg_nei)

    return np.array((alpha1, alpha2, alpha3, alpha4))


def BM_implement(la, relation_window_size=45, growth_rate=1.3,
                 epoch=2000, T=1000, dT=.999, incresing=False,
                 step=10, epsilon=.0005, draw_=False):

    N0 = np.max(lab0)
    N1 = np.max(lab1)

    centr0, vector0 = vf2.cen_vec(img0, save_name="vector0", lab_img=lab0)
    centr1, vector1 = vf2.cen_vec(img1, save_name="vector1", lab_img=lab1)

    relation = {}
    for i in range(1, N0 + 1):
        _, cen_list = vf2.close_cen(centr1,
                                    centr0[i - 1],
                                    win_size=relation_window_size)
        relation[i] = np.int32(cen_list) + 1

    # lvector0 = np.linalg.norm(vector0, axis=1)
    # lvector1 = np.linalg.norm(vector1, axis=1)
    #
    tri0 = Delaunay(centr0)
    tri1 = Delaunay(centr1)

    neighbors0 = {}
    neighbors1 = {}

    neighbors0_nonan = {}
    neighbors1_nonan = {}

    # for i in range(1, N0 + 1):
    #     neighbors0[i], neighbors0_nonan[i] = vf2.neigh_dis(centr0[i - 1],
    #                                                        i - 1, centr0,
    #                                                        tri0, bound_dis=60)
    # for i in range(1, N1 + 1):
    #     neighbors1[i], neighbors1_nonan[i] = vf2.neigh_dis(centr1[i - 1],
    #                                                        i - 1, centr1,
    #                                                        tri1, bound_dis=60)

    for i in range(1, N0 + 1):
        neighbors0[i], neighbors0_nonan[i] = vf2.neigh_without_cell_cut(i - 1, centr0,
                                                                        tri0, lab0,
                                                                        bound_dis=50)
    for i in range(1, N1 + 1):
        neighbors1[i], neighbors1_nonan[i] = vf2.neigh_without_cell_cut(i - 1, centr1,
                                                                        tri1, lab1,
                                                                        bound_dis=50)

    avg_neigh_num = np.mean([len(x) for x in neighbors0_nonan.values()])
    # print('avg neigh:', avg_neigh_num)

    # measure_dictionary = measure_finder(centr0, centr1, vector0, vector1, growth_rate, relation)
    #
    # m_set = set_for_probability_maker(measure_dictionary, nall_=True, nall_val=2)
    #
    # prob_table = np.zeros((N0, N1))
    #
    # for seg_num in tqdm(range(1, N0 + 1)):
    #     possibility_data = list(map(lambda x: prob_with_neighbors(seg_num, x, centr0, centr1,
    #                                                               measure_dictionary,
    #                                                               m_set, neighbors0_nonan,
    #                                                               neighbors1_nonan, relation,
    #                                                               not_zero=True, big_num=10**9),
    #                                 relation[seg_num]))
    #
    #     for num in range(len(possibility_data)):
    #         prob_table[seg_num - 1, possibility_data[num][1][0][1] - 1] = possibility_data[num][0]
    #
    # lprob_table = np.log(prob_table)
    # np.save('logprob_table.npy', lprob_table)

    # lprob_table = np.load('lprob_table.npy')
    # prob_table = lprob_table.copy()


    # plt.hist(m_set[2], bins=40)
    # plt.savefig('/media/sorena/VBox/Doc/Research/WeeklyReport/2020-01-24/Images/{}/mu3_2.png'.format(img_num))
    # plt.show()
    # plt.hist(m_set[1], bins=40)
    # plt.savefig('/media/sorena/VBox/Doc/Research/WeeklyReport/2020-01-24/Images/{}/mu2_2.png'.format(img_num))
    # plt.show()
    # plt.hist(m_set[0], bins=40)
    # plt.savefig('/media/sorena/VBox/Doc/Research/WeeklyReport/2020-01-24/Images/{}/mu1_2.png'.format(img_num))
    # plt.show()
    #
    # m_set = set_for_probability_maker(measure_dictionary, nall_=False)
    #
    # plt.hist(m_set[2], bins=40)
    # plt.savefig('/media/sorena/VBox/Doc/Research/WeeklyReport/2020-01-24/Images/{}/mu3_all.png'.format(img_num))
    # plt.show()
    # plt.hist(m_set[1], bins=40)
    # plt.savefig('/media/sorena/VBox/Doc/Research/WeeklyReport/2020-01-24/Images/{}/mu2_all.png'.format(img_num))
    # plt.show()
    # plt.hist(m_set[0], bins=40)
    # plt.savefig('/media/sorena/VBox/Doc/Research/WeeklyReport/2020-01-24/Images/{}/mu1_all.png'.format(img_num))
    # plt.show()

    #
    # A = prob_with_neighbors(1, relation[1][2], measure_dictionary,
    #                         m_set, neighbors0_nonan, neighbors1_nonan, relation)
    #
    # B = prob_with_neighbors(1, relation[1][1], measure_dictionary,
    #                         m_set, neighbors0_nonan, neighbors1_nonan, relation)

    target_function = {}
    for num in relation:
        target_function[num] = [np.where(prob_table[num - 1, :] == np.max(prob_table[num - 1, :]))[0][0] + 1]
        # target_function[num] = vf2.choose_target_no_split(relation, num, [])

    # cell_num = len(target_function) + 1
    # target_function2 = target_function.copy()

    # costs = np.zeros([epoch + 1, 5])
    # old_cost = total_cost(la, target_function, neighbors0_nonan, neighbors1_nonan, lprob_table,
    #                       centr0, centr1, avg_neigh_num)
    # costs[0, :] = old_cost
    #
    # for inter_num in tqdm(range(epoch)):
    #     for num_ in range(1, cell_num):
    #         old_pred = target_function[num_]
    #         new_pred = vf2.choose_target_no_split(relation, num_, old_pred)
    #
    #         temp_cost = total_cost(la, target_function, neighbors0_nonan, neighbors1_nonan, lprob_table,
    #                                centr0, centr1, avg_neigh_num, num_, new_pred)
    #
    #         delta_cost = temp_cost[0] - old_cost[0]
    #
    #         if delta_cost < 0:
    #             target_function[num_] = new_pred
    #             old_cost = temp_cost
    #         else:
    #             u = np.random.uniform()
    #             if u < np.exp(-delta_cost / T):
    #                 target_function[num_] = new_pred
    #                 old_cost = temp_cost
    #
    #     costs[inter_num + 1, :] = old_cost
    #
    #     T = T * dT

    # if incresing:
    #     la = np.array(la) / step
    #     step_la = la.copy()
    #     milestones = np.arange(1, step) * epoch / step
    #     lam_change = 0
    #     last_change = 0

    cell_num = len(target_function) + 1
    costs = np.zeros([epoch + 1, 11])
    old_cost = total_cost(la, target_function, neighbors0_nonan, neighbors1_nonan, lprob_table, centr0,
                          centr1, avg_neigh_num)

    costs[0, 0:5] = old_cost
    costs[0, 5:8] = la
    costs[0, 8] = 0
    delta_e = np.zeros([epoch * N0, 2])
    for inter_num in range(epoch):
        for num_ in range(1, cell_num):
            old_pred = target_function[num_]
            new_pred = vf2.choose_target_no_split(relation, num_, old_pred)
            temp_cost = total_cost(la, target_function, neighbors0_nonan, neighbors1_nonan, lprob_table,
                                   centr0, centr1, avg_neigh_num, num_, new_pred)
            delta_cost = temp_cost[0] - old_cost[0]
            if delta_cost < 0:
                target_function[num_] = new_pred
                old_cost = temp_cost
            else:
                delta_e[inter_num * N0 + num_ - 1, 0] = delta_cost
                u = np.random.uniform()
                if u < np.exp(-delta_cost / T):
                    delta_e[inter_num * N0 + num_ - 1, 1] = np.exp(-delta_cost / T)
                    target_function[num_] = new_pred
                    old_cost = temp_cost
        costs[inter_num + 1, 0:5] = old_cost
        # if incresing and lam_change < len(milestones):
        #     if (inter_num - last_change > 10) and (np.abs(costs[inter_num + 1, 0] - costs[inter_num - 9, 0]) < epsilon
        #                                            or inter_num > milestones[lam_change]):
        #         lam_change += 1
        #         la += step_la
        #         last_change = inter_num.copy()
        #
        # elif incresing and lam_change >= len(milestones):
        #     if (inter_num - last_change > 10) and np.abs(costs[inter_num + 1, 0] - costs[inter_num - 9, 0]) < epsilon:
        #         costs[inter_num + 1, 5:8] = la
        #         costs[inter_num + 1, 8] = (costs[inter_num, 0] - costs[inter_num + 1, 0]) / T
        #         return target_function, costs
        costs[inter_num + 1, 5:8] = la
        costs[inter_num + 1, 8] = (costs[inter_num, 0] - costs[inter_num + 1, 0])/T
        costs[inter_num + 1, 9] = costs[inter_num + 1, 0] - costs[inter_num, 0]
        costs[inter_num + 1, 10] = T

        if draw_ and (inter_num + 1) % 100 == 0:
            for pnum in range(6):
                plt.plot(costs[0:inter_num + 1, pnum])
                plt.show()
            print(acc_fun(true_predict, target_function))

        if inter_num > 50 and np.sum(np.abs(np.diff(costs[inter_num - 30:inter_num + 1, 0]))) < epsilon:
            return acc_fun(true_predict, target_function), target_function, costs, la,\
                   delta_e[:((inter_num + 1) * N0), :]

        T = T * dT

    return acc_fun(true_predict, target_function), target_function, costs, la, delta_e


def whole_for_multiprocess(lam0, lam1, lam2, avg_nei, epoch=1000, T=100, dT=99/100,
                           incresing=True, step=10, epsilon=.0005):
    """
    This function gets variables and use the boltzmann machine on the system and return the result
    of the target function and costs values
    :param lam0:
    :param lam1:
    :param lam2:
    :param lam3:
    :param avg_nei: average number of the neighbors
    :param epoch: number of epochs
    :param T: initial temprature
    :param dT: decreasing amount of temprature
    :param incresing: if we want to increase the values of lambda
    :param step: number of steps to increase the value of the lambda
    :return: target function and costs values for each epoch and each parameters
    """
    if incresing:
        la = np.array([lam0, lam1, lam2]) / step
        milestones = np.arange(1, step) * epoch / step
        lam_change = 0
    else:
        la = np.array([lam0, lam1, lam2])
    target_function = {}
    for num in relation:
        target_function[num] = [np.where(prob_table[num - 1, :] == np.max(prob_table[num - 1, :]))[0][0] + 1]
        # target_function[num] = vf2.choose_target_no_split(relation, num, [])
    cell_num = len(target_function) + 1
    # target_function2 = target_function.copy()
    costs = np.zeros([epoch + 1, 11])
    old_cost = total_cost(la, target_function, neighbors0_nonan, neighbors1_nonan, lprob_table, centroid0,
                          centroid1, avg_nei)
    costs[0, 0:5] = old_cost
    costs[0, 5:8] = la
    costs[0, 8] = 0
    for inter_num in range(epoch):
        for num_ in range(1, cell_num):
            old_pred = target_function[num_]
            new_pred = vf2.choose_target_no_split(relation, num_, old_pred)
            temp_cost = total_cost(la, target_function, neighbors0_nonan, neighbors1_nonan, lprob_table,
                                   centroid0, centroid1, avg_nei, num_, new_pred)
            delta_cost = temp_cost[0] - old_cost[0]
            if delta_cost < 0:
                target_function[num_] = new_pred
                old_cost = temp_cost
            else:
                u = np.random.uniform()
                if u < np.exp(-delta_cost / T):
                    target_function[num_] = new_pred
                    old_cost = temp_cost
        costs[inter_num + 1, 0:5] = old_cost
        if incresing and lam_change < len(milestones):
            if (costs[inter_num + 1, 0] - costs[inter_num - 9, 0] < epsilon or
                                        epoch > milestones[lam_change]):
                lam_change += 1
                la += np.array([lam0, lam1, lam2]) / step

        elif incresing and lam_change >= len(milestones):
            if costs[inter_num + 1, 0] - costs[inter_num - 9, 0] < epsilon:
                costs[inter_num + 1, 5:8] = la
                costs[inter_num + 1, 8] = -(costs[inter_num + 1, 0] - costs[inter_num, 0]) / T
                return acc_fun(true_predict, target_function), target_function, costs, [lam0, lam1, lam2]

        costs[inter_num + 1, 5:8] = la
        costs[inter_num + 1, 8] = -(costs[inter_num + 1, 0] - costs[inter_num, 0])/T

        T = T * dT
    # i0, i1 = color_all_cells(target_function, lab0, lab1)
    # plt.imshow(i0)
    # # plt.savefig('/media/sorena/VBox/Doc/Research/WeeklyReport/2020-02-07/Images/'
    # #             'multi_lam/i0-lam1-{}-lam2-{}.png'.format(lam1, lam2))
    # plt.savefig('/home/sorena/servers/storage/SS/Images/Samples/'
    #             'i0-lam1-{}-lam2-{}-lam3-{}.png'.format(lam1, lam2, lam3))
    # plt.show()
    # plt.imshow(i1)
    # # plt.savefig('/media/sorena/VBox/Doc/Research/WeeklyReport/2020-02-07/Images/'
    # #             'multi_lam/i1-lam1-{}-lam2-{}.png'.format(lam1, lam2))
    # plt.savefig('/home/sorena/servers/storage/SS/Images/Samples/'
    #             'i1-lam1-{}-lam2-{}-lam3-{}.png'.format(lam1, lam2, lam3))
    # plt.show()
    return acc_fun(true_predict, target_function), target_function, costs, [lam0, lam1, lam2]


def acc_fun(true_dic, pred_dic):
    """
    here we compare true known dictionary as true_dic and predicted dictionary
    :param true_dic: known dictionary
    :param pred_dic: predicted dictionary
    :return: accuracy/float, list of wrong predicted/list-
    """
    truenum = 0
    wrong_list = []
    n0 = len(true_dic)
    for nu in range(1, n0 + 1):
        if true_dic[nu][0] == pred_dic[nu][0]:
            truenum += 1
        else:
            wrong_list.append(nu)

    return truenum / n0, wrong_list


def neigh_not_result(tar, l0, l1, nei0, nei1, address_=False):
    """
    this function show which neighbors are not consider as correct for the formula in prediction of
    target function
    :param tar: target function
    :param l0: lab0
    :param l1: lab1
    :param nei0: neighbors0
    :param nei1: neighbors1
    :param address_: address if you want to save the images
    :return: show which cell won't stand as the neighbors in the second image
    """
    i1 = np.zeros((np.shape(l1)[0], np.shape(l1)[1], 3)).astype(np.uint8)
    i1[l1 > 0] = [255, 255, 255]
    i0 = np.zeros((np.shape(l0)[0], np.shape(l0)[1], 3)).astype(np.uint8)
    i0[l0 > 0] = [255, 255, 255]
    counter_ = 0
    for i in range(1, np.max(l0) + 1):
        for j in nei0[i]:
            if tar[j][0] not in nei1[tar[i][0]]:
            # if tar[j][0] not in [y for x in nei1[tar[i][0]] for y in nei1[x]] + nei1[tar[i][0]]:
                counter_ += 1
                ii0 = i0.copy()
                ii1 = i1.copy()
                ii0[l0 == i] = [255, 0, 0]
                for w in nei0[i]:
                    ii0[l0 == w] = [200, 200, 0]
                ii0[l0 == j] = [0, 255, 0]
                ii1[l1 == tar[i][0]] = [255, 0, 0]
                for w in nei1[tar[i][0]]:
                    ii1[l1 == w] = [200, 200, 0]
                ii1[l1 == tar[j][0]] = [0, 255, 0]
                plt.imshow(ii0)
                if address_:
                    plt.savefig(address_ + 'i0{}.png'.format(counter_))
                plt.show()
                plt.imshow(ii1)
                if address_:
                    plt.savefig(address_ + 'i1{}.png'.format(counter_))
                plt.show()

    print('number of the cells which they were not stand as the neighbor in the second image is:', counter_)


def return_cost_param(lam, tar_f, neigh0, neigh1, lprob_t, cent0,
                      cent1, avg_nei, cnum=None, tar_n=None):
    """
    this function return parameters if you change mapping function at cnum to
    new value for tar_n
    :param lam:
    :param tar_f:
    :param neigh0:
    :param neigh1:
    :param lprob_t:
    :param cent0:
    :param cent1:
    :param avg_nei:
    :param cnum:
    :param tar_n:
    :return:
    """
    n0 = len(tar_f)
    dcost = np.array(total_cost(lam, tar_f, neigh0, neigh1, lprob_t, cent0,
                          cent1, avg_nei, cnum, tar_n)) -\
            np.array(total_cost(lam, tar_f, neigh0, neigh1, lprob_t, cent0,
                          cent1, avg_nei))

    return dcost[0], dcost[1] * n0, dcost[2] * n0 * (n0 - 1)/2,\
           dcost[3] * 2 * n0, dcost[4] * 4 * n0 * avg_nei


def complete_bm(l1, l2, l3, img_seg0, img_seg1, lab_img0, lab_img1,
                true_pred, rel_window=45, g_rate=1.3,
                epoch=2000, T=1000, dT=.99, epsilon=.005,
                nei_win=100, draw_=False):
    la = (l1, l2, l3)
    N0 = np.max(lab_img0)
    N1 = np.max(lab_img1)

    centr0, vector0 = vf2.cen_vec(img_seg0, save_name="vector0", lab_img=lab_img0)
    centr1, vector1 = vf2.cen_vec(img_seg1, save_name="vector1", lab_img=lab_img1)

    relation = {}
    for i in range(1, N0 + 1):
        _, cen_list = vf2.close_cen(centr1,
                                    centr0[i - 1],
                                    win_size=rel_window)
        relation[i] = np.int32(cen_list) + 1

    tri0 = Delaunay(centr0)
    tri1 = Delaunay(centr1)

    neighbors0_nonan = {}
    neighbors1_nonan = {}

    for i in range(1, N0 + 1):
        _, neighbors0_nonan[i] = vf2.neigh_without_cell_cut(i - 1, centr0,
                                                            tri0, lab_img0,
                                                            bound_dis=nei_win)
    for i in range(1, N1 + 1):
        _, neighbors1_nonan[i] = vf2.neigh_without_cell_cut(i - 1, centr1,
                                                            tri1, lab_img1,
                                                            bound_dis=nei_win)

    avg_neigh_num = np.mean([len(x) for x in neighbors0_nonan.values()])

    measure_dictionary = measure_finder(centr0, centr1, vector0, vector1, g_rate, relation)

    m_set = set_for_probability_maker(measure_dictionary, nall_=True, nall_val=2)

    prob_table = np.zeros((N0, N1))

    for seg_num in range(1, N0 + 1):
        possibility_data = list(map(lambda x: prob_with_neighbors(seg_num, x, centr0, centr1,
                                                                  measure_dictionary,
                                                                  m_set, neighbors0_nonan,
                                                                  neighbors1_nonan, relation,
                                                                  not_zero=True, big_num=10**9),
                                    relation[seg_num]))

        for num in range(len(possibility_data)):
            prob_table[seg_num - 1, possibility_data[num][1][0][1] - 1] = possibility_data[num][0]

    lprob_table = np.log(prob_table)
    # np.save('logprob_table.npy', lprob_table)

    # lprob_table = np.load('lprob_table.npy')
    # prob_table = lprob_table.copy()

    target_function = {}
    for num in relation:
        target_function[num] = [np.where(prob_table[num - 1, :] == np.max(prob_table[num - 1, :]))[0][0] + 1]

    cell_num = len(target_function) + 1
    costs = np.zeros([epoch + 1, 11])
    old_cost = total_cost(la, target_function, neighbors0_nonan, neighbors1_nonan, lprob_table, centr0,
                          centr1, avg_neigh_num)

    costs[0, 0:5] = old_cost
    costs[0, 5:8] = la
    costs[0, 8] = 0
    delta_e = np.zeros([epoch * N0, 2])
    for inter_num in range(epoch):
        for num_ in range(1, cell_num):
            old_pred = target_function[num_]
            new_pred = vf2.choose_target_no_split(relation, num_, old_pred)
            temp_cost = total_cost(la, target_function, neighbors0_nonan, neighbors1_nonan, lprob_table,
                                   centr0, centr1, avg_neigh_num, num_, new_pred)
            delta_cost = temp_cost[0] - old_cost[0]
            if delta_cost < 0:
                target_function[num_] = new_pred
                old_cost = temp_cost
            else:
                delta_e[inter_num * N0 + num_ - 1, 0] = delta_cost
                u = np.random.uniform()
                if u < np.exp(-delta_cost / T):
                    delta_e[inter_num * N0 + num_ - 1, 1] = np.exp(-delta_cost / T)
                    target_function[num_] = new_pred
                    old_cost = temp_cost
        costs[inter_num + 1, 0:5] = old_cost
        costs[inter_num + 1, 5:8] = la
        costs[inter_num + 1, 8] = (costs[inter_num, 0] - costs[inter_num + 1, 0])/T
        costs[inter_num + 1, 9] = costs[inter_num + 1, 0] - costs[inter_num, 0]
        costs[inter_num + 1, 10] = T

        if draw_ and (inter_num + 1) % 100 == 0:
            for pnum in range(6):
                plt.plot(costs[0:inter_num + 1, pnum])
                plt.show()
            if true_pred:
                print(acc_fun(true_pred, target_function))

        if inter_num > 100 and np.sum(np.abs(np.diff(costs[inter_num - 30:inter_num + 1, 0]))) <\
                epsilon * costs[inter_num - 100, 0]:
            return acc_fun(true_pred, target_function), target_function, costs, la,\
                   delta_e[:((inter_num + 1) * N0), :]

        T = T * dT

    return acc_fun(true_pred, target_function), target_function, costs, la, delta_e


def cost_true_map(img_seg0, img_seg1, lab_img0, lab_img1, true_pred,
                  rel_window=45, g_rate=1.3, la=(1, 1, 1)):
    """
    This function is for finding all parameters for cost function and average numbers
    of neighbors
    :param img_seg0:
    :param img_seg1:
    :param lab_img0:
    :param lab_img1:
    :param true_pred:
    :param rel_window:
    :param g_rate:
    :param la:
    :return:
    """
    N0 = np.max(lab_img0)
    N1 = np.max(lab_img1)

    centr0, vector0 = vf2.cen_vec(img_seg0, save_name="vector0", lab_img=lab_img0)
    centr1, vector1 = vf2.cen_vec(img_seg1, save_name="vector1", lab_img=lab_img1)

    relation = {}
    for i in range(1, N0 + 1):
        _, cen_list = vf2.close_cen(centr1,
                                    centr0[i - 1],
                                    win_size=rel_window)
        relation[i] = np.int32(cen_list) + 1

    tri0 = Delaunay(centr0)
    tri1 = Delaunay(centr1)

    neighbors0_nonan = {}
    neighbors1_nonan = {}

    for i in range(1, N0 + 1):
        _, neighbors0_nonan[i] = vf2.neigh_without_cell_cut(i - 1, centr0,
                                                            tri0, lab_img0,
                                                            bound_dis=50)
    for i in range(1, N1 + 1):
        _, neighbors1_nonan[i] = vf2.neigh_without_cell_cut(i - 1, centr1,
                                                            tri1, lab_img1,
                                                            bound_dis=50)

    avg_num = np.mean([len(x) for x in neighbors0_nonan.values()])

    measure_dictionary = measure_finder(centr0, centr1, vector0, vector1, g_rate, relation)

    m_set = set_for_probability_maker(measure_dictionary, nall_=True, nall_val=2)

    prob_table = np.zeros((N0, N1))

    for seg_num in range(1, N0 + 1):
        possibility_data = list(map(lambda x: prob_with_neighbors(seg_num, x, centr0, centr1,
                                                                  measure_dictionary,
                                                                  m_set, neighbors0_nonan,
                                                                  neighbors1_nonan, relation,
                                                                  not_zero=True, big_num=10**9),
                                    relation[seg_num]))

        for num in range(len(possibility_data)):
            prob_table[seg_num - 1, possibility_data[num][1][0][1] - 1] = possibility_data[num][0]

    lprob_table = np.log(prob_table)
    
    old_cost = total_cost(la, true_pred, neighbors0_nonan,
                          neighbors1_nonan, lprob_table, centr0,
                          centr1, avg_num)

    return old_cost, avg_num


def major_vote_res(res_file):
    """
    This function return the majority vote over multiple mapping (prediction) dictionary
    :param res_file:
    :return:
    """
    tem_dic = {}
    for i in range(1, len(res_file[0][1]) + 1):
        tem_dic[i] = []
    for j in range(len(res_file)):
        for i in range(1, len(res_file[0][1]) + 1):
            tem_dic[i].append(res_file[j][1][i][0])

    for i in range(1, len(tem_dic) + 1):
        tem_dic[i] = [Counter(tem_dic[i]).most_common()[0][0]]

    return tem_dic


def sep_param_from_result(res):
    """
    here we find the different parameters for
    :param res:
    :return:
    """
    stop_t = []
    par0 = []
    par1 = []
    par2 = []
    par3 = []
    par4 = []
    for i in range(len(res)):
        if not np.where(res[i][2][:, 0] == 0)[0].any():
            sopt_min = len(res[i][2][:, 0]) - 1
            stop_t.append(sopt_min)
            par0.append(res[i][2][sopt_min, 0])
            par1.append(res[i][2][sopt_min, 1])
            par2.append(res[i][2][sopt_min, 2])
            par3.append(res[i][2][sopt_min, 3])
            par4.append(res[i][2][sopt_min, 4])
        else:
            stop_min = np.min(np.where(res[i][2][:, 0] == 0)[0])
            stop_t.append(stop_min)
            par0.append(np.min(res[i][2][stop_min - 1, 0]))
            par1.append(np.min(res[i][2][stop_min - 1, 1]))
            par2.append(np.min(res[i][2][stop_min - 1, 2]))
            par3.append(np.min(res[i][2][stop_min - 1, 3]))
            par4.append(np.min(res[i][2][stop_min - 1, 4]))

    return par0, par1, par2, par3, par4, stop_t


def percent_finder(par, percent_, less=True):
    """
    this function looks at the all distribution in the par and looks at the first percent_
    percent of the distibutions to all distributions and return the intersection of them
    :param par: distributions
    :param percent_: intended percent to look at them
    :param less: if true means less than the percent_ if False means more than that
    :return: intersectoin of the first or last percent of the all par distributions
    """
    sets = []
    for pnum in range(len(par)):
        if less:
            sets.append(set(np.where(par[pnum] < np.quantile(par[pnum], percent_))[0]))
        else:
            sets.append(set(np.where(par[pnum] > np.quantile(par[pnum], percent_))[0]))

    return list(set.intersection(*sets))


def BM_implement2(la, relation_window_size=45, growth_rate=1.3,
                  epoch=1000, T=1000, dT=.999, incresing=False,
                  step=10, epsilon=.0005, draw_=False):

    N0 = np.max(lab0)
    N1 = np.max(lab1)

    centr0, vector0 = vf2.cen_vec(img0, save_name="vector0", lab_img=lab0)
    centr1, vector1 = vf2.cen_vec(img1, save_name="vector1", lab_img=lab1)

    relation = {}
    for i in range(1, N0 + 1):
        _, cen_list = vf2.close_cen(centr1,
                                    centr0[i - 1],
                                    win_size=relation_window_size)
        relation[i] = np.int32(cen_list) + 1

    # lvector0 = np.linalg.norm(vector0, axis=1)
    # lvector1 = np.linalg.norm(vector1, axis=1)
    #
    tri0 = Delaunay(centr0)
    tri1 = Delaunay(centr1)

    neighbors0 = {}
    neighbors1 = {}

    neighbors0_nonan = {}
    neighbors1_nonan = {}

    # for i in range(1, N0 + 1):
    #     neighbors0[i], neighbors0_nonan[i] = vf2.neigh_dis(centr0[i - 1],
    #                                                        i - 1, centr0,
    #                                                        tri0, bound_dis=60)
    # for i in range(1, N1 + 1):
    #     neighbors1[i], neighbors1_nonan[i] = vf2.neigh_dis(centr1[i - 1],
    #                                                        i - 1, centr1,
    #                                                        tri1, bound_dis=60)

    for i in range(1, N0 + 1):
        neighbors0[i], neighbors0_nonan[i] = vf2.neigh_without_cell_cut(i - 1, centr0,
                                                                        tri0, lab0,
                                                                        bound_dis=50)
    for i in range(1, N1 + 1):
        neighbors1[i], neighbors1_nonan[i] = vf2.neigh_without_cell_cut(i - 1, centr1,
                                                                        tri1, lab1,
                                                                        bound_dis=50)

    avg_neigh_num = np.mean([len(x) for x in neighbors0_nonan.values()])

    # measure_dictionary = measure_finder(centr0, centr1, vector0, vector1, growth_rate, relation)
    #
    # m_set = set_for_probability_maker(measure_dictionary, nall_=True, nall_val=2)
    #
    # prob_table = np.zeros((N0, N1))
    #
    # for seg_num in tqdm(range(1, N0 + 1)):
    #     possibility_data = list(map(lambda x: prob_with_neighbors(seg_num, x, centr0, centr1,
    #                                                               measure_dictionary,
    #                                                               m_set, neighbors0_nonan,
    #                                                               neighbors1_nonan, relation,
    #                                                               not_zero=True, big_num=10**9),
    #                                 relation[seg_num]))
    #
    #     for num in range(len(possibility_data)):
    #         prob_table[seg_num - 1, possibility_data[num][1][0][1] - 1] = possibility_data[num][0]
    #
    # lprob_table = np.log(prob_table)
    # np.save('logprob_table.npy', lprob_table)

    # lprob_table = np.load('lprob_table.npy')
    # prob_table = lprob_table.copy()


    # plt.hist(m_set[2], bins=40)
    # plt.savefig('/media/sorena/VBox/Doc/Research/WeeklyReport/2020-01-24/Images/{}/mu3_2.png'.format(img_num))
    # plt.show()
    # plt.hist(m_set[1], bins=40)
    # plt.savefig('/media/sorena/VBox/Doc/Research/WeeklyReport/2020-01-24/Images/{}/mu2_2.png'.format(img_num))
    # plt.show()
    # plt.hist(m_set[0], bins=40)
    # plt.savefig('/media/sorena/VBox/Doc/Research/WeeklyReport/2020-01-24/Images/{}/mu1_2.png'.format(img_num))
    # plt.show()
    #
    # m_set = set_for_probability_maker(measure_dictionary, nall_=False)
    #
    # plt.hist(m_set[2], bins=40)
    # plt.savefig('/media/sorena/VBox/Doc/Research/WeeklyReport/2020-01-24/Images/{}/mu3_all.png'.format(img_num))
    # plt.show()
    # plt.hist(m_set[1], bins=40)
    # plt.savefig('/media/sorena/VBox/Doc/Research/WeeklyReport/2020-01-24/Images/{}/mu2_all.png'.format(img_num))
    # plt.show()
    # plt.hist(m_set[0], bins=40)
    # plt.savefig('/media/sorena/VBox/Doc/Research/WeeklyReport/2020-01-24/Images/{}/mu1_all.png'.format(img_num))
    # plt.show()

    #
    # A = prob_with_neighbors(1, relation[1][2], measure_dictionary,
    #                         m_set, neighbors0_nonan, neighbors1_nonan, relation)
    #
    # B = prob_with_neighbors(1, relation[1][1], measure_dictionary,
    #                         m_set, neighbors0_nonan, neighbors1_nonan, relation)

    target_function = {}
    for num in relation:
        target_function[num] = [np.where(prob_table[num - 1, :] == np.max(prob_table[num - 1, :]))[0][0] + 1]
        # target_function[num] = vf2.choose_target_no_split(relation, num, [])

    # cell_num = len(target_function) + 1
    # target_function2 = target_function.copy()

    # costs = np.zeros([epoch + 1, 5])
    # old_cost = total_cost(la, target_function, neighbors0_nonan, neighbors1_nonan, lprob_table,
    #                       centr0, centr1, avg_neigh_num)
    # costs[0, :] = old_cost
    #
    # for inter_num in tqdm(range(epoch)):
    #     for num_ in range(1, cell_num):
    #         old_pred = target_function[num_]
    #         new_pred = vf2.choose_target_no_split(relation, num_, old_pred)
    #
    #         temp_cost = total_cost(la, target_function, neighbors0_nonan, neighbors1_nonan, lprob_table,
    #                                centr0, centr1, avg_neigh_num, num_, new_pred)
    #
    #         delta_cost = temp_cost[0] - old_cost[0]
    #
    #         if delta_cost < 0:
    #             target_function[num_] = new_pred
    #             old_cost = temp_cost
    #         else:
    #             u = np.random.uniform()
    #             if u < np.exp(-delta_cost / T):
    #                 target_function[num_] = new_pred
    #                 old_cost = temp_cost
    #
    #     costs[inter_num + 1, :] = old_cost
    #
    #     T = T * dT

    # if incresing:
    #     la = np.array(la) / step
    #     step_la = la.copy()
    #     milestones = np.arange(1, step) * epoch / step
    #     lam_change = 0
    #     last_change = 0

    cell_num = len(target_function) + 1
    costs = np.zeros([epoch + 1, 11])
    old_cost = total_cost(la, target_function, neighbors0_nonan, neighbors1_nonan, lprob_table, centr0,
                          centr1, avg_neigh_num)
    costs[0, 0:5] = old_cost
    costs[0, 5:8] = la
    costs[0, 8] = 0
    for inter_num in range(epoch):
        for num_ in range(1, cell_num):
            old_pred = target_function[num_]
            new_pred = vf2.choose_target_no_split(relation, num_, old_pred)
            if np.random.uniform() > .5:
                temp_cost = total_cost(la, target_function, neighbors0_nonan, neighbors1_nonan, lprob_table,
                                       centr0, centr1, avg_neigh_num, num_, new_pred)
                delta_cost = temp_cost[0] - old_cost[0]
                if delta_cost < 0:
                    target_function[num_] = new_pred
                    old_cost = temp_cost
                else:
                    u = np.random.uniform()
                    if u < np.exp(-delta_cost / T):
                        target_function[num_] = new_pred
                        old_cost = temp_cost
            else:
                key_list = []
                val_list = []
                try:
                    key_list.append(num_)
                    val_list.append(new_pred)
                    key_list.append(list(target_function.keys())[list(target_function.values()).index([old_pred])])
                    val_list.append(old_pred)
                    temp_cost = total_cost(la, target_function, neighbors0_nonan, neighbors1_nonan, lprob_table,
                                           centr0, centr1, avg_neigh_num, key_list, val_list)
                    delta_cost = temp_cost[0] - old_cost[0]
                    if delta_cost < 0:
                        target_function[num_] = new_pred
                        target_function[key_list[1]] = val_list[1]
                        old_cost = temp_cost
                    else:
                        u = np.random.uniform()
                        if u < np.exp(-delta_cost / T):
                            target_function[num_] = new_pred
                            old_cost = temp_cost
                except:
                    # temp_cost = total_cost(la, target_function, neighbors0_nonan, neighbors1_nonan, lprob_table,
                    #                        centr0, centr1, avg_neigh_num, num_, new_pred)
                    pass

        costs[inter_num + 1, 0:5] = old_cost
        # if incresing and lam_change < len(milestones):
        #     if (inter_num - last_change > 10) and (np.abs(costs[inter_num + 1, 0] - costs[inter_num - 9, 0]) < epsilon
        #                                            or inter_num > milestones[lam_change]):
        #         lam_change += 1
        #         la += step_la
        #         last_change = inter_num.copy()
        #
        # elif incresing and lam_change >= len(milestones):
        #     if (inter_num - last_change > 10) and np.abs(costs[inter_num + 1, 0] - costs[inter_num - 9, 0]) < epsilon:
        #         costs[inter_num + 1, 5:8] = la
        #         costs[inter_num + 1, 8] = (costs[inter_num, 0] - costs[inter_num + 1, 0]) / T
        #         return target_function, costs
        costs[inter_num + 1, 5:8] = la
        costs[inter_num + 1, 8] = (costs[inter_num, 0] - costs[inter_num + 1, 0])/T
        costs[inter_num + 1, 9] = costs[inter_num + 1, 0] - costs[inter_num, 0]
        costs[inter_num + 1, 10] = T

        if draw_ and (inter_num + 1) % 100 == 0:
            for pnum in range(6):
                plt.plot(costs[0:inter_num + 1, pnum])
                plt.show()
            print(acc_fun(true_predict, target_function))

        if inter_num > 50 and np.sum(np.abs(np.diff(costs[inter_num - 30:inter_num + 1, 0]))) < epsilon:
            return acc_fun(true_predict, target_function), target_function, costs, la

        T = T * dT

    return acc_fun(true_predict, target_function), target_function, costs, la


def pred_lab(l1, tar):
    """
    this function label segments in the I1 with the label in the I0
    :param l1: labeled image for I1
    :param tar: target mapping
    :return: labeled image with labels in I0 segments label
    """
    plab = np.zeros(np.shape(l1)).astype(np.uint8)
    for k in tar.keys():
        plab[l1 == tar[k][0]] = k

    return plab


def dic_sorting(param):
    """
    This function remove all the vectors that they are totally greater at least
    one of the elements of vectors of the list of vectors (param)
    :param param: list of vectors
    :return: same list after removing greater vectors
    """
    par_list = []
    for i in range(len(param[0])):
        par_list.append(np.array((param[0][i], param[1][i], param[2][i])))
    flag = True
    k = 0
    while flag:
        secflag = True
        com_elem = par_list[k]
        l = 0
        while secflag:
            if par_list[l][0] > com_elem[0] and par_list[l][1] > com_elem[1] and par_list[l][2] > com_elem[2]:
                del par_list[l]
                if l >= len(par_list):
                    secflag = False
            else:
                l += 1
                if l >= len(par_list):
                    secflag = False
        k += 1
        if k >= len(par_list):
            flag = False

    return par_list


def combm_split(l1, l2, l3, img_seg0, img_seg1, lab_img0, lab_img1,
                true_pred, trans0, trans1, i0thresh,
                rel_window=45, g_rate=1.3,
                epoch=2000, T=1000, dT=.99, epsilon=.005,
                nei_win=100, draw_=False):
    la = (l1, l2, l3)
    N0 = np.max(lab_img0)
    N1 = np.max(lab_img1)

    centr0, vector0 = vf2.cen_vec(img_seg0, save_name="vector0", lab_img=lab_img0)
    centr1, vector1 = vf2.cen_vec(img_seg1, save_name="vector1", lab_img=lab_img1)

    relation = {}
    for i in range(1, N0 + 1):
        _, cen_list = vf2.close_cen(centr1,
                                    centr0[i - 1],
                                    win_size=rel_window)
        relation[i] = np.int32(cen_list) + 1

    tri0 = Delaunay(centr0)
    tri1 = Delaunay(centr1)

    neighbors0_nonan = {}
    neighbors1_nonan = {}

    for i in range(1, N0 + 1):
        _, neighbors0_nonan[i] = vf2.neigh_without_cell_cut(i - 1, centr0,
                                                            tri0, lab_img0,
                                                            bound_dis=nei_win)
    for i in range(1, N1 + 1):
        _, neighbors1_nonan[i] = vf2.neigh_without_cell_cut(i - 1, centr1,
                                                            tri1, lab_img1,
                                                            bound_dis=nei_win)

    avg_neigh_num = np.mean([len(x) for x in neighbors0_nonan.values()])

    measure_dictionary = measure_finder(centr0, centr1, vector0, vector1, g_rate, relation)

    m_set = set_for_probability_maker(measure_dictionary, nall_=True, nall_val=2)

    prob_table = np.zeros((N0, N1))

    for seg_num in range(1, N0 + 1):
        possibility_data = list(map(lambda x: prob_with_neighbors(seg_num, x, centr0, centr1,
                                                                  measure_dictionary,
                                                                  m_set, neighbors0_nonan,
                                                                  neighbors1_nonan, relation,
                                                                  not_zero=True, big_num=10**9),
                                    relation[seg_num]))

        for num in range(len(possibility_data)):
            prob_table[seg_num - 1, possibility_data[num][1][0][1] - 1] = possibility_data[num][0]

    lprob_table = np.log(prob_table)
    # np.save('logprob_table.npy', lprob_table)

    # lprob_table = np.load('lprob_table.npy')
    # prob_table = lprob_table.copy()

    target_function = {}
    for num in relation:
        target_function[num] = [np.where(prob_table[num - 1, :] == np.max(prob_table[num - 1, :]))[0][0] + 1]

    cell_num = len(target_function) + 1
    costs = np.zeros([epoch + 1, 11])
    old_cost = total_cost(la, target_function, neighbors0_nonan, neighbors1_nonan, lprob_table, centr0,
                          centr1, avg_neigh_num)

    costs[0, 0:5] = old_cost
    costs[0, 5:8] = la
    costs[0, 8] = 0
    delta_e = np.zeros([epoch * N0, 2])
    # print('bm start')
    for inter_num in range(epoch):
        for num_ in range(1, cell_num):
            # print(inter_num, num_)
            old_pred = target_function[num_]
            new_pred = vf2.choose_target_no_split(relation, num_, old_pred)
            if not new_pred:
                continue
            temp_cost = total_cost(la, target_function, neighbors0_nonan, neighbors1_nonan, lprob_table,
                                   centr0, centr1, avg_neigh_num, num_, new_pred)
            delta_cost = temp_cost[0] - old_cost[0]
            if delta_cost < 0:
                target_function[num_] = new_pred
                old_cost = temp_cost
            else:
                delta_e[inter_num * N0 + num_ - 1, 0] = delta_cost
                u = np.random.uniform()
                if u < np.exp(-delta_cost / T):
                    delta_e[inter_num * N0 + num_ - 1, 1] = np.exp(-delta_cost / T)
                    target_function[num_] = new_pred
                    old_cost = temp_cost
        costs[inter_num + 1, 0:5] = old_cost
        costs[inter_num + 1, 5:8] = la
        costs[inter_num + 1, 8] = (costs[inter_num, 0] - costs[inter_num + 1, 0])/T
        costs[inter_num + 1, 9] = costs[inter_num + 1, 0] - costs[inter_num, 0]
        costs[inter_num + 1, 10] = T

        if draw_ and (inter_num + 1) % 100 == 0:
            for pnum in range(6):
                plt.plot(costs[0:inter_num + 1, pnum])
                plt.show()
            if true_pred:
                print(synthetic_acc_fun(true_pred, target_function, trans0, trans1, i0thresh))

        if inter_num > 100 and np.sum(np.abs(np.diff(costs[inter_num - 30:inter_num + 1, 0]))) <\
                epsilon * costs[inter_num - 100, 0]:
            return synthetic_acc_fun(true_pred, target_function, trans0, trans1, i0thresh),\
                   target_function, costs, la, delta_e[:((inter_num + 1) * N0), :]

        T = T * dT

    return synthetic_acc_fun(true_pred, target_function, trans0, trans1, i0thresh), target_function, costs, la, delta_e


def synthetic_acc_fun(true_dic, pred_dic, transfer0, transfer1, i0_thresh):
    """
    here we compare true known dictionary as true_dic and predicted dictionary
    :param transfer0: transfer dictionary from lab0 to ilab0/ transfer_dic output of labtolab function
    :param transfer1: transfer dictionary from lab1 to ilab1/ transfer_dic output of labtolab function
    :param i0_thresh: max_id of frame I_0
    :param true_dic: known dictionary
    :param pred_dic: predicted dictionary
    :return: accuracy/float, list of wrong predicted/list-
    """
    truenum = 0
    wrong_list = []
    n0 = len(pred_dic)
    for nu in range(1, n0 + 1):
        if transfer0[nu] == true_dic[transfer1[pred_dic[nu][0]]]:
            truenum += 1
        elif transfer0[nu] > i0_thresh and transfer0[nu] - i0_thresh == true_dic[transfer1[pred_dic[nu][0]]]:
            truenum += 1
        else:
            wrong_list.append(nu)

    return truenum / n0, wrong_list


if __name__ == "__main__":

    relation_window_size = 40
    growth_rate = 1.3
    img_num = 0
    lab0 = np.load('Images/Samples/{}/lab0.npy'.format(img_num)).astype(np.uint8)
    lab1 = np.load('Images/Samples/{}/lab1.npy'.format(img_num)).astype(np.uint8)

    lab0[lab0 == 2] = 0
    lab0[lab0 == 33] = 0
    lab1[lab1 == 3] = 0
    lab1[lab1 == 5] = 0
    lab1[lab1 == 35] = 0
    lab1[lab1 == 42] = 0

    img0 = np.uint8(lab0.copy())
    img0[img0 != 0] = 1
    img1 = np.uint8(lab1.copy())
    img1[img1 != 0] = 1

    lab0 = label(img0, connectivity=1)
    lab1 = label(img1, connectivity=1)
    img0 = np.uint8(lab0.copy())
    img0[img0 != 0] = 1
    img1 = np.uint8(lab1.copy())
    img1[img1 != 0] = 1

    # img_num = 3
    # lab0 = np.load('Images/Samples/{}/lab0.npy'.format(img_num)).astype(np.uint8)
    # lab1 = np.load('Images/Samples/{}/lab1.npy'.format(img_num)).astype(np.uint8)
    #
    # lab0[lab0 == 1] = 0
    # lab0[lab0 == 7] = 0
    # lab1[lab1 == 2] = 0
    # lab1[lab1 == 5] = 0
    # lab1[lab1 == 8] = 0
    # lab1[lab1 == 9] = 0
    # lab0[lab0 == 11] = 0
    # lab0[lab0 == 15] = 0
    # lab1[lab1 == 13] = 0
    # lab1[lab1 == 14] = 0
    # lab1[lab1 == 16] = 0
    # lab1[lab1 == 25] = 0
    # lab0[lab0 == 23] = 0
    # lab0[lab0 == 24] = 0
    # lab1[lab1 == 28] = 0
    # lab1[lab1 == 30] = 0
    # lab1[lab1 == 38] = 0
    # lab1[lab1 == 40] = 0
    # lab0[lab0 == 28] = 0
    # lab0[lab0 == 29] = 0
    # lab1[lab1 == 27] = 0
    # lab1[lab1 == 43] = 0
    # lab1[lab1 == 36] = 0
    # lab1[lab1 == 50] = 0
    # lab0[lab0 == 51] = 0
    # lab0[lab0 == 56] = 0
    # lab1[lab1 == 62] = 0
    # lab1[lab1 == 64] = 0
    # lab1[lab1 == 65] = 0
    # lab1[lab1 == 67] = 0
    # lab0[lab0 == 69] = 0
    # lab1[lab1 == 79] = 0
    # lab1[lab1 == 94] = 0
    # lab1[lab1 == 101] = 0
    #
    # img0 = np.uint8(lab0.copy())
    # img0[img0 != 0] = 1
    # img1 = np.uint8(lab1.copy())
    # img1[img1 != 0] = 1
    #
    # lab0 = label(img0, connectivity=1)
    # lab1 = label(img1, connectivity=1)
    # img0 = np.uint8(lab0.copy())
    # img0[img0 != 0] = 1
    # img1 = np.uint8(lab1.copy())
    # img1[img1 != 0] = 1

    N0 = np.max(lab0)
    N1 = np.max(lab1)

    centroid0, vec0 = vf2.cen_vec(img0, save_name="vec0", lab_img=lab0)
    centroid1, vec1 = vf2.cen_vec(img1, save_name="vec1", lab_img=lab1)

    relation = {}
    for i in range(1, N0 + 1):
        _, cen_list = vf2.close_cen(centroid1,
                                    centroid0[i - 1],
                                    win_size=relation_window_size)
        relation[i] = np.int32(cen_list) + 1

    # lvec0 = np.linalg.norm(vec0, axis=1)
    # lvec1 = np.linalg.norm(vec1, axis=1)
    #
    tri0 = Delaunay(centroid0)
    tri1 = Delaunay(centroid1)

    neighbors0 = {}
    neighbors1 = {}

    neighbors0_nonan = {}
    neighbors1_nonan = {}

    # for i in range(1, N0 + 1):
    #     neighbors0[i], neighbors0_nonan[i] = vf2.neigh_dis(centroid0[i - 1],
    #                                                        i - 1, centroid0,
    #                                                        tri0, bound_dis=60)
    # for i in range(1, N1 + 1):
    #     neighbors1[i], neighbors1_nonan[i] = vf2.neigh_dis(centroid1[i - 1],
    #                                                        i - 1, centroid1,
    #                                                        tri1, bound_dis=60)
    for i in range(1, N0 + 1):
        neighbors0[i], neighbors0_nonan[i] = vf2.neigh_without_cell_cut(i - 1, centroid0,
                                                                        tri0, lab0,
                                                                        bound_dis=50)
    for i in range(1, N1 + 1):
        neighbors1[i], neighbors1_nonan[i] = vf2.neigh_without_cell_cut(i - 1, centroid1,
                                                                        tri1, lab1,
                                                                        bound_dis=50)
    # for i in range(1, N0 + 1):
    #     neighbors0[i], neighbors0_nonan[i] = vf2.neigh_proportional_cell_cut(i - 1, centroid0,
    #                                                                          tri0, lab0, ep=.1,
    #                                                                          bound_dis=50)
    # for i in range(1, N1 + 1):
    #     neighbors1[i], neighbors1_nonan[i] = vf2.neigh_proportional_cell_cut(i - 1, centroid1,
    #                                                                          tri1, lab1, ep=.1,
    #                                                                          bound_dis=50)

    avg_neigh_num = np.mean([len(x) for x in neighbors0_nonan.values()])

    measure_dictionary = measure_finder(centroid0, centroid1, vec0, vec1, growth_rate, relation)

    m_set = set_for_probability_maker(measure_dictionary, nall_=True, nall_val=2)

    prob_table = np.zeros((N0, N1))

    for seg_num in tqdm(range(1, N0 + 1)):
        possibility_data = list(map(lambda x: prob_with_neighbors(seg_num, x, centroid0, centroid1,
                                                                  measure_dictionary,
                                                                  m_set, neighbors0_nonan,
                                                                  neighbors1_nonan, relation,
                                                                  not_zero=True, big_num=10**9),
                                    relation[seg_num]))

        for num in range(len(possibility_data)):
            prob_table[seg_num - 1, possibility_data[num][1][0][1] - 1] = possibility_data[num][0]

    lprob_table = np.log(prob_table)
    # np.save('lprob_table_nn.npy', lprob_table)

    # lprob_table = np.load('lprob_table.npy')
    # prob_table = lprob_table.copy()

    true_predict = {1: [1], 2: [2], 3: [3], 4: [4], 5: [7], 6: [5], 7: [6], 8: [8], 9: [9], 10: [11], 11: [10],
                    12: [12], 13: [13], 14: [14], 15: [16], 16: [15], 17: [20], 18: [18], 19: [17], 20: [19],
                    21: [24], 22: [21], 23: [22], 24: [25], 25: [26], 26: [23], 27: [27], 28: [29], 29: [28],
                    30: [30], 31: [34], 32: [32], 33: [31], 34: [33], 35: [37], 36: [35], 37: [36], 38: [38],
                    39: [41], 40: [43], 41: [39], 42: [40], 43: [44], 44: [42], 45: [48], 46: [46], 47: [45],
                    48: [49], 49: [47], 50: [50], 51: [52], 52: [51], 53: [54], 54: [55], 55: [53], 56: [56],
                    57: [57], 58: [59], 59: [58]}

    res = BM_implement((1000000, 1000000, 1000000), T=1000, dT=.999,
                       epoch=1000, draw_=True)
    # results = []
    # lamlam = (1000000, 1000000, 1000000)
    # with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
    #     print(multiprocessing.cpu_count())
    #     with tqdm(total=10) as pbar:
    #         for i, res in enumerate(
    #                 pool.imap_unordered(BM_implement, itertools.repeat(lamlam, 10))):
    #             pbar.update()
    #             results.append(res)


    # mean_res = np.zeros((4, 2))
    # lll = [(1000000, 1000000, 1000000), (0.1, 1000000, 1000000), (1, 1000000, 1000000),
    #        (1000000, 1000000, 100)]
    # for adad, lamlam in enumerate(lll):
    #     results = []
    #     with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
    #         print(multiprocessing.cpu_count())
    #         with tqdm(total=100) as pbar:
    #             for i, res in enumerate(
    #                     pool.imap_unordered(BM_implement, itertools.repeat(lamlam, 100))):
    #                 pbar.update()
    #                 results.append(res)
    #
    #     acc_list = []
    #     for i in range(len(results)):
    #         acc_list.append(results[i][0][0])
    #
    #     print('lamlam:', lamlam, np.mean(acc_list))
    #     mean_res[adad, 0] = adad
    #     mean_res[adad, 1] = np.mean(acc_list)
    #
    # np.save('/home/sorena/servers/storage/SS/Images/Samples/same_lam/mean_res.npy', mean_res)
#########################################################
######################## multiprocessing ###################
    # ll = [.01, .1, 1, 10, 100, 10000, 1000000]
    # results = []
    # with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
    #
    #     # for _ in tqdm(pool.istarmap(foo, iterable),
    #     #                    total=len(iterable)):
    #     #     pass
    #
    #     print(multiprocessing.cpu_count())
    #     # results = pool.imap_unordered(BM_implement, itertools.product(ll, ll, ll))
    #
    #     with tqdm(total=7**3) as pbar:
    #         for i, res in enumerate(pool.imap_unordered(BM_implement, itertools.product(ll, ll, ll))):
    #             pbar.update()
    #             results.append(res)
    #
    # np.save('/home/sorena/servers/storage/SS/Images/Samples/same_lam/result_T1000.npy', results)
############################ end of this part of multiprocessing #########3

# for i in tqdm(range(1)):
#     acc, tar, cost = whole_for_multiprocess(1, 1, 1, 1, avg_neigh_num, epoch=600, T=100, dT=99/100)
#     print('number:', i)
#     print('accuracy:', acc[0])
#     print('wrings:', acc[1])
#     for k in wrong:
#         print('predict of {} is {}'.format(k, tar[k][0]))
    # np.save('/home/sorena/servers/storage/SS/Images/Samples/same_lam/cost{}.npy'.format(i), cost)
    #
    # i0, i1 = color_all_cells(tar, lab0, lab1)
    # plt.imshow(i0)
    # plt.savefig('/home/sorena/servers/storage/SS/Images/Samples/same_lam/i0test{}.png'.format(i))
    # plt.show()
    # plt.imshow(i1)
    # plt.savefig('/home/sorena/servers/storage/SS/Images/Samples/same_lam/i1test{}.png'.format(i))
    # plt.show()


###################################################################
# Multiprocess
# img_num = 0
# relation_window_size = 45
# growth_rate = 1.3
# lab0 = np.load('Images/Samples/{}/lab0.npy'.format(img_num)).astype(np.uint8)
# lab1 = np.load('Images/Samples/{}/lab1.npy'.format(img_num)).astype(np.uint8)
#
# lab0[lab0 == 2] = 0
# lab0[lab0 == 33] = 0
# lab1[lab1 == 3] = 0
# lab1[lab1 == 5] = 0
# lab1[lab1 == 35] = 0
# lab1[lab1 == 42] = 0
#
# img0 = np.uint8(lab0.copy())
# img0[img0 != 0] = 1
# img1 = np.uint8(lab1.copy())
# img1[img1 != 0] = 1
#
# lab0 = label(img0, connectivity=1)
# lab1 = label(img1, connectivity=1)
# img0 = np.uint8(lab0.copy())
# img0[img0 != 0] = 1
# img1 = np.uint8(lab1.copy())
# img1[img1 != 0] = 1
#
# N0 = np.max(lab0)
# N1 = np.max(lab1)
#
# centroid0, vec0 = vf2.cen_vec(img0, save_name="vec0", lab_img=lab0)
# centroid1, vec1 = vf2.cen_vec(img1, save_name="vec1", lab_img=lab1)
#
# relation = {}
# for i in range(1, N0 + 1):
#     _, cen_list = vf2.close_cen(centroid1,
#                                 centroid0[i - 1],
#                                 win_size=relation_window_size)
#     relation[i] = np.int32(cen_list) + 1
#
# # lvec0 = np.linalg.norm(vec0, axis=1)
# # lvec1 = np.linalg.norm(vec1, axis=1)
# #
# tri0 = Delaunay(centroid0)
# tri1 = Delaunay(centroid1)
#
# neighbors0 = {}
# neighbors1 = {}
#
# neighbors0_nonan = {}
# neighbors1_nonan = {}
#
# for i in range(1, N0 + 1):
#     neighbors0[i], neighbors0_nonan[i] = vf2.neigh_dis(centroid0[i - 1],
#                                                        i - 1, centroid0,
#                                                        tri0, bound_dis=60)
# for i in range(1, N1 + 1):
#     neighbors1[i], neighbors1_nonan[i] = vf2.neigh_dis(centroid1[i - 1],
#                                                        i - 1, centroid1,
#                                                        tri1, bound_dis=60)
#
# # avg_neigh_num = np.mean([len(x) for x in neighbors0.values()])
#
# measure_dictionary = measure_finder(centroid0, centroid1, vec0, vec1, growth_rate, relation)
#
# m_set = set_for_probability_maker(measure_dictionary, nall_=True, nall_val=2)
#
# prob_table = np.zeros((N0, N1))
#
# for seg_num in tqdm(range(1, N0 + 1)):
#     possibility_data = list(map(lambda x: prob_with_neighbors(seg_num, x, centroid0, centroid1,
#                                                               measure_dictionary,
#                                                               m_set, neighbors0_nonan,
#                                                               neighbors1_nonan, relation,
#                                                               not_zero=True, big_num=10**9),
#                                 relation[seg_num]))
#
#     for num in range(len(possibility_data)):
#         prob_table[seg_num - 1, possibility_data[num][1][0][1] - 1] = possibility_data[num][0]
#
# lprob_table = np.log(prob_table)
#
# # la = [100000, 10000, 10000]
# ll = [10, 100, 1000, 10000, 100000, 1000000]
# with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
#     results = pool.starmap(whole_for_multiprocess, itertools.product(ll, repeat=3))
########################################################################################################


# plt.hist(m3_set, bins=40)
# plt.savefig('/media/sorena/VBox/Doc/Research/WeeklyReport/2020-01-24/Images/mu3_all.png')
# plt.show()
# plt.hist(m2_set, bins=40)
# plt.savefig('/media/sorena/VBox/Doc/Research/WeeklyReport/2020-01-24/Images/mu2_all.png')
# plt.show()
# plt.hist(m1_set, bins=40)
# plt.savefig('/media/sorena/VBox/Doc/Research/WeeklyReport/2020-01-24/Images/mu1_all.png')
# plt.show()

# list(map(lambda x: [x, probability_finder(measure_dictionary[21][x], [m1_set, m2_set, m3_set])[1]],
# measure_dictionary[21].keys()))

# D = list(map(lambda x: prob_with_neighbors(26, x, measure_dictionary,
#                         m_set, neighbors0_nonan, neighbors1_nonan, relation), relation[26]))


#
# for seg_num in range(1, N0 + 1):
#     D = list(map(lambda x: prob_with_neighbors(seg_num, x, measure_dictionary, m_set, neighbors0_nonan,
#                                                neighbors1_nonan, relation,
#                                                not_zero=True, big_num=10**9), relation[seg_num]))
#
#     min_D = D[int(np.where(np.array(D)[:, 0] == np.max(np.array(D)[:, 0]))[0])]
#
#     im0, im1 = cell_painting(min_D[1], lab0, lab1)
#
#     plt.imshow(im0)
#     plt.savefig('/media/sorena/VBox/Doc/Research/WeeklyReport/2020-01-31/Images/min_test/{}_0.png'.format(seg_num))
#     plt.show()
#     plt.imshow(im1)
#     plt.savefig('/media/sorena/VBox/Doc/Research/WeeklyReport/2020-01-31/Images/min_test/{}_1.png'.format(seg_num))
#     plt.show()

    # min_possibility_data = possibility_data[int(np.where(np.array(possibility_data)[:, 0] ==
    #                                                      np.max(np.array(possibility_data)[:, 0]))[0])]
    #
    # im0, im1 = cell_painting(min_D[1], lab0, lab1)
    #
    # plt.imshow(im0)
    # plt.savefig('/media/sorena/VBox/Doc/Research/WeeklyReport/2020-01-31/Images/min_test/{}_0.png'.format(seg_num))
    # plt.show()
    # plt.imshow(im1)
    # plt.savefig('/media/sorena/VBox/Doc/Research/WeeklyReport/2020-01-31/Images/min_test/{}_1.png'.format(seg_num))
    # plt.show()
    #
#

#
# #############################################
# # comparing cost function for different values for lambdas
#
# if __name__ == '__main__':
#     ll = [10, 100, 1000, 10000, 100000, 1000000]
#     with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
#         results = pool.starmap(whole_for_multiprocess, itertools.product(ll, repeat=2))

# st = sep_param_from_result(results)
# stop_t = st[5]
# address_ = '/media/sorena/VBox/Doc/Research/WeeklyReport/2020-03-24/Images/lam_mil3/'
# for i in range(10):
#     for j in range(5):
#         plt.plot(results[i][2][:stop_t[i], j])
#         plt.xticks(np.arange(0, np.max(stop_t) + 30, 20))
#         plt.savefig(address_ + '{}/w{}.png'.format(i, j))
#         plt.show()
#         if j == 0:
#             plt.plot(results[i][2][:stop_t[i], j])
#             plt.xticks(np.arange(0, np.max(stop_t) + 30, 20))
#             plt.savefig(address_ + '{}/{}.png'.format(i, j))
#             plt.show()
#         if j == 1:
#             plt.plot(results[i][2][:stop_t[i], j] * N0)
#             plt.xticks(np.arange(0, np.max(stop_t) + 30, 20))
#             plt.savefig(address_ + '{}/{}.png'.format(i, j))
#             plt.show()
#         if j == 2:
#             plt.plot(results[i][2][:stop_t[i], j] * N0 * (N0 - 1) / 2)
#             plt.xticks(np.arange(0, np.max(stop_t) + 30, 20))
#             plt.savefig(address_ + '{}/{}.png'.format(i, j))
#             plt.show()
#         if j == 3:
#             plt.plot(results[i][2][:stop_t[i], j] * N0 * 2)
#             plt.xticks(np.arange(0, np.max(stop_t) + 30, 20))
#             plt.savefig(address_ + '{}/{}.png'.format(i, j))
#             plt.show()
#         if j == 4:
#             plt.plot(results[i][2][:stop_t[i], j] * N0 * 4 * avg_neigh_num)
#             plt.xticks(np.arange(0, np.max(stop_t) + 30, 20))
#             plt.savefig(address_ + '{}/{}.png'.format(i, j))
#             plt.show()
#     for k in range(5):
#         a = np.min(results[i][2][:stop_t[i], k])
#         b = np.max(results[i][2][:stop_t[i], k])
#         plt.plot((results[i][2][:stop_t[i], k] - a) / (b - a), label='{}'.format(k))
#     plt.legend(loc='best')
#     plt.savefig(address_ + '{}/combined.png'.format(i))
#     plt.show()




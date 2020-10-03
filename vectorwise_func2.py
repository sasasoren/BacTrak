import cv2
import numpy as np
from conv_func import conv_fun, label_img,\
    col_cell, draw_flow
import scipy.io as sio
from statistics import mean
from skimage.measure import label, regionprops
from sklearn.metrics import accuracy_score
import random
import matplotlib.pyplot as plt
import sys
import compare_optical_flow as comp
from scipy.interpolate import griddata
from latex import insert_img, insert_img2, insert_img3,\
    img_table, insert_img3_onecap, img8_input
import conv_func as cf
from scipy.spatial import distance, Delaunay
from skimage.morphology import skeletonize
import time
from tqdm import tqdm
import itertools
import pandas as pd
import vectorwise_func as vf
import moving_simulation as ms
from skimage.draw import line


save_add = "Images/mksample/"

par_var = [.1, .2, .5, 1, 2, 5, 10]
relation_window_size = 45
# par_var = 2.0**np.arange(-1, 11)
# par_var = [.5, 1.0, 8.0, 64, 512, 1024]


def cen_vec(img2d, save_name, save_ad=save_add, lab_img=None):
    # we have binary image 2 dimension as input and centroid and
    # vectors correspondent with all segments as output.
    if lab_img is None:
        lab0 = label(img2d, connectivity=1)
    else:
        lab0 = lab_img

    centroid0 = np.zeros((np.max(lab0), 2))
    vec0 = np.zeros((np.max(lab0), 2))
    img_test0 = comp.for_show(img2d)
    for i in range(1, np.max(lab0) + 1):
        cell_img = np.zeros(np.shape(img2d))
        cell_img[lab0 == i] = 1
        skeleton0 = skeletonize(cell_img)
        y, x = np.where(skeleton0 == 1)
        centroid0[i - 1, 0] = np.int(np.mean(x))
        centroid0[i - 1, 1] = np.int(np.mean(y))
        A = list(zip(x, y))
        vec_arg = np.argmax(distance.cdist(centroid0[i - 1, :].
                                           reshape(1, 2),
                                           A, 'euclidean')[0])
        vec0[i - 1, 0] = A[vec_arg][0] - centroid0[i - 1, 0]
        vec0[i - 1, 1] = A[vec_arg][1] - centroid0[i - 1, 1]

        # cv2.circle(img_test0, tuple(np.int32(centroid0[i - 1])),
        #            2, (255, 0, 0))
        # cv2.arrowedLine(img_test0,
        #                 tuple(np.int32(centroid0[i - 1]))
        #                 , A[vec_arg], (0, 0, 255), 1)

    # plt.imshow(img_test0)
    # plt.savefig(save_ad + save_name + ".png")
    # plt.show()

    return centroid0, vec0


def close_cen(centroid, point, win_size=15):
    cents = centroid.tolist()
    cents = list(filter(lambda w: np.all(
        w[0] <= point[0] + win_size), cents))
    cents = list(filter(lambda w: np.all(
        w[1] <= point[1] + win_size), cents))
    cents = list(filter(lambda w: np.all(
        w[0] >= point[0] - win_size), cents))
    cents = list(filter(lambda w: np.all(
        w[1] >= point[1] - win_size), cents))
    cent_list = []
    for cen in cents:
        cent_list.append(np.where((centroid == cen).all(axis=1))[0][0])

    return cents, cent_list


def choose_target(rel_dic, seg_num, taken_number):
    # This function gets relation dictionary and segment number
    # and current value of target and will return
    # new segment number as potential substitute for the last one
    poss_list = rel_dic[seg_num]
    poss_list = list(poss_list)
    poss_list_nonan = poss_list.copy()
    poss_list.append(np.nan)
    for el in taken_number:
        poss_list.remove(el)

    if not poss_list:
        new_target = np.NaN
        print("{} is seg_num and {} is the taken_number".format(seg_num, taken_number))
    else:
        new_target = [random.choice(poss_list)]
        if new_target[0] != new_target[0]:
            if len(poss_list_nonan) > 1:
                new_target = random.sample(poss_list_nonan, k=2)
            else:
                new_target = poss_list_nonan

    return new_target


def choose_target_no_split(rel_dic, seg_num, taken_number):
    # This function gets relation dictionary and segment number
    # and current value of target. The function will return
    # new segment number as potential substitute for the last one
    poss_list = rel_dic[seg_num]
    poss_list = list(poss_list)

    for el in taken_number:
        poss_list.remove(el)

    if not poss_list:
        new_target = []
        # print("{} is seg_num and {} is the taken_number in choose_target_no_split function"
        #       .format(seg_num, taken_number))
    else:
        new_target = [random.choice(poss_list)]

    return new_target


def make_random_target_no_split(rel):
    t_function = {}
    for num in rel:
        t_function[num] = choose_target_no_split(rel, num, [])

    return t_function


def choose_nan(rel_dic, seg_num):
    poss_list = rel_dic[seg_num]
    poss_list = list(poss_list)

    new_target = random.sample(poss_list, k=2)

    return new_target


def w_centroid(cent0, cent1):
    # This function will return norm squared of cent0-cent1
    w_i = np.linalg.norm((cent0 - cent1))
    w_sum = np.sum(w_i)
    return w_sum


def same_seg_delta(seg_num, t_seg, search_set):
    # t_seg should be just value of related dictionary
    # This function gets target function and count how many
    # same target segment exist in search set
    counter_ = 0
    for j in range(len(search_set)):
        if seg_num - 1 != j and t_seg == search_set[j]:
            counter_ += 1

    return counter_ / 2


def find_neighbors(pindex, triang):
    # This function will find neighbors of point pindex from
    # triang and return index of neighbors
    return triang.vertex_neighbor_vertices[1][
           triang.vertex_neighbor_vertices[0][pindex]:
           triang.vertex_neighbor_vertices[0][pindex+1]]


def neigh_dis(point, pind, pset, triangle, bound_dis=50):
    # This function gets pindex (index of desired point) and
    # coordinate of that and pset is the set of targeted set.
    # Output of the function is neighbors truncated from long
    # distance neighbors
    neig = find_neighbors(pind, triangle)
    neigh_set = neig.tolist()
    for neighbor in neig:
        if np.linalg.norm((point-pset[neighbor])) > bound_dis:
            neigh_set.remove(neighbor)

    neigh_with_nan = [x+1 for x in neigh_set]
    neigh_without_nan = neigh_with_nan.copy()
    neigh_with_nan.append(np.nan)
    return neigh_with_nan, neigh_without_nan


def neigh_without_cell_cut(pind, pset, triangle, lab, bound_dis=50):
    """
    we find neighbors of cell number pind with Delaunay triangulation and if there is any overlap between the line
    from two neighbors and another cell we remove that neighbor as a candidate for neighbors.
    :param pind: index of cell integer
    :param pset: set of all centroid
    :param triangle: Delaunay triangulation between the cells
    :param lab: labeled image for cell numbers
    :param bound_dis: threshold for removing the cells from far distance
    :return: neighbors with nan and without nan
    """
    neig = find_neighbors(pind, triangle)
    neigh_set = neig.tolist()
    for neighbor in neig:
        if np.linalg.norm((pset[pind]-pset[neighbor])) > bound_dis:
            neigh_set.remove(neighbor)

        else:
            line_pixel_set = list(lab[tuple(zip(line(int(pset[pind][1]), int(pset[pind][0]),
                                                     int(pset[neighbor][1]), int(pset[neighbor][0]))))][0])

            if {0, pind + 1, neighbor + 1} != (set(np.unique(line_pixel_set)) | {0}):
                neigh_set.remove(neighbor)

    neigh_with_nan = [x+1 for x in neigh_set]
    neigh_without_nan = neigh_with_nan.copy()
    neigh_with_nan.append(np.nan)
    return neigh_with_nan, neigh_without_nan


def ratio_cut(base, des, mid, lab, cent):
    """
    This function will look at the three cells and find the ratio of the middle cell
    that have been cut by the line fron centroid of base to centroid of destination
    :param base:
    :param des:
    :param mid:
    :param lab:
    :param cent:
    :return:
    """
    vb = cent[base] - cent[des]
    s = list(map(lambda x: np.cross(vb, (cent[base][0] - x[1], cent[base][1] - x[0])),
                 list(zip(*np.where(lab == mid)))))
    if len(np.where(np.array(s) > 0)[0]) > len(np.where(np.array(s) < 0)[0]):
        return len(np.where(np.array(s) < 0)[0]) / len(np.where(lab == mid)[0])
    else:
        return len(np.where(np.array(s) > 0)[0]) / len(np.where(lab == mid)[0])


def neigh_proportional_cell_cut(pind, pset, triangle, lab, ep=.2, bound_dis=50):
    """
    we find neighbors of cell number pind with Delaunay triangulation and if there is any overlap between the line
    from two neighbors and another cell we remove that neighbor as a candidate for neighbors.
    :param pind: index of cell integer
    :param pset: set of all centroid
    :param triangle: Delaunay triangulation between the cells
    :param lab: labeled image for cell numbers
    :param ep: threshold to accept or reject neighbors based on the proportion
    that cut other cell
    :param bound_dis: threshold for removing the cells from far distance
    :return: neighbors with nan and without nan
    """
    neig = find_neighbors(pind, triangle)
    neigh_set = neig.tolist()
    for neighbor in neig:
        if np.linalg.norm((pset[pind]-pset[neighbor])) > bound_dis:
            neigh_set.remove(neighbor)

        else:
            line_pixel_set = list(lab[tuple(zip(line(int(pset[pind][1]), int(pset[pind][0]),
                                                     int(pset[neighbor][1]), int(pset[neighbor][0]))))][0])

            if len((set(np.unique(line_pixel_set)) | {0}) - {0, pind + 1, neighbor + 1}) == 1:
                mid_cell = list((set(np.unique(line_pixel_set)) | {0}) - {0, pind + 1, neighbor + 1})
                if mid_cell == [0]:
                    pass
                else:
                    if ratio_cut(pind, neighbor, mid_cell[0], lab, pset) > ep:
                        neigh_set.remove(neighbor)
            elif len((set(np.unique(line_pixel_set)) | {0}) - {0, pind + 1, neighbor + 1}) > 1:
                neigh_set.remove(neighbor)

    neigh_with_nan = [x+1 for x in neigh_set]
    neigh_without_nan = neigh_with_nan.copy()
    neigh_with_nan.append(np.nan)
    return neigh_with_nan, neigh_without_nan


def hm_target_neighbor(base_neigh, target_neigh, target_func):
    # This function counts the number of predicted cells which
    # they are not the new neighbors of the predicted target cell
    counter_ = 0
    for neigh in base_neigh:
        if neigh != neigh:
            counter_ = counter_
        elif target_func[neigh][0] not in target_neigh:
            counter_ += 1

    return counter_


def make_img(img_size, hm, spd=6, rtn=80,
             d_rtn=10, lg_=(6, 11), d_lg=3, brd=20, split=False):
    # This function create two images as t=0 and 1 with size
    # img_size with hm steps and change in position with at
    # most speed of spd  and rotation of rtn, d_rt is the difference
    # between rtn0 and 1. brd is the border of image. lg: length growth.
    # b_size is the interval size

    img0 = np.zeros((img_size, img_size), np.uint8)
    img1 = np.zeros((img_size, img_size), np.uint8)

    img_lab0 = np.zeros((img_size, img_size), np.uint8)
    img_lab1 = np.zeros((img_size, img_size), np.uint8)

    num_ = np.int8(((img_size - 2*brd)/hm) + 1)
    xv, yv = np.meshgrid(np.arange(brd, img_size - brd, num_),
                         np.arange(brd, img_size - brd, num_))
    # crn is the number of cell in one layer almost equal to hm
    crn = len(np.arange(brd, img_size - brd, num_))
    rot0 = np.random.choice(np.arange(-rtn, rtn+1), (crn, crn))
    drt = np.random.choice(np.arange(-d_rtn, d_rtn+1), (crn, crn))
    rot1 = rot0 + drt
    minlg, maxlg = lg_
    cell_lg = np.random.choice(np.arange(minlg, maxlg), (crn, crn))
    dif_lg = np.random.choice(np.arange(d_lg), (crn, crn))
    lg_chng = cell_lg + dif_lg

    hor_chng = np.random.choice(np.arange(-spd, spd+1), crn)
    # hor_chng = np.zeros(crn, np.uint8)
    # hor_chng[0] = np.random.choice(np.arange(-spd, spd+1), 1, np.uint8)
    # for i in range(1, crn):
    #     temp = np.random.choice(np.arange(-spd, spd+1), 1, np.uint8)
    #     hor_chng[i] = hor_chng[i-1] + temp

    cent_chng = np.zeros((crn * crn, 2, 2), np.uint8)

    cnum = 1
    for ver in range(crn):
        ver_chng = np.random.choice(np.arange(-spd, spd+1))
        for hor in range(crn):
            cv2.ellipse(img0, (xv[ver, hor], yv[ver, hor]),
                        (cell_lg[ver, hor], 2), rot0[ver, hor], 0, 360, 255, -1)
            cv2.ellipse(img_lab0, (xv[ver, hor], yv[ver, hor]),
                        (cell_lg[ver, hor], 2),rot0[ver, hor], 0, 360, cnum, -1)

            cv2.ellipse(img1, (xv[ver, hor] + ver_chng, yv[ver, hor] +
                               hor_chng[hor]), (lg_chng[ver, hor], 2),
                        rot1[ver, hor], 0, 360, 255, -1)
            cv2.ellipse(img_lab1, (xv[ver, hor] + ver_chng, yv[ver, hor] +
                                   hor_chng[hor]), (lg_chng[ver, hor], 2),
                        rot1[ver, hor], 0, 360, cnum, -1)
            cent_chng[cnum - 1, 0, :] = xv[ver, hor], yv[ver, hor]
            cent_chng[cnum - 1, 1, :] = xv[ver, hor] + ver_chng,\
                                        yv[ver, hor] + hor_chng[hor]

            ver_chng += np.random.choice(np.arange(-spd, spd + 1))

            cnum += 1

        hor_chng += np.random.choice(np.arange(-spd, spd + 1), crn)

    if split is True:
        new_label_list = {}
        max_where = np.where(lg_chng == np.max(lg_chng))
        new_label = np.max(img_lab1)
        for m in range(len(max_where[0])):
            max_lab = max_where[0][m] * crn + max_where[1][m] + 1
            new_label += 1
            # print(new_label)
            x, y = cent_chng[max_lab - 1, 1, :]
            x = x - 4
            y = y - 2
            img1[img_lab1 == max_lab] = 0
            img_lab1[img_lab1 == max_lab] = 0
            cv2.ellipse(img1, (x, y), (lg_[0], 2),
                        rot1[max_where[0][m], max_where[1][m]],
                        0, 360, 255, -1)
            cv2.ellipse(img_lab1, (x, y), (lg_[0], 2),
                        rot1[max_where[0][m], max_where[1][m]],
                        0, 360, int(max_lab), -1)
            cv2.ellipse(img1, (x + 7, y + 9), (lg_[0], 2),
                        rot1[max_where[0][m], max_where[1][m]],
                        0, 360, 255, -1)
            cv2.ellipse(img_lab1, (x + 7, y + 9),
                        (lg_[0], 2), rot1[max_where[0][m], max_where[1][m]],
                        0, 360, int(new_label), -1)

            new_label_list[max_lab] = [max_lab, new_label]
            # new_label_list returns label for both of the splited cells

        return img0, img1, img_lab0, img_lab1, cent_chng, new_label_list

    # cv2.imwrite(save_ad+"img0.png", img0)
    # cv2.imwrite(save_ad+"img1.png", img1)
    # cv2.imwrite(save_ad+"lab_img0.png", img_lab0)
    # cv2.imwrite(save_ad+"lab_img1.png", img_lab1)
    # np.save(save_ad+"cent_chng.npy", cent_chng)

    return img0, img1, img_lab0, img_lab1, cent_chng


def dict_to_list(diction, nrep=True):
    lst = []
    if nrep == True:
        for val in diction.values():
            for n in [val]:
                if n in lst:
                    continue
                else:
                    lst.append(n)
    else:
        for val in diction.values():
            for n in [val]:
                lst.append(n)

    return lst


def v_box(wch_, rel, vector0, vector1, targ, new=None, old=None):
    # v_box will return min of ||v_i - V_j|| for all cell j in moving window of
    # cell i from img0 and over all of j which they are not taken
    # by target function.
    # wch_: number of cell in img0 (aka i). rel: relation dictionary.
    # targ: target function.
    box = []
    lst = dict_to_list(targ)
    new_lst = [x for y in lst for x in y]

    if old != None:
        lst.remove(old)
    elif new != None:
        lst.append(new)

    for nei in rel[wch_]:
        if nei == nei and nei not in new_lst:
            box.append(np.min((w_centroid(vector0[wch_ - 1], vector1[nei - 1]),
                               w_centroid(vector0[wch_ - 1], -vector1[nei - 1]))))

    if box == []:
        return 0
    else:
        return np.min(box)


def diff_cost(n_pred, o_pred, n_, cent0, cent1, v0, v1, neigh0, neigh1,
              avg_neigh, c_num, trgt_function, relat,
              alpha_=2, beta_=1000, gamma_=13, thrsh=1,
              alpha2_=2, alpha3_=.5, spl=.1):
    # In this function we are computing delta cost for our model.
    # n_pred: number of new prediction for the cell,
    # o_pred: number of old prediction for our model,
    # n_: number of cell in image0, relat: relation dictionary.
    # cent0, cent1: centroid for image 0 and 1,
    # v0, v1: correspondent vectors for each cell,
    # neigh0, neigh1: neighbors for all cells in image 0 ad 1,
    # avg_neigh: average number of neighbors,
    # c_num: total number of cell in image 0,
    # alpha_, beta_, gamma_, thrsh: parameters of th model
    # spl: is the rate of splitting cells
    if n_pred != n_pred:
        d_w = -w_centroid(cent0[n_ - 1], cent1[o_pred - 1])

        d_V = -np.min((w_centroid(v0[n_ - 1], v1[o_pred - 1]),
                      w_centroid(v0[n_ - 1], -v1[o_pred - 1])))

        d_same_seg = -same_seg_delta(n_, o_pred, list(trgt_function.values()))

        d_neighbor = len(neigh0[n_]) - 1 - hm_target_neighbor(neigh0[n_],
                                                              neigh1[o_pred],
                                                              trgt_function)

        d_star = np.max((0, v_box(n_, relat, v0, v1,
                                  trgt_function, old=o_pred) - thrsh))

        lst = dict_to_list(trgt_function, nrep=False)
        nan_num = np.count_nonzero(np.isnan(lst))
        d_perc = np.max((0, (nan_num + 1)/len(trgt_function) - spl)) -\
                 np.max((0, nan_num/len(trgt_function) - spl))

    elif o_pred != o_pred:
        d_w = w_centroid(cent0[n_ - 1], cent1[n_pred - 1])

        d_V = np.min((w_centroid(v0[n_ - 1], v1[n_pred - 1]),
                      w_centroid(v0[n_ - 1], -v1[n_pred - 1])))

        d_same_seg = same_seg_delta(n_, n_pred, list(trgt_function.values()))

        d_neighbor = hm_target_neighbor(neigh0[n_], neigh1[n_pred],
                                        trgt_function) - len(neigh0[n_]) - 1

        d_star = -np.max((0, v_box(n_, relat, v0, v1,
                                   trgt_function, new=n_pred) - thrsh))

        lst = dict_to_list(trgt_function, nrep=False)
        nan_num = np.count_nonzero(np.isnan(lst))
        d_perc = np.max((0, (nan_num - 1) / len(trgt_function) - spl)) - \
                 np.max((0, nan_num / len(trgt_function) - spl))

    else:
        # Here we are computing delta w in the formula
        d_w = w_centroid(cent0[n_ - 1], cent1[n_pred - 1]) - \
                 w_centroid(cent0[n_ - 1], cent1[o_pred - 1])
        # Here we are computing delta V in formula because we don't know
        # the direction of vectors we consider min of + and - as output
        d_V = np.min((w_centroid(v0[n_ - 1], v1[n_pred - 1]),
                          w_centroid(v0[n_ - 1], -v1[n_pred - 1]))) - \
                  np.min((w_centroid(v0[n_ - 1], v1[o_pred - 1]),
                          w_centroid(v0[n_ - 1], -v1[o_pred - 1])))

        d_same_seg = same_seg_delta(n_, n_pred, list(trgt_function.values())) - \
                     same_seg_delta(n_, o_pred, list(trgt_function.values()))

        d_neighbor = hm_target_neighbor(neigh0[n_], neigh1[n_pred],
                                        trgt_function) - hm_target_neighbor(
            neigh0[n_], neigh1[o_pred], trgt_function)

        d_star = 0

        d_perc = 0

    d_cost = ((d_w + alpha_ * d_V + beta_ * d_same_seg / (c_num - 2) +
               gamma_ * d_neighbor / avg_neigh -
               alpha2_ * d_star) / (c_num - 1)) + (alpha3_ * d_perc)

    return d_cost


def nan_or_not(dic):
    # This function separates cells with nan predict from not nan predicts
    # nan_dic = {k: v for k, v in dic.items() if ~pd.Series(v).notna().all()}
    # notnan_dic = {k: v for k, v in dic.items() if pd.Series(v).notna().all()}
    nan_dic = {k: v for k, v in dic.items() if len(dic[k]) > 1}
    notnan_dic = {k: v for k, v in dic.items() if len(dic[k]) == 1}
    return nan_dic, notnan_dic


def v_cost(t_fun, v0, v1):
    # This function return cost of the differece of vectors part of the formula
    v_c = np.sum(np.min((np.linalg.norm(
        (v0[np.array(list(t_fun.keys())) - 1] +
         v1[np.array(list(t_fun.values()))[:, 0] - 1]), axis=1),
                         np.linalg.norm((v0[np.array(list(t_fun.keys())) - 1] -
                                         v1[np.array(list(t_fun.values()))[:, 0] - 1]),
                                        axis=1)), axis=0) /
                 np.linalg.norm(v0[np.array(list(t_fun.keys())) - 1], axis=1))
    return v_c


def v_cost_tau(t_fun, v0, v1, tau):
    # This function return cost of the difference of vectors part of the formula
    v_c = np.sum(np.square(np.min((np.linalg.norm(
        (v0[np.array(list(t_fun.keys())) - 1] +
         v1[np.array(list(t_fun.values()))[:, 0] - 1]), axis=1),
                         np.linalg.norm((v0[np.array(list(t_fun.keys())) - 1] -
                                         v1[np.array(list(t_fun.values()))[:, 0] - 1]),
                                        axis=1)), axis=0) /
                         np.linalg.norm(v0[np.array(list(t_fun.keys())) - 1], axis=1) - tau,
                         np.zeros(len(v0))))
    return v_c


def same_seg_cost(t_fun):
    # finding cost of same segment part of the actual formula
    # t_fun is the target function
    s_cost = np.sum(list(map(lambda x: same_seg_delta(x, t_fun[x],
                                                      list(t_fun.values())),
                             list(t_fun.keys()))))
    return s_cost


def neighbor_cost(neigh0, neigh1, t_fun, notnan_t):
    # finding cost of neighbors part of the actual formula
    # neigh0 and 1 are neighbors for image 0 and 1
    # t_fun is target function
    # notnan_t is target function without nan parts
    neigh_cost = np.sum(list(map(lambda x: hm_target_neighbor(
        neigh0[x], neigh1[notnan_t[x][0]], t_fun),
                    list(notnan_t.keys()))))

    return neigh_cost


def v_nan_cost(t_fun, v0, v1, col=0):
    # This function return cost of the differece of vectors part of the formula D_ij = min(||2V_j -+ v_i||)
    if t_fun == {}:
        v_c = 0
    else:
        v_c = np.sum(np.min((np.linalg.norm(
            (v0[np.array(list(t_fun.keys())) - 1] +
             2 * v1[np.array(list(t_fun.values()))[:, col] - 1]), axis=1),
                             np.linalg.norm((v0[np.array(list(t_fun.keys())) - 1] -
                                             2 * v1[np.array(list(t_fun.values()))[:, col] - 1]),
                                            axis=1)), axis=0) /
                     np.linalg.norm(v0[np.array(list(t_fun.keys())) - 1], axis=1))
    return v_c


def same_cell_overlap(base_func, tar_func, same_set=True):
    '''
    This function counts number of same segment chosen by target function. if just two cell
    have the same target function value it will return 1 more than that it will grow w.r.t
    number of
    :param base_func:
    :param tar_func:
    :param same_set:
    :return: how many same segments have same target value
    '''

    counter_ = 0
    for cells in base_func.values():
        for c in cells:
            for cells2 in tar_func.values():
                for c2 in cells2:
                    if c == c2:
                        counter_ += 1

    if same_set:
        counter_ = (counter_ - len(base_func)) / 2

    return counter_


def predict_overlap(base_func, tar_func, same_set=True):
    '''
    This function counts number of same segment chosen by target function. if just two cell
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


def total_cost(cen0, cen1, v0, v1, neigh0, neigh1, ta_fun, rel, avg_neigh,
               alpha_, spl=.1, tsh=1, n_pre=None):
    # finding total cost for each epoch and return separate cost and total cost as
    # output. Inputs are the same as d_cost.

    t_fun = ta_fun.copy()

    if n_pre != None:
        t_fun[n_pre[0]] = n_pre[1]

    N_0 = len(t_fun)
    nan_target, notnan_target = nan_or_not(t_fun)

    # coefficient of alpha0
    w_cost = w_centroid(cen0[np.array(list(notnan_target.keys())) - 1],
                        cen1[np.array(list(notnan_target.values())) - 1]) / N_0
    # coefficient of alpha1
    V_cost = v_cost(notnan_target, v0, v1) / N_0

    # coefficient of alpha2
    same_s_cost = same_seg_cost(notnan_target) / (N_0 * (N_0 - 1))

    # coefficient of alpha3
    neigh_cost = neighbor_cost(neigh0, neigh1, t_fun, notnan_target) / avg_neigh

    # coefficient of alpha4
    star_cost = np.sum(list(map(lambda x: np.max(
        (0, v_box(x, rel, v0, v1, t_fun) - tsh)),
                                     list(nan_target.keys()))))

    # coefficient of alpha5
    split_cost = np.max((0, (len(nan_target) / N_0) - spl))

    # coefficient of alpha6
    V_nan_cost = (v_nan_cost(nan_target, v0, v1, col=0) + v_nan_cost(nan_target, v0, v1, col=1)) / N_0

    # coefficient of alpha7
    if nan_target == {}:
        w_nan_cost = 0
    else:
        w_nan_cost = w_centroid(cen1[np.array(list(nan_target.values()))[:, 0] - 1],
                                cen1[np.array(list(nan_target.values()))[:, 1] - 1])

    # coefficient of alpha8
    same_cost_nans = 2 * same_cell_overlap(nan_target, nan_target, same_set=True) / (N_0 * (N_0 - 1))

    # coefficient of alpha9
    same_cost_nan_and_not = 2 * same_cell_overlap(nan_target, notnan_target, same_set=False) / \
                            (N_0 * (N_0 - 1))

    t_cost = alpha_[0] * w_cost + alpha_[1] * V_cost + alpha_[2] * same_s_cost + alpha_[3] * neigh_cost \
             + alpha_[4] * star_cost + alpha_[5] * split_cost + alpha_[6] * V_nan_cost \
             + alpha_[7] * w_nan_cost +alpha_[8] * same_cost_nans + alpha_[9] * same_cost_nan_and_not

    return t_cost, w_cost, V_cost, same_s_cost, neigh_cost, star_cost, split_cost, V_nan_cost, \
           w_nan_cost, same_cost_nans, same_cost_nan_and_not


def total_cost_no_split(cen0, cen1, v0, v1, neigh0, neigh1, ta_fun, avg_neigh,
                        alpha_, n_pre=None):
    # finding total cost for each epoch and return separate cost and total cost as
    # output. Inputs are the same as d_cost.

    t_fun = ta_fun.copy()

    if n_pre is not None:
        t_fun[n_pre[0]] = n_pre[1]
        if n_pre[1] != n_pre[1]:
            return np.infty, 0, 0, 0, 0

    N_0 = len(t_fun)
    notnan_target = t_fun.copy()

    # coefficient of alpha0
    w_cost = w_centroid(cen0[np.array(list(notnan_target.keys())) - 1],
                        cen1[np.array(list(notnan_target.values())) - 1]) / N_0
    # coefficient of alpha1
    V_cost = v_cost(notnan_target, v0, v1) / N_0

    # coefficient of alpha2
    same_s_cost = same_seg_cost(notnan_target) / (N_0 * (N_0 - 1))

    # coefficient of alpha3
    neigh_cost = neighbor_cost(neigh0, neigh1, t_fun, notnan_target) / avg_neigh

    t_cost = alpha_[0] * w_cost + alpha_[1] * V_cost + alpha_[2] * same_s_cost +\
             alpha_[3] * neigh_cost

    return t_cost, w_cost, V_cost, same_s_cost, neigh_cost


def cell_cost(cen0, cen1, v0, v1, neigh0, neigh1, ta_fun, rel, avg_neigh,
              alpha_, spl=.1, tsh=1, n_pre=None):
    '''
    This function return the cost of each cell
    :param cen0:
    :param cen1:
    :param v0:
    :param v1:
    :param neigh0:
    :param neigh1:
    :param ta_fun:
    :param rel:
    :param avg_neigh:
    :param alpha_:
    :param spl:
    :param tsh:
    :param n_pre:
    :return:
    '''
    t_fun = ta_fun.copy()

    N_0 = len(t_fun)
    cost_lst = np.zeros([N_0, 9])

    nan_target, notnan_target = nan_or_not(t_fun)

    key_list_nan = list(nan_target.keys())
    key_list_notnan = list(notnan_target.keys())

    for key_num in range(1, N_0 + 1):
        if key_num in key_list_nan:
            cost_lst[key_num - 1, 0] = 0
        else:
            cost_lst[key_num - 1, 0] = w_centroid(cen0[key_num - 1],
                                                  cen1[notnan_target[key_num][0] - 1]) / N_0

        if key_num in key_list_nan:
            cost_lst[key_num - 1, 1] = 0
        else:
            cost_lst[key_num - 1, 1] = v_cost({key_num: notnan_target[key_num]}, v0, v1) / N_0

        if key_num in key_list_nan:
            cost_lst[key_num - 1, 2] = 0
        else:
            cost_lst[key_num - 1, 2] = 2 * (same_cell_overlap({key_num: notnan_target[key_num]},
                                                              notnan_target, same_set=False) - 1) / (N_0 * (N_0 - 1))

        if key_num in key_list_nan:
            cost_lst[key_num - 1, 3] = 0
        else:
            cost_lst[key_num - 1, 3] = neighbor_cost(neigh0, neigh1,
                                                     t_fun, {key_num: notnan_target[key_num]})

        # I set it equal to zero for now because we have to change it.
        if key_num in key_list_nan:
            cost_lst[key_num - 1, 4] = 0
        else:
            cost_lst[key_num - 1, 4] = 0

        if key_num in key_list_notnan:
            cost_lst[key_num - 1, 5] = 0
        else:
            cost_lst[key_num - 1, 5] = (v_nan_cost({key_num: nan_target[key_num]}, v0, v1, col=0) +
                                        v_nan_cost({key_num: nan_target[key_num]}, v0, v1, col=1)) / N_0

        if key_num in key_list_notnan:
            cost_lst[key_num - 1, 6] = 0
        else:
            cost_lst[key_num - 1, 6] = w_centroid(cen1[nan_target[key_num][0] - 1],
                                                  cen1[nan_target[key_num][1] - 1])

        if key_num in key_list_notnan:
            cost_lst[key_num - 1, 7] = 0
        else:
            cost_lst[key_num - 1, 7] = 2 * same_cell_overlap({key_num: nan_target[key_num]},
                                                             nan_target, same_set=False) / (N_0 * (N_0 - 1))

        if key_num in key_list_notnan:
            cost_lst[key_num - 1, 8] = 2 * same_cell_overlap({key_num: notnan_target[key_num]},
                                                             nan_target, same_set=False) / (N_0 * (N_0 - 1))
        else:
            cost_lst[key_num - 1, 8] = 2 * same_cell_overlap({key_num: nan_target[key_num]},
                                                             notnan_target, same_set=False) / (N_0 * (N_0 - 1))

    return cost_lst


def small_dic_to_complete(s_dic, cell_nums):
    '''
    This function gets brieve dictionary and spand it to the bigh dictionary for target function
    :param s_dic: small dictionary
    :param cell_nums: number of cell in dictionary
    :return: target function dictionary
    '''
    comp_dic = {}
    for n in range(1, cell_nums + 1):
        if n in list(s_dic.keys()):
            comp_dic[n] = s_dic[n]
        else:
            comp_dic[n] = [n]

    return comp_dic


def total_cost_per_parameter(cen0, cen1, v0, v1, neigh0, neigh1, ta_fun, rel, avg_neigh,
                             spl=.1, tsh=1, n_pre=None):
    '''
    This function will find the value of each parameter for the original cost function
    I will use it for finding the area of convex optimization.

    :return: All the parameter for the cost function.
    '''

    t_fun = ta_fun.copy()

    if n_pre is not None:
        t_fun[n_pre[0]] = n_pre[1]

    N_0 = len(t_fun)
    nan_target, notnan_target = nan_or_not(t_fun)

    w_cost = w_centroid(cen0[np.array(list(notnan_target.keys())) - 1],
                        cen1[np.array(list(notnan_target.values())) - 1]) / N_0
    # coefficient of alpha1
    V_cost = v_cost(notnan_target, v0, v1) / N_0

    # coefficient of alpha2
    same_s_cost = same_seg_cost(notnan_target) / (N_0 * (N_0 - 1))

    # coefficient of alpha3
    neigh_cost = neighbor_cost(neigh0, neigh1, t_fun, notnan_target) / avg_neigh

    # coefficient of alpha4
    star_cost = np.sum(list(map(lambda x: np.max(
        (0, v_box(x, rel, v0, v1, t_fun) - tsh)),
                                     list(nan_target.keys()))))

    # coefficient of alpha5
    split_cost = np.max((0, (len(nan_target) / N_0) - spl))

    # coefficient of alpha6
    V_nan_cost = (v_nan_cost(nan_target, v0, v1, col=0) + v_nan_cost(nan_target, v0, v1, col=1)) / N_0

    # coefficient of alpha7
    w_nan_cost = w_centroid(cen1[np.array(list(nan_target.values()))[:, 0] - 1],
                            cen1[np.array(list(nan_target.values()))[:, 1] - 1])

    # coefficient of alpha8
    same_cost_nans = 2 * same_cell_overlap(nan_target, nan_target, same_set=True) / (N_0 * (N_0 - 1))

    # coefficient of alpha9
    same_cost_nan_and_not = 2 * same_cell_overlap(nan_target, notnan_target, same_set=False) / \
                            (N_0 * (N_0 - 1))

    return w_cost, V_cost, same_s_cost, neigh_cost, star_cost, split_cost, V_nan_cost\
        , w_nan_cost, same_cost_nans, same_cost_nan_and_not


def total_cost_per_parameter_no_split(cen0, cen1, v0, v1, neigh0, neigh1,
                                      ta_fun, avg_neigh, n_pre=None):
    '''
    This function will find the value of each parameter for the original cost function
    I will use it for finding the area of convex optimization.

    :return: All the parameter for the cost function.
    '''

    t_fun = ta_fun.copy()

    if n_pre is not None:
        t_fun[n_pre[0]] = n_pre[1]

    N_0 = len(t_fun)
    nan_target, notnan_target = nan_or_not(t_fun)

    w_cost = w_centroid(cen0[np.array(list(notnan_target.keys())) - 1],
                        cen1[np.array(list(notnan_target.values())) - 1]) / N_0
    # coefficient of alpha1
    V_cost = v_cost(notnan_target, v0, v1) / N_0

    # coefficient of alpha2
    same_s_cost = same_seg_cost(notnan_target) / (N_0 * (N_0 - 1))

    # coefficient of alpha3
    neigh_cost = neighbor_cost(neigh0, neigh1, t_fun, notnan_target) / avg_neigh

    return w_cost, V_cost, same_s_cost, neigh_cost


def weight_loss_one_seg(cent0, cent1, nei0, base, t_fun):
    '''
    This function gets these variable and computes:
     \sum_{j\in \Omega_i}\frac{||C_{f(i)} - C_{f(j)}||}{||c_i - c_j||}
     which is the part of the first element of the cost function.
    :param cent0: centroid for image t=0 (I_0) (matrix of position of segments)
    :param cent1:centroid for image t=1 (I_1) (matrix of position of segments)
    :param nei0: neighbors of segment at t=0 (correspondent dictionary)
    :param base: segment number in image t=0 (integer number)
    :param t_fun: target function (dictionary)
    :return: value of w_loss
    '''

    if len(nei0[base]) == 0:
        return 0
    w_loss = 0
    for h in nei0[base]:
        w_loss += w_centroid(cent1[t_fun[base][0] - 1], cent1[t_fun[h][0] - 1]) / \
                  w_centroid(cent0[base - 1], cent0[h - 1])

    return w_loss / len(nei0[base])


def weight_loss(cent0, cent1, nei0, t_fun):
    '''
    This function compute first element of the cost function which is:
    \sum_{i=1}^{N_0} \sum_{j\in \Omega_i}\frac{||C_{f(i)} - C_{f(j)}||}{||c_i - c_j||}
    which is the parralel computation on weight_loss_one_seg function.
    :param cent0: centroid for image t=0 (I_0) (matrix of position of segments)
    :param cent1:centroid for image t=1 (I_1) (matrix of position of segments)
    :param nei0: neighbors of segment at t=0 (correspondent dictionary)
    :param t_fun: target function (dictionary)
    :return: return value of w_cost
    '''
    w_cost = np.sum(list(map(lambda x: weight_loss_one_seg(cent0, cent1,
                                                           nei0, x, t_fun),
                             list(t_fun.keys()))))

    return w_cost / len(t_fun)


def new_parameter_cost_no_split(cen0, cen1, v0, v1, neigh0, neigh1,
                                      ta_fun, avg_neigh, n_pre=None, tau=.2):
    '''
    This function will find the value of each parameter for the new cost function
    \begin{array}{rl}
    E =& \alpha_0 G(c) \\
    \\
    &+ \alpha_1 \frac{1}{N_0}\sum\limits_{i=1}^{N_0} (\frac{d_{if(i)}}{||v_i||_2} - \tau)^+\\
    \\
    &+ \alpha_2 \sum\limits_{i=1}^{N_0}\textbf{I}_{\{f(i)\in\Gamma_i\}}  \\
    \\
    &+ \alpha_3 \frac{1}{hN_0} \sum\limits_{i=1}^{N_0} \sum\limits_{j\in \Omega_i}\textbf{I}_{f(j)\notin \Omega_{f(i)}}
    \end{array}
    I will use it for finding the area of convex optimization.

    :return: All the parameter for the cost function.
    '''

    t_fun = ta_fun.copy()

    if n_pre is not None:
        t_fun[n_pre[0]] = n_pre[1]

    N_0 = len(t_fun)
    nan_target, notnan_target = nan_or_not(t_fun)

    # coefficient of alpha0
    w_cost = weight_loss(cen0, cen1, neigh0, t_fun)

    # coefficient of alpha1
    V_cost = v_cost_tau(notnan_target, v0, v1, tau) / N_0

    # coefficient of alpha2
    same_s_cost = predict_overlap(notnan_target, notnan_target, same_set=True) / N_0

    # coefficient of alpha3
    neigh_cost = neighbor_cost(neigh0, neigh1, t_fun, notnan_target) / (avg_neigh * N_0)

    return w_cost, V_cost, same_s_cost, neigh_cost


def new_total_cost_no_split(cen0, cen1, v0, v1, neigh0, neigh1, ta_fun, avg_neigh,
                            alpha_, tau=.2, n_pre=None):
    # finding total cost for each epoch and return separate cost and total cost as
    # output. Inputs are the same as d_cost.

    t_fun = ta_fun.copy()

    if n_pre is not None:
        t_fun[n_pre[0]] = n_pre[1]
        if n_pre[1] != n_pre[1]:
            return np.infty, 0, 0, 0, 0

    N_0 = len(t_fun)
    notnan_target = t_fun.copy()


    # coefficient of alpha0
    w_cost = weight_loss(cen0, cen1, neigh0, t_fun)

    # coefficient of alpha1
    V_cost = v_cost_tau(notnan_target, v0, v1, tau) / N_0

    # coefficient of alpha2
    same_s_cost = predict_overlap(notnan_target, notnan_target, same_set=True) / N_0

    # coefficient of alpha3
    neigh_cost = neighbor_cost(neigh0, neigh1, t_fun, notnan_target) / (avg_neigh * N_0)

    t_cost = alpha_[0] * w_cost + alpha_[1] * V_cost + alpha_[2] * same_s_cost +\
             alpha_[3] * neigh_cost

    return t_cost, w_cost, V_cost, same_s_cost, neigh_cost


def remove_cell_per_length(img, lower_bound, upper_bound, labeled=None):
    '''
    This function will remove cells which has length greater than upper bound and
    less than lower bound
    :param img: binary image
    :param lower_bound:
    :param upper_bound:
    :param lab: image with natural number which those numbers are number of the cell
    :return: labeles image
    '''
    lab = labeled.copy()
    cent, vec = cen_vec(img, save_name="vec0", lab_img=lab)
    vec_norm = np.linalg.norm(vec, axis=1)
    upper_cut = np.where(vec_norm > upper_bound)[0]
    for i in upper_cut:
        lab[lab == i + 1] = 0

    lower_cut = np.where(vec_norm < lower_bound)[0]
    for i in lower_cut:
        lab[lab == i + 1] = 0

    new_number = 1

    for i in range(1, np.max(lab) + 1):
        if (lab == i).any():
            lab[lab == i] = new_number
            new_number += 1

    return lab


def color_cell_per_length(img, labeled, num_sig_step, step_color, coef_sig = 1):
    '''
    This function will remove cells which has length greater than upper bound and
    less than lower bound
    :param img: binary image
    :param lower_bound:
    :param upper_bound:
    :param lab: image with natural number which those numbers are number of the cell
    :return: labeles image
    '''
    lab = labeled.copy()
    test_img = np.zeros(np.shape(labeled))
    cent, vec = cen_vec(img, save_name="vec0", lab_img=labeled)
    vec_norm = np.linalg.norm(vec, axis=1)

    for num_ in range(num_sig_step, 0, -1):
        sig_cut_plus = np.where(vec_norm < np.mean(vec_norm) + num_ * coef_sig * np.std(vec_norm))[0]
        sig_cut_minus = np.where(vec_norm > np.mean(vec_norm) - num_ * coef_sig * np.std(vec_norm))[0]
        sig_cut = list(set(sig_cut_plus) & set(sig_cut_minus))
        for i in sig_cut:
            test_img[lab == i + 1] = 255 - num_ * step_color
    plt.imshow(test_img)
    plt.colorbar()
    plt.show()

    return test_img


def find_target_function(img_seg0, img_seg1, lab0, lab1,
                         al, split_rate=.1, treshhold_=1,
                         iteration=100, T=100, dT=1 / 2):
    # This function gets two image img0 and img1 and related label images
    # beside that it needs parameters for formula and parameters
    # for boltzmann machine algorithm
    # it returns target function which is a function optimized to relate
    # segment labels from img0 to img1. target_function2
    # is the starting point

    # lab0 = np.load('ski30.npy').astype(np.uint8)
    # lab1 = np.load('ski31.npy').astype(np.uint8)
    # img_seg0 = lab0.copy()
    # img_seg0[img_seg0 != 0] = 1
    # img_seg1 = lab1.copy()
    # img_seg1[img_seg1 != 0] = 1
    #
    # al = [8, 500, 2, 1, 1000, 8, .01, 500, 500]
    # split_rate = .1
    # T = 100
    # dT = 1 / 2
    # iteration = 100

    centroid0, vec0 = cen_vec(img_seg0, save_name="vec0", lab_img=lab0)
    centroid1, vec1 = cen_vec(img_seg1, save_name="vec1", lab_img=lab1)

    relation = {}
    for i in range(1, np.max(lab0) + 1):
        _, cen_list = close_cen(centroid1,
                                centroid0[i - 1],
                                win_size=relation_window_size)
        relation[i] = np.int32(cen_list) + 1

    # We find initial optimization with nan for double choice cells
    # tar, tar2 = vf.find_target_function(img_seg0, img_seg1, lab0, lab1,
    #                                     alpha_=al[0], beta_=al[1], gamma_=al[2], thrsh=1,
    #                                     alpha2_=al[3], alpha3_=al[4], split_rate=.1)
    #
    # target_function = {}
    # for k in tar.keys():
    #     if tar[k] == tar[k]:
    #         target_function[k] = [tar[k]]
    #     else:
    #         target_function[k] = choose_nan(relation, k)

    target_function = {}
    for num in relation:
        target_function[num] = choose_target(relation, num, [])

    tri0 = Delaunay(centroid0)
    tri1 = Delaunay(centroid1)

    neighbors0 = {}
    neighbors1 = {}

    neighbors0_nonan = {}
    neighbors1_nonan = {}

    for i in range(1, np.max(lab0) + 1):
        neighbors0[i], neighbors0_nonan[i] = neigh_dis(centroid0[i - 1], i - 1, centroid0,
                                                       tri0, bound_dis=60)

    for i in range(1, np.max(lab1) + 1):
        neighbors1[i], neighbors1_nonan[i] = neigh_dis(centroid1[i - 1], i - 1, centroid1,
                                                       tri1, bound_dis=60)

    avg_neigh_num = mean([len(x) for x in neighbors0.values()])

    cell_num = len(target_function) + 1
    target_function2 = target_function.copy()

    costs = np.zeros([iteration + 1, 11])
    old_cost = total_cost(centroid0, centroid1, vec0, vec1, neighbors0,
                          neighbors1, target_function, relation,
                          avg_neigh_num, al, spl=split_rate, tsh=treshhold_)
    costs[0, :] = old_cost

    for inter_num in tqdm(range(iteration)):
        for num_ in range(1, cell_num):
            old_pred = target_function[num_]
            new_pred = choose_target(relation, num_, old_pred)

            temp_cost = total_cost(centroid0, centroid1, vec0, vec1, neighbors0, neighbors1, target_function, relation
                                   , avg_neigh_num, al, spl=split_rate, tsh=treshhold_, n_pre=[num_, new_pred])

            delta_cost = temp_cost[0] - old_cost[0]

            if delta_cost < 0:
                target_function[num_] = new_pred
                old_cost = temp_cost
            else:
                u = np.random.uniform()
                if u < np.exp(-delta_cost / T):
                    target_function[num_] = new_pred
                    old_cost = temp_cost

        #     delta_cost = diff_cost(new_pred, old_pred, num_,
        #                            centroid0, centroid1, vec0, vec1,
        #                            neighbors0, neighbors1,
        #                            avg_neigh_num, cell_num,
        #                            target_function, relation, alpha_=alpha_,
        #                            beta_=beta_, gamma_=gamma_, thrsh=thrsh,
        #                            alpha2_=alpha2_, alpha3_=alpha3_,
        #                            spl=split_rate)
        #
        #     if delta_cost < 0:
        #         target_function[num_] = new_pred
        #     else:
        #         u = np.random.uniform()
        #         if u < np.exp(-delta_cost / T):
        #             target_function[num_] = new_pred
        #
        # costs.append(total_cost(centroid0, centroid1, vec0, vec1,
        #                         neighbors0, neighbors1, target_function,
        #                         relation, avg_neigh_num, cell_num,
        #                         alpha_=alpha_, beta_=beta_,
        #                         gamma_=gamma_, thrsh=thrsh,
        #                         alpha2_=alpha2_, alpha3_=alpha3_,
        #                         spl=split_rate))
        #
        costs[inter_num + 1, :] = old_cost

        T = T * dT

    ind_cost = cell_cost(centroid0, centroid1, vec0, vec1, neighbors0, neighbors1,
                         target_function, relation, avg_neigh_num, al, spl=.1, tsh=1)

    return target_function, target_function2, ind_cost


def print_data_for_not_correct_one(img0, img1, label0, label1, target_func, cor_tar=None):
    '''
    This function get the parameters and show the values for the wrong
    choices of target functions
    :param img0:
    :param img1:
    :param label0:
    :param label1:
    :param target_func:
    :return:
    '''
    centroid0, vec0 = cen_vec(img0, save_name="vec0", lab_img=label0)
    centroid1, vec1 = cen_vec(img1, save_name="vec1", lab_img=label1)

    relation = {}
    for i in range(1, np.max(lab0) + 1):
        _, cen_list = close_cen(centroid1,
                                centroid0[i - 1],
                                win_size=relation_window_size)
        relation[i] = np.int32(cen_list) + 1

    tri0 = Delaunay(centroid0)
    tri1 = Delaunay(centroid1)

    neighbors0 = {}
    neighbors1 = {}

    neighbors0_nonan = {}
    neighbors1_nonan = {}

    for i in range(1, np.max(label0) + 1):
        neighbors0[i], neighbors0_nonan[i] = neigh_dis(centroid0[i - 1],
                                                       i - 1, centroid0,
                                                       tri0, bound_dis=80)

    for i in range(1, np.max(lab1) + 1):
        neighbors1[i], neighbors1_nonan[i] = neigh_dis(centroid1[i - 1],
                                                       i - 1, centroid1,
                                                       tri1, bound_dis=80)

    avg_neigh_num = mean([len(x) for x in neighbors0.values()])

    for i in range(1, len(target_func) + 1):
        test_target = target_func.copy()

        if cor_tar is None:
            if target_func[i][0] != i:
                params = new_parameter_cost_no_split(centroid0, centroid1,
                                                     vec0, vec1,
                                                     neighbors0_nonan,
                                                     neighbors1_nonan,
                                                     target_func,
                                                     avg_neigh_num)
                test_target[i] = [i]
                params2 = new_parameter_cost_no_split(centroid0, centroid1,
                                                      vec0, vec1,
                                                      neighbors0_nonan,
                                                      neighbors1_nonan,
                                                      test_target,
                                                      avg_neigh_num)

                print(i, target_func[i], 'norm vector:', np.linalg.norm(vec0[i - 1]),
                      np.linalg.norm(vec1[target_func[i][0] - 1]))
                print('parameters without tar[i] = i')
                print(params)
                print('parameters with tar[i] = i')
                print(params2)
        else:
            if target_func[i][0] != cor_tar[i][0]:
                params = new_parameter_cost_no_split(centroid0, centroid1,
                                                     vec0, vec1,
                                                     neighbors0_nonan,
                                                     neighbors1_nonan,
                                                     target_func,
                                                     avg_neigh_num)
                test_target[i] = cor_tar[i][0]
                params2 = new_parameter_cost_no_split(centroid0, centroid1,
                                                      vec0, vec1,
                                                      neighbors0_nonan,
                                                      neighbors1_nonan,
                                                      test_target,
                                                      avg_neigh_num)

                print(i, target_func[i], 'norm vector:', np.linalg.norm(vec0[i - 1]),
                      np.linalg.norm(vec1[target_func[i][0] - 1]))
                print('parameters without tar[i] = i')
                print(params)
                print('parameters with tar[i] = i')
                print(params2)


def find_target_function_no_split(img_seg0, img_seg1, lab0, lab1,
                                  al, split_rate=.1, treshhold_=1,
                                  iteration=100, T=100, dT=1 / 2,
                                  tau_v=.2):
    # This function gets two image img0 and img1 and related label images
    # beside that it needs parameters for formula and parameters
    # for boltzmann machine algorithm
    # it returns target function which is a function optimized to relate
    # segment labels from img0 to img1. target_function2
    # is the starting point

    # lab0 = np.load('ski30.npy').astype(np.uint8)
    # lab1 = np.load('ski31.npy').astype(np.uint8)
    # img_seg0 = lab0.copy()
    # img_seg0[img_seg0 != 0] = 1
    # img_seg1 = lab1.copy()
    # img_seg1[img_seg1 != 0] = 1
    #
    # al = [8, 500, 2, 1, 1000, 8, .01, 500, 500]
    # split_rate = .1
    # T = 100
    # dT = 1 / 2
    # iteration = 100

    centroid0, vec0 = cen_vec(img_seg0, save_name="vec0", lab_img=lab0)
    centroid1, vec1 = cen_vec(img_seg1, save_name="vec1", lab_img=lab1)

    relation = {}
    for i in range(1, np.max(lab0) + 1):
        _, cen_list = close_cen(centroid1,
                                centroid0[i - 1],
                                win_size=relation_window_size)
        relation[i] = np.int32(cen_list) + 1

    # We find initial optimization with nan for double choice cells
    # tar, tar2 = vf.find_target_function(img_seg0, img_seg1, lab0, lab1,
    #                                     alpha_=al[0], beta_=al[1], gamma_=al[2], thrsh=1,
    #                                     alpha2_=al[3], alpha3_=al[4], split_rate=.1)
    #
    # target_function = {}
    # for k in tar.keys():
    #     if tar[k] == tar[k]:
    #         target_function[k] = [tar[k]]
    #     else:
    #         target_function[k] = choose_nan(relation, k)

    target_function = {}
    for num in relation:
        target_function[num] = choose_target_no_split(relation, num, [])

    tri0 = Delaunay(centroid0)
    tri1 = Delaunay(centroid1)

    neighbors0 = {}
    neighbors1 = {}

    neighbors0_nonan = {}
    neighbors1_nonan = {}

    for i in range(1, np.max(lab0) + 1):
        neighbors0[i], neighbors0_nonan[i] = neigh_dis(centroid0[i - 1], i - 1, centroid0,
                                                       tri0, bound_dis=60)

    for i in range(1, np.max(lab1) + 1):
        neighbors1[i], neighbors1_nonan[i] = neigh_dis(centroid1[i - 1], i - 1, centroid1,
                                                       tri1, bound_dis=60)

    avg_neigh_num = mean([len(x) for x in neighbors0.values()])

    cell_num = len(target_function) + 1
    target_function2 = target_function.copy()

    costs = np.zeros([iteration + 1, 5])
    old_cost = new_total_cost_no_split(centroid0, centroid1, vec0,
                                       vec1, neighbors0_nonan,
                                       neighbors1_nonan, target_function,
                                       avg_neigh_num, al,
                                       tau=tau_v)
    costs[0, :] = old_cost

    for inter_num in tqdm(range(iteration)):
        for num_ in range(1, cell_num):
            old_pred = target_function[num_]
            new_pred = choose_target_no_split(relation, num_, old_pred)

            temp_cost = new_total_cost_no_split(centroid0, centroid1, vec0,
                                                vec1, neighbors0_nonan,
                                                neighbors1_nonan, target_function,
                                                avg_neigh_num, al,
                                                tau=tau_v, n_pre=[num_, new_pred])

            delta_cost = temp_cost[0] - old_cost[0]

            if delta_cost < 0:
                target_function[num_] = new_pred
                old_cost = temp_cost
            else:
                u = np.random.uniform()
                if u < np.exp(-delta_cost / T):
                    target_function[num_] = new_pred
                    old_cost = temp_cost

        #     delta_cost = diff_cost(new_pred, old_pred, num_,
        #                            centroid0, centroid1, vec0, vec1,
        #                            neighbors0, neighbors1,
        #                            avg_neigh_num, cell_num,
        #                            target_function, relation, alpha_=alpha_,
        #                            beta_=beta_, gamma_=gamma_, thrsh=thrsh,
        #                            alpha2_=alpha2_, alpha3_=alpha3_,
        #                            spl=split_rate)
        #
        #     if delta_cost < 0:
        #         target_function[num_] = new_pred
        #     else:
        #         u = np.random.uniform()
        #         if u < np.exp(-delta_cost / T):
        #             target_function[num_] = new_pred
        #
        # costs.append(total_cost(centroid0, centroid1, vec0, vec1,
        #                         neighbors0, neighbors1, target_function,
        #                         relation, avg_neigh_num, cell_num,
        #                         alpha_=alpha_, beta_=beta_,
        #                         gamma_=gamma_, thrsh=thrsh,
        #                         alpha2_=alpha2_, alpha3_=alpha3_,
        #                         spl=split_rate))
        #
        costs[inter_num + 1, :] = old_cost

        T = T * dT
        # print("in function find_target_function_no_split accuracy is :", acc_no_split(target_function))

    # ind_cost = cell_cost(centroid0, centroid1, vec0, vec1, neighbors0, neighbors1,
    #                      target_function, relation, avg_neigh_num, al, spl=.1, tsh=1)

    return target_function, target_function2


def acc_no_split(t_fun):
    '''
    This function returns accuracy for the simple target function which value of
    dictionary is equal to key of that value.
    :param t_fun: target function to find the accuracy
    :return: accuracy \in [0,1]
    '''
    counter_ = 0
    for key in t_fun.keys():
        if t_fun[key][0] == key:
            counter_ += 1

    return counter_/len(t_fun)


def result_label(img_lab, labeled1, t_fun, color_=(0, 0, 255)):

    test_tar = {}
    for i in range(1, np.max(labeled1) + 1):
        test_tar[i] = ''

    for key, value in t_fun.items():
        for val in value:
            test_tar[val] = test_tar[val] + '-' + str(key)

    for i in range(1, np.max(labeled1) + 1):
        cord_ = np.where(labeled1 == i)
        y_center = int(np.sum(cord_[0]) / len(cord_[0])) + 1
        x_center = int(np.sum(cord_[1]) / len(cord_[1])) - 1
        cent = np.mean(np.where(labeled1 == i + 1), axis=1)
        cv2.putText(img_lab, test_tar[i], (x_center, y_center),
                    cv2.FONT_HERSHEY_SIMPLEX, 3 / 10, color_, 1)
    return img_lab


if __name__ == "__main__":
    # lab0 = np.load('ski30.npy').astype(np.uint8)
    # lab1 = np.load('ski31.npy').astype(np.uint8)
    # img_seg0 = lab0.copy()
    # img_seg0[img_seg0 != 0] = 1
    # img_seg1 = lab1.copy()
    # img_seg1[img_seg1 != 0] = 1

    #This part is for removing some cells w.r.t. their cell length
    #
    # mum = sio.loadmat('mum16pix')
    # img_seg0 = mum['mum'][0][1][0, 0]
    # img_seg0[img_seg0 > 1] = 0
    # lab0 = label(img_seg0, connectivity=1)
    #
    # img_seg1 = mum['mum'][0][2][0, 0]
    # img_seg1[img_seg1 > 1] = 0
    # lab1 = label(img_seg1, connectivity=1)

    lab0 = np.load('Images/Samples/4/lab0.npy').astype(np.uint8)
    lab1 = np.load('Images/Samples/4/lab1.npy').astype(np.uint8)

    img_seg0 = lab0.copy()
    img_seg0[img_seg0 != 0] = 1
    img_seg1 = lab1.copy()
    img_seg1[img_seg1 != 0] = 1

    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    plt.imshow(img_seg0)
    a.set_title('Before')
    a = fig.add_subplot(1, 2, 2)
    plt.imshow(img_seg1)
    a.set_title('After')
    plt.show()

    rem_point = 12.8
    eps = .05
    g = 1.28
    new_lab0 = remove_cell_per_length(img_seg0, (1 - eps) * rem_point, (1 + eps) * rem_point * 2 / g, lab0)
    new_lab1 = remove_cell_per_length(img_seg1, (1 - eps) * rem_point * g, (1 + eps) * rem_point * 2, lab1)

    img0 = np.uint8(new_lab0.copy())
    img0[img0 != 0] = 1
    img1 = np.uint8(new_lab1.copy())
    img1[img1 != 0] = 1
    #
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    plt.imshow(img0)
    a.set_title('Before')
    a = fig.add_subplot(1, 2, 2)
    plt.imshow(img1)
    a.set_title('After')
    plt.show()

    # mum = sio.loadmat('mum_1min_33slice_310to620_750to1310')
    # img_seg0 = mum['mum'][0][0][0, 0]
    # img_seg0[img_seg0 > 1] = 0
    # img_seg0[159:163, 180:192] = 0
    # img_seg0[168:185, 104:108] = 0
    # img_seg0[124:126, 60:74] = 0
    # lab0 = label(img_seg0, connectivity=1)
    #
    # img_seg1 = mum['mum'][0][1][0, 0]
    # img_seg1[img_seg1 > 1] = 0
    # lab1 = label(img_seg1, connectivity=1)
    # img_seg1[lab1 == 8] = 0
    # lab1 = label(img_seg1, connectivity=1)



    # lab0 = np.load('Images/moving_simul/w/tall0/lab0.npy').astype(np.uint8)
    # lab1 = np.load('Images/moving_simul/w/tall0/lab1.npy').astype(np.uint8)
    # img_seg0 = lab0.copy()
    # img_seg0[img_seg0 != 0] = 1
    # img_seg1 = lab1.copy()
    # img_seg1[img_seg1 != 0] = 1

    # alpha_, beta_, gamma_, thrsh, alpha2_, alpha3_ = 8, 1024, 64, 1, 1, 1000
    # al = [49, 38168, 132322, 133, 55, 5960, 19842, 3, 1599, 1689]
    # test_result = np.zeros((64, 4))
    # test = [.1, 1, 10, 100]
    # num = 0
    # for (a, b, c) in itertools.product(test, test, test):
    #     al = [a, b, 1000, c]
    #     test_result[num, 0] = a
    #     test_result[num, 1] = b
    #     test_result[num, 2] = c
    #     split_rate = .1
    #     T = 100
    #     dT = 1/2
    #     iteration = 100
    # al = [0.28223388, 0.29470904, 0.08980623, 0.33325121]

    # al = [0.30730242, 0.02118294, 0.47354079, 0.19797386]
    # al = [0.43029858, 0.06860031, 0.15977757, 0.34132334]
    # al = [0.17101532, 0.60586056, 0.08832582, 0.13479845]
    # al = [0.05182307, 0.01709711, 0.28051854, 0.65056128]
    # al = [0.36495809, 0.5722658,  0.04903073, 0.01374539]
    al = [0.3, 0.2,  0.3, 0.3]
    tar1, tar2 = find_target_function_no_split(img0, img1, new_lab0,
                                               new_lab1, al, iteration=1500,
                                               T=100, dT=99/100, tau_v=.25)

    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(ms.label_img(ms.for_show(img0), img0, lab=new_lab0))
    a.set_title('Before')
    a = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(result_label(ms.for_show(img1), new_lab1, tar1))
    a.set_title('After')
    plt.show()

        #
        # test_result[num, 3] = acc_no_split(tar1)
        # num += 11

    # print(acc_no_split(tar1))


# img0, img1, img_lab0, img_lab1, cent_chng, new_label_list = make_img(
#     200, 7, save_add, spd=4, rtn=40,
#     d_rtn=10, lg_=(6, 11), d_lg=3, brd=20, split=True)
#
# plt.imshow(img0, cmap=plt.cm.gray, interpolation='nearest')
# plt.axis('off')
# plt.show()
# plt.imshow(img1, cmap=plt.cm.gray, interpolation='nearest')
# plt.axis('off')
# plt.show()




# lab_0 = np.load('ski30.npy').astype(np.uint8)
# lab_1 = np.load('ski31.npy').astype(np.uint8)
# img_0 = lab_0.copy()
# img_0[img_0 != 0] = 1
# img_1 = lab_1.copy()
# img_1[img_1 != 0] = 1

# t_fun = vectorwise_bm(img_0, img_1, lab_0, lab_1,
#                       alpha_, beta_, gamma_, thrsh, alpha2_, alpha3_, T=100)

# cnt = 0
# for num_ in range(1, cell_num):
#     if target_function[num_] == num_:
#         cnt += 1

    # print("accuracy is equal : ", cnt / (cell_num-1))

    # total_table[total_counter, 0] = alpha_
    # total_table[total_counter, 1] = beta_
    # total_table[total_counter, 2] = gamma_
    # total_table[total_counter, 3] = thrsh
    # total_table[total_counter, 4] = alpha2_
    # total_table[total_counter, 5] = alpha3_
    # total_table[total_counter, 6] = cnt / (cell_num-1)
    #
    # total_counter += 1
    # print("total counter:", total_counter)

# end = time.time()
#
# # print("accuracy is equal : ", cnt/cell_num)
# print("computational time is: ", end-start)



#
# img0 = np.zeros((200, 200), np.uint8)
# img1 = np.zeros((200, 200), np.uint8)
# lab_img0 = np.zeros((200, 200), np.uint8)
# lab_img1 = np.zeros((200, 200), np.uint8)
# xv, yv = np.meshgrid(np.arange(20, 180, 20),
#                      np.arange(20, 180, 14))
#
# adaad = [9, 12, 15, 16, 21, 22, 25, 27, 30, 41, 43, 48, 66, 72, 90]
# dovom = 96
# for adad in adaad:
#     dovom += 1
#     x, y = cent_chng[adad - 1, 1, :]
#     x = x - 4
#     y = y - 2
#     img1[lab_img1 == adad] = 0
#     lab_img1[lab_img1 == adad] = 0
#     cv2.ellipse(img1, (x, y), (6, 2), 35, 0, 360, 255, -1)
#     cv2.ellipse(lab_img1, (x, y), (6, 2), 35, 0, 360, adad, -1)
#     cv2.ellipse(img1, (x + 7, y + 9), (6, 2), 35, 0, 360, 255, -1)
#     cv2.ellipse(lab_img1, (x + 7, y + 9), (6, 2), 35, 0, 360, dovom, -1)
#
# save_ad = "Images/mksample/"
# cv2.imwrite(save_ad + "img000.png", img0)
# cv2.imwrite(save_ad + "img111.png", img1)
# cv2.imwrite(save_ad + "lab_img000.png", lab_img0)
# cv2.imwrite(save_ad + "lab_img111.png", lab_img1)



#
# xdix1 = img_seg0.copy()
# xdix2 = img_seg1.copy()
#
# xdix1 = cf.label_img(comp.for_show(xdix1), xdix1)
# xdix2 = cf.label_img(comp.for_show(xdix2), xdix2)
#
# plt.imshow(xdix1)
# plt.savefig("Images/vec_optim/xdix0.png")
# plt.show()
# plt.imshow(xdix2)
# plt.savefig("Images/vec_optim/xdix1.png")
# plt.show()


















# lab0 = label(img_seg0, connectivity=1)
# lab1 = label(img_seg1, connectivity=1)

# img2_seg0 = img_seg0.copy()
# img2_seg1 = img_seg1.copy()
#
# img_test0 = comp.for_show(img_seg0)
# img_test1 = comp.for_show(img_seg1)
# skeleton0 = skeletonize(img2_seg0/255)
# skeleton1 = skeletonize(img2_seg1/255)





# centroid0 = np.zeros((np.max(lab0), 2))
# vec0 = np.zeros((np.max(lab0), 2))
# for i in range(1, np.max(lab0) + 1):
#     cell_img = np.zeros(np.shape(img_seg0))
#     cell_img[lab0 == i] = 1
#     skeleton0 = skeletonize(cell_img)
#     y, x = np.where(skeleton0 == 1)
#     centroid0[i - 1, 0] = np.int(np.mean(x))
#     centroid0[i - 1, 1] = np.int(np.mean(y))
#     A = list(zip(x, y))
#     vec_arg = np.argmax(distance.cdist(centroid0[i - 1, :].
#                                        reshape(1, 2),
#                                        A, 'euclidean')[0])
#     vec0[i - 1, 0] = centroid0[i - 1, 0] - A[vec_arg][0]
#     vec0[i - 1, 1] = centroid0[i - 1, 1] - A[vec_arg][1]
#     cv2.circle(img_test0, tuple(np.int32(centroid0[i - 1])),
#                2, (255, 0, 0))
#     cv2.arrowedLine(img_test0, tuple(np.int32(centroid0[i - 1]))
#                     , A[vec_arg], (0, 0, 255), 1)
#
# centroid1 = np.zeros((np.max(lab1), 2))
# vec1 = np.zeros((np.max(lab1), 2))
# for i in range(1, np.max(lab1)+1):
#     cell_img = np.zeros(np.shape(img_seg1))
#     cell_img[lab1 == i] = 1
#     skeleton1 = skeletonize(cell_img)
#     y, x = np.where(skeleton1 == 1)
#     centroid1[i - 1, 0] = np.int(np.mean(x))
#     centroid1[i - 1, 1] = np.int(np.mean(y))
#     cv2.circle(img_test1, tuple(np.int32(centroid1[i - 1])),
#                2, (255, 0, 0))
#     A = list(zip(x, y))
#     vec_arg = np.argmax(distance.cdist(centroid1[i - 1, :]
#                                        .reshape(1, 2),
#                                        A, 'euclidean')[0])
#     vec1[i - 1, 0] = A[vec_arg][0] - centroid1[i - 1, 0]
#     vec1[i - 1, 1] = A[vec_arg][1] - centroid1[i - 1, 1]
#     cv2.arrowedLine(img_test1, tuple(np.int32(centroid1[i - 1]))
#                     , A[vec_arg], (0, 0, 255), 1)



























# lab0 = label(img_seg0, connectivity=1)
# lab1 = label(img_seg1, connectivity=1)

# img2_seg0 = img_seg0.copy()
# img2_seg1 = img_seg1.copy()
#
# img_test0 = comp.for_show(img_seg0)
# img_test1 = comp.for_show(img_seg1)
# skeleton0 = skeletonize(img2_seg0/255)
# skeleton1 = skeletonize(img2_seg1/255)





# centroid0 = np.zeros((np.max(lab0), 2))
# vec0 = np.zeros((np.max(lab0), 2))
# for i in range(1, np.max(lab0) + 1):
#     cell_img = np.zeros(np.shape(img_seg0))
#     cell_img[lab0 == i] = 1
#     skeleton0 = skeletonize(cell_img)
#     y, x = np.where(skeleton0 == 1)
#     centroid0[i - 1, 0] = np.int(np.mean(x))
#     centroid0[i - 1, 1] = np.int(np.mean(y))
#     A = list(zip(x, y))
#     vec_arg = np.argmax(distance.cdist(centroid0[i - 1, :].
#                                        reshape(1, 2),
#                                        A, 'euclidean')[0])
#     vec0[i - 1, 0] = centroid0[i - 1, 0] - A[vec_arg][0]
#     vec0[i - 1, 1] = centroid0[i - 1, 1] - A[vec_arg][1]
#     cv2.circle(img_test0, tuple(np.int32(centroid0[i - 1])),
#                2, (255, 0, 0))
#     cv2.arrowedLine(img_test0, tuple(np.int32(centroid0[i - 1]))
#                     , A[vec_arg], (0, 0, 255), 1)
#
# centroid1 = np.zeros((np.max(lab1), 2))
# vec1 = np.zeros((np.max(lab1), 2))
# for i in range(1, np.max(lab1)+1):
#     cell_img = np.zeros(np.shape(img_seg1))
#     cell_img[lab1 == i] = 1
#     skeleton1 = skeletonize(cell_img)
#     y, x = np.where(skeleton1 == 1)
#     centroid1[i - 1, 0] = np.int(np.mean(x))
#     centroid1[i - 1, 1] = np.int(np.mean(y))
#     cv2.circle(img_test1, tuple(np.int32(centroid1[i - 1])),
#                2, (255, 0, 0))
#     A = list(zip(x, y))
#     vec_arg = np.argmax(distance.cdist(centroid1[i - 1, :]
#                                        .reshape(1, 2),
#                                        A, 'euclidean')[0])
#     vec1[i - 1, 0] = A[vec_arg][0] - centroid1[i - 1, 0]
#     vec1[i - 1, 1] = A[vec_arg][1] - centroid1[i - 1, 1]
#     cv2.arrowedLine(img_test1, tuple(np.int32(centroid1[i - 1]))
#                     , A[vec_arg], (0, 0, 255), 1)




# lab0 = np.load('Images/moving_simul/w/1/lab0.npy').astype(np.uint8)
# img_seg0 = lab0.copy()
# img_seg0[img_seg0 != 0] = 1
# centroid0, vec0 = cen_vec(img_seg0, save_name="vec0", lab_img=lab0)
# all_vec = np.concatenate((np.linalg.norm(vec0, axis=1),all_vec))
# plt.hist(all_vec)
# plt.show()



# To find m for removing the cell
# for i in range(20):
#     x = 10 + (i/5)
#     new_lab0 = remove_cell_per_length(img_seg0, .9 * x, 2.2 * x, lab0)
#     new_lab1 = remove_cell_per_length(img_seg1, .9 * x, 2.2 * x, lab1)
#
#     img0 = np.uint8(new_lab0.copy())
#     img0[img0 != 0] = 1
#     img1 = np.uint8(new_lab1.copy())
#     img1[img1 != 0] = 1
#
#     fig = plt.figure()
#     a = fig.add_subplot(1, 2, 1)
#     imgplot = plt.imshow(ms.label_img(ms.for_show(img0), img0, lab=new_lab0))
#     a.set_title('Before for i = {}'.format(i))
#     a = fig.add_subplot(1, 2, 2)
#     imgplot = plt.imshow(ms.label_img(ms.for_show(img1), img1, lab=new_lab1))
#     a.set_title('After for i = {}'.format(i))
#     plt.show()
#     print('for i = {}, max lab0 = {} and max lab1 = {}'.format(i, np.max(new_lab0), np.max(new_lab1)))
#
#     fig = plt.figure()
#     a = fig.add_subplot(1, 2, 1)
#     imgplot = plt.imshow(ms.label_img(ms.for_show(img_seg0), img_seg0, lab=lab0))
#     a.set_title('Before')
#     a = fig.add_subplot(1, 2, 2)
#     imgplot = plt.imshow(ms.label_img(ms.for_show(img_seg1), img_seg1, lab=lab1))
#     a.set_title('After')
#     plt.show()








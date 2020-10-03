import synthetic_split_parent_with_children as ssp
import numpy as np
import json
import matplotlib.pyplot as plt
import probprediction as prb
import collections
from tqdm import tqdm
import multiprocessing
import glob
from collections import Counter


# with open('Images/james_sample/with_split/100/17_51_551591750160_0-0_jsonData.json') as f:
# with open('Images/james_sample/with_split/ws.json') as f:
def total_movement_function(f_name):
    with open(f_name) as f:
        json_file = json.load(f)
    fr = []
    for i in range(5):
        fr.append(ssp.Frame(json_file, 354 + i, 'Images/james_sample/with_split/ws.json', hierarchy_lookback=0))
        fr[i].draw()
        fr[i].cen_vec()

    bm = ssp.BM(fr[1], fr[0], .01, .0001, 200, .05)
    # bm.execute_with_neighbor_bm_match_finder()
    bm.execute_first_bm()

    temp_lab0 = fr[0].labeled.copy()
    temp_lab1 = fr[1].labeled.copy()

    truemap = {}
    for i in range(1, 1 + fr[0].cell_number):
        truemap[i] = [i]

    trmp = ssp.true_map_split(fr[1])

    for key, val in bm.pairs_with_parent.items():
        temp_lab0[fr[0].labeled == key] = 0
        fr[0].image[fr[0].labeled == key] = 0
        temp_lab1[fr[1].labeled == val[0]] = 0
        temp_lab1[fr[1].labeled == val[1]] = 0
        try:
            del trmp[val[0][0]]
        except Exception:
            pass
        try:
            del trmp[val[0][1]]
        except Exception:
            pass

        fr[1].image[fr[1].labeled == val[0]] = 0
        fr[1].image[fr[1].labeled == val[1]] = 0

    lab0, tr0, rtr0 = ssp.labtolab(fr[0].image, fr[0].labeled)
    lab1, tr1, trt1 = ssp.labtolab(fr[1].image, fr[1].labeled)

    ress= []
    for _ in range(5):
        one_res = prb.combm_split(100000, 100000, 100000000,
                                  fr[0].image, fr[1].image,
                                  lab0, lab1, trmp, tr0, tr1, fr[0].max_id,
                                  rel_window=40, g_rate=1.05,
                                  epoch=2000, T=1000, dT=.99, epsilon=.005,
                                  nei_win=100, draw_=False)

        ress.append(one_res[1])

    voted_result = major_vote_res(ress)

    split_result = acc_find(bm.pairs, bm, tr0, tr1, split=True)
    nonsplit_res = acc_find(voted_result, bm, tr0, tr1, split=False)
    # new_target = target_creator(bm, voted_result, tr0, tr1)
    # acc = total_accuracy(bm, new_target)
    # sec_acc = ssp.parent_accuracy(bm)

    return nonsplit_res[0], nonsplit_res[1], split_result[0], split_result[1], bm.frame0.file_name


def target_creator(bol, target, lab_translator0, lab_translator1):
    ntarget = collections.defaultdict(list)
    for ke, va in target.items():
        ntarget[lab_translator1[va[0]]].append(lab_translator0[ke])

    for ke, va in bol.pairs_with_parent.items():
        for v in va[0]:
            ntarget[v].append(ke)

    return ntarget


def acc_find(target, bol, lab_translator0, lab_translator1, split=True):
    """
    This function gets target function and result of boltzmann machine and returns number of
     correct and wrong predictions
    :param target: either output of boltzmann machine for without split or if split is true it is not important
    :param bol: object of result of boltzmann machine for split cell
    :param split: if we look at result for split True other wise False
    :param lab_translator0:
    :param lab_translator1:
    :return: correct and wrong predictions
    """
    if split:
        new_target = collections.defaultdict(list)
        for pair in bol.pairs:
            new_target[pair[0]].append(bol.possible_parents[pair])
            new_target[pair[1]].append(bol.possible_parents[pair])
    else:
        new_target = collections.defaultdict(list)
        for ke, va in target.items():
            new_target[lab_translator1[va[0]]].append(lab_translator0[ke])

    s = 0
    for key, val in new_target.items():
        if [bol.frame1.cell_list[key].parent] == val:
            s += 1

    return s, len(new_target) - s


def total_accuracy(bol, target_func):
    s = 0
    for k, v in bol.frame1.cell_list.items():
        if target_func[k] == [v.parent]:
            s += 1

    return s/bol.frame1.cell_number


def major_vote_res(res_file):
    """
    This function return the majority vote over multiple mapping (prediction) dictionary
    :param res_file:
    :return:
    """
    tem_dic = collections.defaultdict(list)

    for j in range(len(res_file)):
        for key, val in res_file[j].items():
            tem_dic[key].append(val[0])

    for key, val in tem_dic.items():
        tem_dic[key] = [Counter(tem_dic[key]).most_common()[0][0]]

    return tem_dic


##################################################################
####################### multiprocessing over files ###############
list_of_files = glob.glob('Images/james_sample/with_split/100/*.json')
results = []
with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
    # for _ in tqdm(pool.istarmap(foo, iterable),
    #                    total=len(iterable)):
    #     pass

    print(multiprocessing.cpu_count())
    # results = pool.imap_unordered(BM_implement, itertools.product(ll, ll, ll))

    with tqdm(total=len(list_of_files)) as pbar:
        for i, res in enumerate(pool.imap_unordered(total_movement_function, list_of_files)):
            pbar.update()
            results.append(res)

np.save('/home/sorena/servers/storage/SS/Images/Samples/same_lam/one_min_whole.npy', results)
##################################################################














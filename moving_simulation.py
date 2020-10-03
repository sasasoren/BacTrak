import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cv2
from skimage.measure import label
import scipy.io as sio
import pickle


def new_vec(v_f, v=None, cell_num=3, al=.6, be=.4, v_a=-5, v_b=6):
    '''
    This function returns \alpha V + \beta V_f
    :param v_f: vector of flow
    :param v: movement vector of the cell if there is nothing it will
    make a random vector
    :param cell_num: number of cells
    :param al: alpha parameter of formula
    :param be: beta parameter of formula
    :return: \alpha V + \beta V_f
    '''
    if v is None:
        v = np.random.randint(v_a, v_b, (cell_num, 2))

    return al * v + be * v_f


def lab_move(labnd, v):
    '''
    This function moves cells of the labeled cells with respect to their
    related vector from v
    :param labnd: layered labeled binary cell. we have n = number of cell
    layer and in each layer we have binary image of related cell number
    :param v: vector of movement the cells. for each cell we have separate
    vector
    :return: labeled n dimentional image of cells.(labnd after
     vector v movemnet))
    '''
    if np.isnan(v).any():
        v[np.where(np.isnan(v))] = 0

    ls = labnd.shape
    new_lab = np.zeros(ls)
    # temp_lab = np.zeros((ls[0], ls[1]))
    for num in range(ls[2]):
        if (np.where(labnd[:, :, num] == 1)[0]+int(v[num, 0]) >= ls[0]).any():
            new_lab[np.where(labnd[:, :, num] == 1)[0] - int(v[num, 0]),
                    np.where(labnd[:, :, num] == 1)[1] + int(v[num, 1]), num] = 1
        elif (np.where(labnd[:, :, num] == 1)[1]+int(v[num, 1]) >= ls[1]).any():
            new_lab[np.where(labnd[:, :, num] == 1)[0] + int(v[num, 0]),
                    np.where(labnd[:, :, num] == 1)[1] - int(v[num, 1]), num] = 1
        else:
            new_lab[np.where(labnd[:, :, num] == 1)[0] + int(v[num, 0]),
                    np.where(labnd[:, :, num] == 1)[1] + int(v[num, 1]), num] = 1

    return new_lab


def vect_correct(labnd, vec, v_a=-5, v_b=6):
    '''
    correcting the direction of vectors that they caused overlap
    :param labnd: layered labeled binary cell. we have n = number of cell
    layer and in each layer we have binary image of related cell number
    :param vec: vector of movement the cells. for each cell we have separate
    vector
    :return: new vector after correcting the vector caused overlap.
    '''
    lab_sum = np.sum(labnd, axis=2)

    for n in range(len(vec)):
        if np.max(lab_sum[np.where(labnd[:, :, n] == 1)]) >= 3:
            vec[n] = -vec[n]
        elif np.max(lab_sum[np.where(labnd[:, :, n] == 1)]) == 2:
            cent0 = np.array([np.mean(np.where(labnd[:, :, n])[0]),
                              np.mean(np.where(labnd[:, :, n])[1])])

            eff_vecs = []
            for j in range(len(vec)):
                if j == n:
                    continue
                else:
                    if np.max(labnd[:, :, j][np.where(
                            labnd[:, :, n] == 1)]) > 0:
                        eff_vecs.append(vec[j])

                        cent1 = np.array([np.mean(np.where(
                            labnd[:, :, j])[0]),
                                          np.mean(np.where(
                                              labnd[:, :, j])[1])])

                        v_cent = cent0 - cent1
                        if np.linalg.norm(v_cent) > 10:
                            # print("It was v_cent for", n, " and ", j, "is ", v_cent)
                            v_cent = np.divide(v_cent, 10)
                        # print("v_cent for", n, " and ", j, "is ", v_cent)
                        eff_vecs.append(v_cent)

            vec[n] = np.mean(eff_vecs, axis=0)

        elif np.max(lab_sum[np.where(labnd[:, :, n] == 1)]) == 1:
            vec[n] = np.random.randint(v_a, v_b, (1, 2))

    if np.shape(labnd)[2] > np.shape(vec)[0]:
        temp_vec = np.zeros((np.shape(labnd)[2], 2))
        temp_vec[0:np.shape(vec)[0], :] = vec

        return temp_vec

    return vec


def cell_rotate(labnd, all_theta):
    '''
    This function rotate each cell with respect to related specified theta from all_theta
    :param labnd: layered labeled binary cell. we have n = number of cell
    :param all_theta: list of degree of rotation for each cell
    :return: layered labeled binary cell after rotation for related all_theta degrees
    '''
    if len(labnd.shape) < 3:
        lab_d = 1
    else:
        lab_d = labnd.shape[2]

    for i in range(lab_d):
        if len(labnd.shape) < 3:
            cell_slide = labnd
        else:
            cell_slide = labnd[:, :, i]

        theta = all_theta[i]

        grid_x, grid_y = np.mgrid[0:cell_slide.shape[0], 0:cell_slide.shape[1]]
        cent = np.array([np.mean(np.where(cell_slide == 1)[0]), np.mean(np.where(cell_slide == 1)[1])])
        pts0 = np.where(cell_slide == 1)[0] - cent[0], np.where(cell_slide == 1)[1] - cent[1]
        xy = np.array([pts0[0] * np.cos(theta * np.pi / 180) -
                       pts0[1] * np.sin(theta * np.pi / 180),
                       pts0[0] * np.sin(theta * np.pi / 180) +
                       pts0[1] * np.cos(theta * np.pi / 180)]).T
        pts1 = np.array([xy[:, 0] + cent[0], xy[:, 1] + cent[1]]).T
        values = np.ones(np.shape(np.where(cell_slide == 1)[0]))
        img_interpolate = griddata(pts1, values, (grid_x, grid_y), method='linear')
        if len(labnd.shape) < 3:
            labnd = np.uint8(np.matrix.round(img_interpolate))
        else:
            labnd[:, :, i] = np.uint8(np.matrix.round(img_interpolate))


    return labnd


def split_cell(labnd, lab_first, lab_sec, splt_dic):
    '''
    this function split cells from splt_dic
    :param labnd: layered labeled binary cell. we have n = number of cell
    :param lab_first: labeled image 0 which is not layered.
    :param lab_sec: labeled image 1 which is not layered.
    :param splt_dic: dictionary which link each cells to splited cell in the second image
    :return: labnd after adding new layers from slit dictionary and dictionary of related new cells
    '''
    output_dic = {}

    for key in splt_dic.keys():
        lab_shape = np.shape(labnd)
        temp_lab = np.zeros((lab_shape[0], lab_shape[1], lab_shape[2] + 1))
        temp_lab[:, :, 0:lab_shape[2]] = labnd
        temp_lab[:, :, key - 1] = 0

        cent_first = np.mean(np.where(lab_first == key), axis=1)
        cent_sec0 = np.mean(np.where(lab_sec == splt_dic[key][0]), axis=1)
        cent_sec1 = np.mean(np.where(lab_sec == splt_dic[key][1]), axis=1)

        dis = cent_first - np.mean((cent_sec0, cent_sec1), axis=0)

        pix_0 = np.where(lab_sec == splt_dic[key][0])[0] + np.int(dis[0]),\
                np.where(lab_sec == splt_dic[key][0])[1] + np.int(dis[1])

        pix_1 = np.where(lab_sec == splt_dic[key][1])[0] + np.int(dis[0]), \
                np.where(lab_sec == splt_dic[key][1])[1] + np.int(dis[1])

        temp_lab[pix_0[0], pix_0[1], key - 1] = 1
        temp_lab[pix_1[0], pix_1[1], lab_shape[2]] = 1

        output_dic[key] = [key, lab_shape[2] + 1]
        labnd = temp_lab.copy()

    return labnd, output_dic


def layered_to_labeled(layered):
    '''
    get layered array and return labeled image
    :param layered: layered labeled binary cell. we have n = number of cell
    :return: labeled image
    '''
    sh = np.shape(layered)
    lab = np.zeros((sh[0], sh[1]))
    for n in range(sh[2]):
        lab[layered[:, :, n] == 1] = n + 1

    return lab


# def show_lab(bin_img, lab):
#     for n in range(np.max(lab)):
#         cent = np.mean(np.where(lab == n + 1), axis=1)
#         cv2.putText(bin_img, str(n + 1), cent, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))


def label_img(img_lab, img_bin, color_=(0, 0, 255), lab=None):
    """
    write the label of cell on the img_lab image
    :param img_lab: RGB image of binary cell
    :param img_bin: binary image of cells
    :param color_: color for the names
    :param lab: labeled image
    :return: image of cells with its label
    """
    if lab is None:
        labeled_img = label(img_bin, connectivity=1)
    else:
        labeled_img = lab

    # print("Number of cells: ", np.max(labeled_img) + 1)
    for i in range(1, np.max(labeled_img) + 1):
        cord_ = np.where(labeled_img == i)
        y_center = int(np.sum(cord_[0]) / len(cord_[0])) + 1
        x_center = int(np.sum(cord_[1]) / len(cord_[1])) - 1
        cent = np.mean(np.where(labeled_img == i + 1), axis=1)
        cv2.putText(img_lab, str(i), (x_center, y_center),
                    cv2.FONT_HERSHEY_SIMPLEX, 3 / 10, color_, 1)
    return img_lab


def for_show(img):
    if np.max(img) == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) * 255
    elif np.max(img) == 2:
        img[img == 1] = 150
        img[img == 2] = 255
        img = cv2.cvtColor(np.uint8(img), cv2.COLOR_GRAY2BGR)
    elif np.max(img) == 3:
        img[img == 1] = 100
        img[img == 2] = 180
        img[img == 3] = 255
        img = cv2.cvtColor(np.uint8(img), cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def move_with_split_rot(lab0, lab1, split_dictionary,
                        mu_theta=3, std_theta=.2,
                        step=2):

    # mu_theta=3
    # std_theta=.2
    # step=2
    # lab0 = np.load('ski30.npy').astype(np.uint8)
    # lab0[lab0 == 57] = 56
    # lab0[lab0 == 58] = 57
    # lab0[lab0 == 59] = 58
    # lab1 = np.load('ski31.npy').astype(np.uint8)
    # img_seg0 = lab0.copy()
    # img_seg0[img_seg0 != 0] = 1
    # img_seg1 = lab1.copy()
    # img_seg1[img_seg1 != 0] = 1
    #
    # split_dictionary = {29: [32, 37], 30: [33, 35], 39: [46, 47], 56: [61, 62]}

    l = list(lab0.shape)
    l.append(np.max(lab0))

    layered_lab0 = np.zeros((l[0], l[1], l[2]))

    for n in range(np.max(lab0)):
        layered_lab0[np.where(lab0 == n + 1)[0], np.where(lab0 == n + 1)[1], n] = 1

    plt.imshow(np.sum(layered_lab0, axis=2))
    plt.show()
    v = np.zeros((l[2], 2))
    v[:, 0], v[:, 1] = 4, 3
    nv = new_vec(v, cell_num=l[2])
    layered_lab1 = lab_move(layered_lab0, nv)
    plt.imshow(np.sum(layered_lab1, axis=2))
    plt.show()
    layered_lab1, change_dic = split_cell(layered_lab1, lab0, lab1, split_dictionary)
    l = list(layered_lab1.shape)
    v = np.zeros((l[2], 2))
    v[:, 0], v[:, 1] = 4, 3
    a_thet = mu_theta + std_theta * np.random.randn(l[2])
    layered_lab1 = cell_rotate(layered_lab1, a_thet)
    plt.imshow(np.sum(layered_lab1, axis=2))
    plt.show()
    # nv = vect_correct(layered_lab1, nv)
    #
    # nv = new_vec(v, nv, cell_num=l[2])
    #
    # layered_lab1 = lab_move(layered_lab1, nv)
    # plt.imshow(np.sum(layered_lab1, axis=2))
    # plt.show()
    for cou in range(step):
        nv = vect_correct(layered_lab1, nv)
        nv = new_vec(v, nv, cell_num=l[2])
        layered_lab1 = lab_move(layered_lab1, nv)
        if cou == (step - 1):
            plt.imshow(np.sum(layered_lab1, axis=2))
            plt.show()

    return layered_to_labeled(layered_lab1), change_dic


def cell_show_num(lab, im, cell_num):
    img = im.copy()
    img[np.where(lab == cell_num)] = [0, 0, 255]
    plt.imshow(img)
    plt.show()


def move_without_split_rot(lab0, lab1, mu_theta=3, std_theta=.2,
                        step=2, fl=(3, 4)):

    l = list(lab0.shape)
    l.append(np.max(lab0))

    layered_lab0 = np.zeros((l[0], l[1], l[2]))

    for n in range(np.max(lab0)):
        layered_lab0[np.where(lab0 == n + 1)[0], np.where(lab0 == n + 1)[1], n] = 1

    plt.imshow(np.sum(layered_lab0, axis=2))
    plt.show()
    v = np.zeros((l[2], 2))
    v[:, 0], v[:, 1] = fl[0], fl[1]
    nv = new_vec(v, cell_num=l[2])
    layered_lab1 = lab_move(layered_lab0, nv)
    plt.imshow(np.sum(layered_lab1, axis=2))
    plt.show()
    l = list(layered_lab1.shape)
    v = np.zeros((l[2], 2))
    v[:, 0], v[:, 1] = fl[0], fl[1]
    a_thet = mu_theta + std_theta * np.random.randn(l[2])
    layered_lab1 = cell_rotate(layered_lab1, a_thet)
    plt.imshow(np.sum(layered_lab1, axis=2))
    plt.show()
    # nv = vect_correct(layered_lab1, nv)
    #
    # nv = new_vec(v, nv, cell_num=l[2])
    #
    # layered_lab1 = lab_move(layered_lab1, nv)
    # plt.imshow(np.sum(layered_lab1, axis=2))
    # plt.show()
    for cou in range(step):
        nv = vect_correct(layered_lab1, nv)
        nv = new_vec(v, nv, cell_num=l[2])
        layered_lab1 = lab_move(layered_lab1, nv)
        if cou == (step - 1):
            plt.imshow(np.sum(layered_lab1, axis=2))
            plt.show()

    plt.imshow(np.sum(layered_lab1, axis=2))
    plt.show()

    return layered_to_labeled(layered_lab1)


def pix_to_ellipse(lab, c_num):
    '''
    This function gets labaled image and a number find a related ellipse for that
    cell.
    :param lab: labeled image
    :param c_num: number of the cell which we want to find related ellipse
    :return: new_lab is the image with the same size but 255 for the related ellipse
    center_ is the center of the cell. hw is the height and width of the ellipse
    theta is the related angle
    '''
    new_lab = np.zeros((np.shape(lab)))
    q = np.where(lab == c_num)
    w = q - np.mean(q, axis=1)[:, None]
    u, s, v = np.linalg.svd(w)
    proje = u.dot(w)
    hw = np.max(proje, axis=1)
    theta = np.arccos(u[0, 1])*180/np.pi
    center_ = tuple((np.int32(np.mean(q, axis=1))[1],
                    np.int32(np.mean(q, axis=1))[0]))
    cv2.ellipse(new_lab, center_, tuple(np.int32(hw)), -theta, 0, 360, 255, -1)

    return new_lab, center_, hw, theta


def make_layered_ellipse_image(lab):
    '''
    this function wil get lab image and return ellipses instead of each image plus
     related data.
    :param lab: labeled image of segmentation
    :return: new_lab: labeled image after swap with ellipse,
     center_: center of each ellipse, hw: height and width of each ellipse,
      theta: related rotation of each ellipse
    '''
    l = list(lab.shape)
    l.append(np.max(lab))

    new_lab = np.zeros((l[0], l[1], l[2]))

    max_lab = np.max(lab)
    center_ = np.zeros((max_lab, 2))
    hw = np.zeros((max_lab, 2))
    theta = np.zeros(max_lab)
    for i in range(max_lab):
        temp_lab, center_[i, :], hw[i, :], theta[i] = pix_to_ellipse(lab, i + 1)
        new_lab[np.where(temp_lab == 255)[0], np.where(temp_lab == 255)[1], i] = 1

    return new_lab, center_, hw, theta


def make_ellipse_image(size_, center_, hw, theta):
    new_lab = np.zeros(size_)
    for i in range(len(theta)):
        cv2.ellipse(new_lab, tuple(center_[i]), tuple(np.int32(hw[i])),
                    -theta[i], 0, 360, i+1, -1)

    return new_lab


def ellipse_lab_move(labnd, v, center_, hw, theta):
    '''
    This function moves ellipse pseudo-cells of the labeled cells
    with respect to their related vector from v
    :param labnd: layered labeled binary cell. we have n = number of cell
    layer and in each layer we have binary image of related cell number
    :param v: vector of movement the cells. for each cell we have separate
    vector
    :return: labeled n-dimension image of cells.(labnd after
     vector v movement))
    '''
    if np.isnan(v).any():
        v[np.where(np.isnan(v))] = 0

    ls = labnd.shape
    new_lab = np.zeros(ls)

    for num in range(ls[2]):
        temp_lab = np.zeros((ls[0], ls[1]))
        cv2.ellipse(temp_lab, (int(center_[num, 0] + v[num, 0]),
                               int(center_[num, 1] + v[num, 1])),
                    tuple(np.int32(hw[num])), -theta[num], 0, 360, 1, -1)
        if (np.where(temp_lab == 1)[0] >= ls[0]).any() or \
                (np.where(temp_lab == 1)[0] <= 0).any():
            v[num, 0] = -v[num, 0]
        if (np.where(temp_lab == 1)[1] >= ls[1]).any() or \
                (np.where(temp_lab == 1)[1] <= 0).any():
            v[num, 1] = -v[num, 1]

        new_temp = np.zeros((ls[0], ls[1]))
        cv2.ellipse(new_temp, (int(center_[num, 0] + v[num, 0]),
                               int(center_[num, 1] + v[num, 1])),
                    tuple(np.int32(hw[num])), -theta[num], 0, 360, 1, -1)
        new_lab[:, :, num] = new_temp.copy()

    return new_lab


def moving_ellipses(lab, dif_length=(1.15, 1.25), dif_theta=(-10, 10),
                    step=2, vec=(3, 4)):
    layered_lab, center_, hw, theta = make_layered_ellipse_image(lab)
    np.save('Images/moving_simul/w/tall0/lab0.npy', layered_to_labeled(layered_lab))
    plt.imshow(for_show(np.sum(layered_lab, axis=2)))
    plt.savefig('Images/moving_simul/w/tall0/ms1-2.png')
    plt.show()
    growth = np.random.uniform(dif_length[0], dif_length[1], (len(hw), 1))
    hw[:, 0] = np.multiply(hw[:, 0], growth.transpose())

    l = np.shape(layered_lab)
    v = np.zeros((l[2], 2))
    v[:, 0], v[:, 1] = vec[0], vec[1]
    theta = theta + np.random.randint(dif_theta[0], dif_theta[1], l[2])
    nv = new_vec(v, cell_num=l[2])
    layered_lab = ellipse_lab_move(layered_lab, nv, center_, hw, theta)
    plt.imshow(for_show(np.sum(layered_lab, axis=2)))
    plt.show()
    for cou in range(step):
        nv = vect_correct(layered_lab, nv)
        nv = new_vec(v, nv, cell_num=l[2])
        layered_lab = ellipse_lab_move(layered_lab, nv, center_, hw, theta)
        plt.imshow(for_show(np.sum(layered_lab, axis=2)))
        plt.savefig('Images/moving_simul/w/tall0/ms1-3.png')
        plt.show()

    return layered_to_labeled(layered_lab)


if __name__ == "__main__":
    lab0 = np.load('ski30.npy').astype(np.uint8)
    # lab0[lab0 == 57] = 56
    # lab0[lab0 == 58] = 57
    # lab0[lab0 == 59] = 58
    lab1 = np.load('ski31.npy').astype(np.uint8)
    img_seg0 = lab0.copy()
    img_seg0[img_seg0 != 0] = 1
    img_seg1 = lab1.copy()
    img_seg1[img_seg1 != 0] = 1
    #
    # split_dictionary_ski31 = {29: [32, 37], 30: [33, 35], 39: [46, 47], 56: [61, 62]}

    # mum = sio.loadmat('mum16pix')
    # img_seg0 = mum['mum'][0][1][0, 0]
    # img_seg0[img_seg0 > 1] = 0
    # lab0 = label(img_seg0, connectivity=1)
    #
    # img_seg1 = mum['mum'][0][2][0, 0]
    # img_seg1[img_seg1 > 1] = 0
    # lab1 = label(img_seg1, connectivity=1)

    # spl_dic2_1_to_1 = {6: [7, 8], 8: [5, 15], 11: [14, 20], 14: [16, 19], 16: [21, 34], 13: [12, 17]}
    # spl_dic2_1_to_1 = {23: [32, 26], 32: [55, 42], 43: [52, 65], 49: [59, 60], 53: [66, 67], 59: [72, 76]}
    # spl_dic0_1_to_1 = {11: [12, 14], 15: [15, 30], 17: [16, 29], 18: [26, 31], 21: [25, 51], 32: [39, 60],
    #                    48: [65, 81], 58: [70, 79], 72: [89, 100]}
    # , 14:[11,17]}
    plt.imshow(for_show(img_seg0))
    plt.savefig('Images/moving_simul/w/tall0/ms1-1.png')
    plt.show()
    new_lab = moving_ellipses(lab0, step=5)

    # new_lab1 = move_without_split_rot(lab0, lab1, step=4)
    #
    # np.save('Images/moving_simul/w/tall0/lab0.npy', lab0)
    np.save('Images/moving_simul/w/tall0/lab1.npy', new_lab)
    #
    # with open('Images/moving_simul/3/rel_dic.pkl', 'wb') as f:
    #     pickle.dump(new_spl_dic, f, pickle.HIGHEST_PROTOCOL)








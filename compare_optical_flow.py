import cv2
import numpy as np
from conv_func import conv_fun, label_img,\
    col_cell
import scipy.io as sio
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import griddata
from scipy.spatial import distance
from skimage.morphology import skeletonize


cell_num = 0

save_add = "Images/compare_opt_flow/"



def for_show(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def res_img(f1):
    img = np.uint8((f1 - np.min(f1))*255/(np.max(f1) - np.min(f1)))
    img = for_show(img)
    return img

def grad_flow(org_flow, name_of_flow,
              save_name, address=save_add):
    # sys.stdout = open(address +
    #                   "grad_flow.txt", "a")
    # print("\n")
    # print("\n")
    # print(save_name)
    # print("values of smoothness for velocity of", name_of_flow)

    # org_flow = org_flow.T
    # org_flow = np.int32(org_flow + 0.5)
    # org_flow = org_flow + np.min(org_flow)
    org_flow = org_flow.T
    l = np.shape(org_flow)
    grad_img = np.zeros((l[1] - 1, l[2] - 1, 2))

    grad_img[:, :, 0] = np.array(
        org_flow[0, 0:l[1] - 1, 0:l[2] - 1] -
        org_flow[0, 1:l[1], 1:l[2]])

    grad_img[:, :, 1] = np.array(
        org_flow[1, 0:l[1] - 1, 0:l[2] - 1] -
        org_flow[1, 1:l[1], 1:l[2]])

    abs_grad_img = np.abs(grad_img)

    L1_norm_grad = abs_grad_img[:, :, 1] + \
                   abs_grad_img[:, :, 0]

    grad_num = np.sum(L1_norm_grad) / ((l[1]-1) * (l[2]-1))
    # print(grad_num)
    plt.imshow(L1_norm_grad)
    plt.savefig(address + save_name + ".png")
    plt.show()
    plt.imshow(grad_img[:, :, 0])
    plt.savefig(address + save_name + "x.png")
    plt.show()
    plt.imshow(grad_img[:, :, 1])
    plt.savefig(address + save_name + "y.png")
    plt.show()

    # sys.stdout.close()


def cell_smoothness(org_flow, img_seg,
                    name_of_flow, cell_num,
                    save_name, address=save_add,
                    whole_=False):
    # sys.stdout = open(address + "cell_smoothness.txt", "a")
    # print("\n")
    # print("\n")
    # print(save_name)

    l = np.shape(img_seg)
    # print(l)

    idx_label = label(img_seg, connectivity=1)
    if whole_ == False:
        # print("values of smoothness for velocity of",
        #       name_of_flow, "for cell number ", cell_num)
        x, y = np.where(idx_label == cell_num)

    else:
        # print("values of smoothness for velocity of",
        #       name_of_flow)
        x, y = np.where(idx_label == idx_label)


    # fx, fy = org_flow[y, x].T
    # lines = np.vstack([x, y, x + fx, y + fy, fx, fy]) \
    #     .T.reshape(-1, 3, 2)
    # lines = np.int32(lines + 0.5)
    # lines = list(filter(lambda w: np.all(w < l[0]), lines))
    # lines = list(filter(lambda w: np.all(w >= 0), lines))
    # print("shape lines: ", np.shape(lines))
    # lines = np.array(lines)

    Z = np.zeros((l[0], l[1]))
    norm_flow = np.zeros((l[0], l[1]))
    # print("len(x): ",len(x))
    for i in range(len(x)):
        if x[i] != l[0] - 1 and y[i] != l[1] - 1:
            Z[x[i], y[i]] = np.abs(org_flow[y[i], x[i]].T[0]
                                   - org_flow[y[i], x[i] + 1].T[0]) \
                            + np.abs(org_flow[y[i], x[i]].T[1]
                                     - org_flow[y[i] + 1, x[i]].T[1])

        elif x[i] == l[0] - 1 & y[i] == l[1] - 1:
            Z[x[i], y[i]] = 0

        elif x[i] == l[0] - 1:
            Z[x[i], y[i]] = np.abs(org_flow[y[i], x[i]].T[1]
                                   - org_flow[y[i] + 1, x[i]].T[1])

        elif y[i] == l[1] - 1:
            Z[x[i], y[i]] = np.abs(org_flow[y[i], x[i]].T[0]
                                   - org_flow[y[i], x[i] + 1].T[0])

        norm_flow[x[i], y[i]] = np.sqrt(org_flow[y[i], x[i]].T[0] ** 2
                                        + org_flow[y[i], x[i]].T[1] ** 2)

    avg_Z = np.sum(Z) / (len(x) * 2)
    avg_norm_flow = np.sum(norm_flow) / (len(x) * 2)
    smoothness_value = np.sum(Z) / (2 * np.sum(norm_flow))

    # print("average of smoothness (Z): ", avg_Z)
    # print("average of norm flow: ", avg_norm_flow)
    # print("soomthness value is: ", smoothness_value)


    # sys.stdout.close()
    return smoothness_value


def trans_img(org_img, idx_img, flow, name_of_flow,
              save_name, address=save_add,
              white_=True):
    # output of the function is an image of transported
    # pixels of white or black of segmentation

    sys.stdout = open(address +
                      "trans_img.txt", "a")
    print("\n")
    print("\n")
    print(save_name)
    print("values of sum of squared differences"
          " between transported frame and target frame",
          name_of_flow)

    if white_ == True:
        x, y = np.where(idx_img == 255)
    else:
        x, y = np.where(idx_img == 0)
    fy, fx = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]) \
        .T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    lines = list(filter(lambda w: np.all(w < 181), lines))
    lines = list(filter(lambda w: np.all(w > 19), lines))
    # print("shape lines: ", np.shape(lines))
    lines = np.array(lines)
    if white_ == True:
        Z = np.zeros((161, 161))
    else:
        Z = np.ones((161, 161)) * 255
    lines = np.subtract(lines, 20)

    for i in range(len(lines)):
        x, y = lines[i, 1, :]
        if white_ == True:
            Z[x, y] = 255
        else:
            Z[x, y] = 0

    plt.imshow(Z)
    plt.show()
    plt.savefig(address + save_name + "_transed_idx.png")
    Z = Z[20:141, 20:141]
    idx_cut = idx_img[40:161, 40:161]
    # plt.imshow(idx_cut)
    # plt.show()
    # print("idx_cut shape: ", np.shape(idx_cut))
    idx_diff = np.ones((121, 121)) * 255

    idx_diff[idx_cut == Z] = 0

    # I have to change for  black(white_ =False)
    # too
    if white_ == True:
        num_w = 255
        num_b = 0
    else:
        num_w = 0
        num_b = 255
    num_one = len(np.where(Z == num_w)[0])
    num_one_org = len(np.where(idx_cut == num_w)[0])
    true_one = np.zeros((121, 121))
    true_one[Z == idx_cut] = 255
    true_one[Z == num_b] = 0
    print("true_one from matrix:", np.sum(true_one) / 255)
    num_true_one = np.sum(true_one) / (255 * num_one)
    print('num_one : ', num_one)
    print('num_one_org : ', num_one_org)
    print('true percent for num_true_one: ', num_true_one)

    accu = np.sum(idx_diff) / (121 * 121 * 255)
    print("accuracy: ", 1 - accu)
    cutted_img = org_img[40:161, 40:161]
    cutted_img[idx_diff == 255] = (255, 0, 0)
    plt.imshow(cutted_img)
    plt.savefig(address + save_name + "_org_with_diff.png")
    plt.show()

    sys.stdout.close()


def trans_one_cell(org_img3, org_img_sec3,
                   img_idx,
                   flow,
                   which_num,
                   save_name,
                   lab_img=None,
                   address=save_add):
    l = np.shape(org_img3)
    # sys.stdout = open(address + "trans_one_cell", "a")
    if np.any(img_idx) == None:
        idx_label = lab_img
    else:
        idx_label = label(img_idx, connectivity=1)

    x, y = np.where(idx_label == which_num)
    fy, fx = flow[y, x].T
    # fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]) \
        .T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    lines = list(filter(lambda w: np.all(w < l[0]), lines))
    lines = list(filter(lambda w: np.all(w >= 0), lines))
    # print("shape lines: ", np.shape(lines))
    lines = np.array(lines)
    for (x1, y1), (x2, y2) in lines:
        org_img3[x1, y1] = (255, 0, 0)
        org_img_sec3[x2, y2] = (0, 255, 255)
    plt.imshow(org_img3)
    plt.savefig(address + save_name + ".png")
    plt.show()

    plt.imshow(org_img_sec3)
    plt.savefig(address + save_name + "_sec.png")
    plt.show()


def trans_one_cel_lk(org_img3, org_img_sec3,
                     img_idx,
                     which_num,
                     save_name,
                     lab_img=None,
                     address=save_add):

    # Lucas kanade params
    lk_params = dict(winSize=(15, 15),
                     maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS |
                               cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    if np.any(img_idx) == None:
        idx_label = lab_img
    else:
        idx_label = label(img_idx, connectivity=1)
    x, y = np.where(idx_label == which_num)

    old_points = np.array(list(zip(x, y)), dtype=np.float32)

    new_points, _, _ = cv2.calcOpticalFlowPyrLK(
        org_img3, org_img_sec3, old_points, None, **lk_params)

    new_pts = new_points.astype(int)

    new_pts = list(filter(lambda w: np.all(w < 201), new_pts))
    new_pts = list(filter(lambda w: np.all(w >= 0), new_pts))

    for i in range(len(old_points)):
        org_img3[x[i], y[i]] = (255, 0, 0)
        if i < len(new_pts):
            x2, y2 = new_pts[i]
            org_img_sec3[x2, y2] = (0, 255, 255)

    plt.imshow(org_img3)
    plt.savefig(address + save_name + ".png")
    plt.show()

    plt.imshow(org_img_sec3)
    plt.savefig(address + save_name + "_sec.png")
    plt.show()


def grad_intense(first_img, sec_img, flow, img_seg,
                 save_name, name_of_flow, white_=True,
                 address=save_add, whole=None):
    # sys.stdout = open(address + "grade_intense", "a")
    # print("\n")
    # print("\n")
    # print(save_name)
    # print("absolute values of different of intensity ", "\n"
    #       " between transported frame and target frame",
    #       name_of_flow)

    l = np.shape(first_img)

    if whole != None:
        x, y = np.where(img_seg == img_seg)
    elif white_ == True:
        x, y = np.where(img_seg == 255)
    else:
        x, y = np.where(img_seg == 0)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]) \
        .T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    lines = list(filter(lambda w: np.all(w < l[0]), lines))
    lines = list(filter(lambda w: np.all(w >= 0), lines))
    # print("shape lines: ", np.shape(lines))
    lines = np.array(lines)

    Z = np.zeros((l[0], l[1]))

    for i in range(len(lines)):
        x1, y1 = lines[i, 0, :]
        x2, y2 = lines[i, 1, :]
        Z[x1, y1] = np.abs(first_img[x1, y1] - sec_img[x2, y2])

    int_diff = np.sum(Z)
    # print("leng lines:", len(lines))
    # print("int_diff value: ", int_diff/255)
    # print("leng non zero Z: ", len(np.where(Z != 0)[0]))
    diff_accuracy = int_diff/(len(lines) * 255)
    # print("diff_accuracy : ", diff_accuracy)
    # sys.stdout.close()
    return diff_accuracy


def down_labeled_img(img, img_sec, x_idx):
    idx_label = label(x_idx, connectivity=1)
    # x, y = np.where(idx_label == 70)
    # obj = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    # flow_DIS = obj.calc(img2, img_sec2, None)
    # fx, fy = flow_DIS[y, x].T

    img_pyr = cv2.pyrDown(img)
    img_pyr_sec = cv2.pyrDown(img_sec)


    l = np.shape(img_pyr)
    pr_mat = np.zeros((np.max(idx_label) + 1, l[0], l[1]))

    for i in range((np.max(idx_label))):
        copy_lab = np.zeros(np.shape(x_idx))
        copy_lab[idx_label == i + 1] = 1
        pr_mat[i + 1, :, :] = cv2.pyrDown(copy_lab)

    pr_mat[0, :, :] = 1 - np.sum(pr_mat, axis=0)

    new_label = np.argmax(pr_mat, axis=0)

    return img_pyr, img_pyr_sec, new_label, pr_mat


def draw_flow(img, flow, step=16, point_=None):
    h, w = img.shape[:2]
    if point_ is not None:
        y, x = point_[:, 0].astype(np.uint8), point_[:, 1].astype(np.uint8)
    else:
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (255, 0, 0), -1)

    if point_ is not None:
        return vis, lines
    else:
        return vis


def interpolate_img(img_2d, flow,
                    save_address,
                    save_name="interpolate",
                    method='linear'):
    x, y = np.where(img_2d == img_2d)
    l=np.shape(img_2d)
    grid_x, grid_y = np.mgrid[0:l[0], 0:l[1]]
    fy, fx = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]) \
        .T.reshape(-1, 2, 2)
    # lines = np.int32(lines + 0.5)
    lines = list(filter(lambda w: np.all(w < l[0]), lines))
    lines = list(filter(lambda w: np.all(w >= 0), lines))
    lines = np.array(lines)
    xy = np.zeros((2, np.size(lines[:, 1, 0])))
    xy[0] = lines[:, 1, 0]
    xy[1] = lines[:, 1, 1]
    pts = xy.T
    # prepts = (lines[:, 0, 0], lines[:, 0, 1])
    prepts = (np.int32(lines[:, 0, 0]), np.int32(lines[:, 0, 1]))
    values = img_2d[prepts]

    img_interpolate = griddata(pts, values, (grid_x, grid_y), method=method)
    img_show = np.uint8(img_interpolate)
    # gray_inter = cv2.cvtColor(img_interpolate, cv2.COLOR_GRAY2BGR)
    img_show = for_show(img_show)
    plt.imshow(img_show)
    plt.savefig(save_address + save_name + ".png")
    plt.show()
    return img_interpolate


def some_cell_flow(org_img1d, org_img_sec1d,
                   org_seg, org_seg_sec,
                   cell_list1, cell_list2,
                   flow_num=1, lucas=None,
                   save_address=save_add):

    # This function ask for list of cells from first and second image and
    # will set the rest of pixels equal to mean of outside of cells after
    # we are computing glow with respect to new images
    lab_img = label(org_seg, connectivity=1)
    lab_img_sec = label(org_seg_sec, connectivity=1)

    # We are setting all the pixels except pixels in cell_list equal to mean
    # all the pixels of outside of cells
    lab_img[~np.in1d(lab_img.ravel(),
                     cell_list1).reshape(lab_img.shape)] = 0
    lab_img_sec[~np.in1d(lab_img_sec.ravel(),
                         cell_list2).reshape(lab_img_sec.shape)] = 0

    org_img1d[lab_img == 0] = int(np.mean(org_img1d[org_seg == 0]))
    org_img_sec1d[lab_img_sec == 0] = int(np.mean(org_img_sec1d[org_seg_sec == 0]))

    first_gray = cv2.cvtColor(org_img1d, cv2.COLOR_GRAY2BGR)
    second_gray = cv2.cvtColor(org_img_sec1d, cv2.COLOR_GRAY2BGR)

    if lucas == None:

        if flow_num == 0:
            flow_DIS = cv2.calcOpticalFlowFarneback(org_img1d, org_img_sec1d, None,
                                                      0.5, 3, 15, 3, 5,
                                                      1.2, 0)
        elif flow_num == 1:
            obj = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
            obj.setGradientDescentIterations(100)
            obj.setUseMeanNormalization(1)
            obj.setVariationalRefinementAlpha(10)
            flow_DIS = obj.calc(org_img1d, org_img_sec1d, None)
        elif flow_num ==2:
            obj = cv2.optflow.DualTVL1OpticalFlow_create()
            flow_DIS = obj.calc(org_img1d, org_img_sec1d, None)
            # flow_Dual_pyr = obj.calc(img_pyr2, img_pyr_sec2, None)
        elif flow_num == 3:
            obj = cv2.optflow.createOptFlow_DeepFlow()
            flow_DIS = obj.calc(org_img1d, org_img_sec1d, None)
            # flow_Deep_pyr = obj.calc(img_pyr2, img_pyr_sec2, None)
        elif flow_num == 4:
            obj = cv2.optflow.createOptFlow_PCAFlow()
            flow_DIS = obj.calc(org_img1d, org_img_sec1d, None)
            # flow_PCA_pyr = obj.calc(img_pyr2, img_pyr_sec2, None)
        elif flow_num == 5:
            obj = cv2.optflow.createOptFlow_SimpleFlow()
            flow_DIS = obj.calc(org_img1d, org_img_sec1d, None)
            # flow_SF_pyr = obj.calc(img_pyr2, img_pyr_sec2, None)

        # f0 = res_img(flow_DIS[:, :, 0])
        # plt.imshow(f0)
        # # plt.colorbar()
        # plt.savefig(save_address + str(flow_num) + "flowx.png")
        # plt.show()
        # f1 = res_img(flow_DIS[:, :, 1])
        # plt.imshow(f1)
        # # plt.colorbar()
        # plt.savefig(save_address + str(flow_num) + "flowy.png")
        # plt.show()
        # flow_norm = np.sqrt(np.add(np.square(flow_DIS[:, :, 0]),
        #                            np.square(flow_DIS[:, :, 1])))
        # flow_norm = res_img(flow_norm)
        # plt.imshow(flow_norm)
        # # plt.colorbar()
        # plt.savefig(save_address + str(flow_num) + "flow.png")
        # plt.show()
        #
        # plt.imshow(first_gray)
        # # plt.colorbar()
        # plt.savefig(save_address + "org.png")
        # plt.show()
        #
        # plt.imshow(second_gray)
        # # plt.colorbar()
        # plt.savefig(save_address + "org2.png")
        # plt.show()
        #
        # img_inter = interpolate_img(org_img1d, flow_DIS, save_address,
        #                             save_name=str(flow_num)+"interpolate")
        # img_inter = for_show(img_inter)
        # plt.imshow(img_inter)
        # plt.show()
        for cell in cell_list1:
            trans_one_cell(first_gray, second_gray,
                           org_seg,
                           flow_DIS,
                           which_num=cell,
                           save_name=str(flow_num) + "predict",
                           lab_img=None,
                           address=save_address)

    else:
        for cell in cell_list1:
            trans_one_cel_lk(first_gray, second_gray,
                             org_seg,
                             which_num=cell,
                             save_name=str(flow_num) + "predict",
                             lab_img=None,
                             address=save_address)
    return org_img1d, org_img_sec1d#, img_inter


def imperf_remover(seg_img):
    # this function removing imperfect segmentation.
    img_seg = seg_img.copy()
    lab1 = label(img_seg)
    sh = np.shape(img_seg)
    for i in range(np.max(lab1)+1):
        x, y = np.where(lab1 == i)
        if any((any(x == 0), any(y == 0),
                any(x == sh[0] - 1), any(y == sh[1] - 1))):
            img_seg[np.where(lab1 == i)] = 0

    return img_seg


def trans_list(img_seg0, img_seg1, flow, cell_num):
    # this function gets number of cell and and
    # returns the list of transferred pixels
    l = np.shape(img_seg1)
    lab1 = label(img_seg0, connectivity=1)
    lab2 = label(img_seg1, connectivity=1)
    x, y = np.where(lab1 == cell_num)
    fy, fx = flow[y, x].T
    lines = np.vstack([x + fx, y + fy]).T.reshape(-1, 2)
    lines = np.int32(lines + 0.5)
    lines = list(filter(lambda w: np.all(w < l[0]), lines))
    lines = list(filter(lambda w: np.all(w >= 0), lines))
    lines = np.array(lines)
    out_list = lab2[lines[:, 0], lines[:, 1]]

    return out_list


def make_pr_list(img_seg0, img_seg1):
    lab0 = label(img_seg0)
    lab1 = label(img_seg1)
    prb_mat = np.zeros((np.max(lab0) + 1, np.max(lab1) + 1))
    count_mat = prb_mat.copy()
    obj = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    flow = obj.calc(img_seg0, img_seg1, None)
    flow[:, :, :] = 0
    for i in range(np.max(lab0) + 1):
        output_list = trans_list(img_seg0, img_seg1, flow, i)
        pix_num = len(output_list)
        unique, counts = np.unique(output_list, return_counts=True)
        count_mat[i, unique] = counts
        prb_mat[i, unique] = counts / pix_num

    return prb_mat, count_mat


def cen_vec(img2d):
    # we have binary image 2 dimension as input and centroid and
    # vectors correspondent with all segments as output.
    lab0 = label(img2d, connectivity=1)
    centroid0 = np.zeros((np.max(lab0), 2))
    vec0 = np.zeros((np.max(lab0), 2))
    img_test0 = for_show(img2d)
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
        vec0[i - 1, 0] = centroid0[i - 1, 0] - A[vec_arg][0]
        vec0[i - 1, 1] = centroid0[i - 1, 1] - A[vec_arg][1]

        cv2.circle(img_test0, tuple(np.int32(centroid0[i - 1])),
                   2, (255, 0, 0))
        cv2.arrowedLine(img_test0, tuple(np.int32(centroid0[i - 1]))
                        , A[vec_arg], (0, 0, 255), 1)

    plt.imshow(img_test0)
    plt.show()

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
        cent_list.append(np.where((centroid == cen).
                                  all(axis=1))[0][0])

    return cents, cent_list


def asli():
    addad = 50
    mat_contents = sio.loadmat('mum-perf-org-new-1-119.mat')
    win_ = [0, 21, 30, 55]
    for num_ in range(1):
        win_num = win_[num_]

        save_add = "Images/compare_opt_flow/" +\
                   str(win_num) + "/"
        # y = mat_contents['A'][0][0][0][win_num]
        # img2 = mat_contents['A'][0][2][0][win_num]
        #
        # y_sec = mat_contents['A'][0][0][1][win_num]
        # img2_sec = mat_contents['A'][0][2][1][win_num]


        # x_idx, x_idx_sec, img3, img3_sec \
        #     = conv_fun(img2, y, img2_sec, y_sec,
        #                img_width=201,
        #                img_length=201,
        #                save_add=save_add,
        #                pr_num_fil=3)
        #
        #
        # cv2.imwrite(save_add + "x_idx_sec.png", x_idx_sec * 255)
        # cv2.imwrite(save_add + "x_idx.png", x_idx * 255)
        # cv2.imwrite(save_add + "optic1.png", img3 * 255)
        # cv2.imwrite(save_add + "optic2.png", img3_sec * 255)

        x_idx_sec = cv2.imread(save_add + "x_idx_sec.png", 0)

        x_idx = cv2.imread(save_add + "x_idx.png", 0)

        img = cv2.imread(save_add + "optic1.png", 1)

        img_sec = cv2.imread(save_add + "optic2.png", 1)

        img2 = cv2.imread(save_add + "optic1.png", 0)

        img_sec2 = cv2.imread(save_add + "optic2.png", 0)

        labeled_img = label(x_idx, connectivity=1)
        regions = regionprops(labeled_img)
        labeled_img_sec = label(x_idx_sec, connectivity=1)
        regions_sec = regionprops(labeled_img_sec)


        obj = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow_DIS = obj.calc(img2, img_sec2, None)
        flow_Farne = cv2.calcOpticalFlowFarneback(img2, img_sec2, None,
                                             0.5, 3, 15, 3, 5,
                                             1.2, 0)

        # trans_img(img_sec, x_idx_sec, flow_DIS,
        #           "white_flow_DIS",
        #           save_name="trans_white_DIS",
        #           address=save_add,
        #           white_=True)
        #
        # trans_img(img_sec, x_idx_sec, flow_DIS,
        #           "black_flow_DIS",
        #           save_name="trans_black_DIS",
        #           address=save_add,
        #           white_=False)
        #
        # trans_img(img_sec, x_idx_sec, flow_Farne,
        #           "white_flow_Farne",
        #           save_name="trans_white_Farne",
        #           address=save_add,
        #           white_=True)
        # trans_img(img_sec, x_idx_sec, flow_Farne,
        #           "black_flow_Farne",
        #           save_name="trans_black_Farne",
        #           address=save_add,
        #           white_=False)

        # grad_flow(flow_DIS, "flow_DIS",
        #           save_name='grad_DIS',
        #           address=save_add)
        #
        # grad_flow(flow_Farne, "flow_Farne",
        #           save_name='grad_Farne',
        #           address=save_add)

        # trans_one_cell(img, img_sec,
    #                x_idx,
    #                flow_DIS,
    #                which_num=addad,
    #                save_name='pic',
    #                address=save_add)
    # trans_one_cell(img, img_sec,
    #                x_idx,
    #                flow_Farne,
    #                which_num=addad,
    #                save_name='pic',
    #                address=save_add)
    #
    # trans_one_cel_lk(img, img_sec,
    #                  x_idx,
    #                  which_num=addad,
    #                  save_name='pic',
    #                  address=save_add)

    # grad_intense(img2, img_sec2, flow_DIS, x_idx,
    #              save_name='DIS', name_of_flow='DIS', white_=True,
    #              address=save_add, whole=1)


if __name__ == '__main__':
    asli()
























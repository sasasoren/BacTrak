import multiprocessing
from collections import Counter
import cv2
from skimage.measure import label
import matplotlib.pyplot as plt
import json
import numpy as np
import probprediction as prb
# import moving_simulation as ms
# import vectorwise_func2 as vf2
import glob
from tqdm import tqdm
from skimage.morphology import skeletonize
from scipy.spatial import distance, Delaunay
import itertools
from operator import itemgetter
from skimage.draw import line


class Cell:
    """
    This Class defines the characteristics of a cell which is centroid id number, length and angle
    """

    def __init__(self, data, frameid, idnumber, org_id, figScale=20):
        self.frameid = frameid
        self.idnumber = idnumber - 1
        self.xcenter = int(figScale * data['frames'][self.frameid]['cells'][self.idnumber]['d'][0])
        self.ycenter = int(figScale * data['frames'][self.frameid]['cells'][self.idnumber]['d'][1])
        self.center = (self.xcenter, self.ycenter)
        self.angle = data['frames'][self.frameid]['cells'][self.idnumber]['d'][2] % (2 * np.pi)
        self.length = figScale * (data['frames'][self.frameid]['cells'][self.idnumber]['d'][3] - 1)
        self.parent = org_id
        self.original_id = org_id
        self.img_center = None
        self.end1 = None
        self.end2 = None
        self.img_length = None


class Frame:
    """
    This class define all data related to a frame. We give the frame number and json file it will extract
    all information from that frame for our algorithm
    """

    def __init__(self, data, frameid, f_name, figscale=20, cell_width=10, hierarchy_lookback=6):
        self.framewidth = data['parameters']['simulationTrapWidthMicrons']
        self.frameheight = data['parameters']['simulationTrapHeightMicrons']
        self.frameid = frameid
        self.file_name = f_name
        self.cell_number = len(data['frames'][self.frameid]['cells'])
        self.cell_list = {}
        self.framescale = figscale
        self.image = np.zeros((self.framewidth * self.framescale, self.frameheight * self.framescale), dtype=np.uint8)
        self.image3 = np.zeros((self.framewidth * self.framescale, self.frameheight * self.framescale, 3),
                               dtype=np.uint8)
        self.labeled = np.zeros((self.framewidth * self.framescale, self.frameheight * self.framescale),
                                dtype=np.uint16)
        self.layered = np.zeros((self.cell_number, self.framewidth * self.framescale,
                                 self.frameheight * self.framescale), dtype=np.uint8)
        self.hierarchy_steps = hierarchy_lookback
        self.hierarchy = {}
        self.split_lab = np.zeros((self.framewidth * self.framescale, self.frameheight * self.framescale),
                                  dtype=np.uint16)
        self.split_img = np.zeros((self.framewidth * self.framescale, self.frameheight * self.framescale),
                                  dtype=np.uint8)
        self.split_list = []

        for i in range(self.cell_number):
            self.cell_list[data['frames'][self.frameid]['cells'][i]['i']] = \
                Cell(data, self.frameid, i + 1,
                     data['frames'][self.frameid]['cells'][i]['i'],
                     figScale=self.framescale)

        for i in range(self.hierarchy_steps, -1, -1):
            for j in range(len(data['frames'][self.frameid - i]['divisions'])):
                try:
                    self.cell_list[data['frames'][self.frameid - i]['divisions'][j][3]].parent = \
                        data['frames'][self.frameid - i]['divisions'][j][1]
                except:
                    pass

                try:
                    self.hierarchy[data['frames'][self.frameid - i]['divisions'][j][1]].append(
                        data['frames'][self.frameid - i]['divisions'][j][3])
                except:
                    self.hierarchy[data['frames'][self.frameid - i]['divisions'][j][1]] = [
                        data['frames'][self.frameid - i]['divisions'][j][3]]

        self.max_id = np.max(list(self.cell_list.keys()))
        self.cell_width = cell_width
        self.flow = (0, 0)
        self.flow_vec = np.zeros((self.cell_number, 2))
        self.angle_chg = np.zeros(self.cell_number)
        self.moving_rate = (5, 2)
        self.angle_rate = (5 * np.pi / 180, np.pi / 180)
        self.seed = 0

    def draw(self):

        for cell in self.cell_list.values():
            tempimg = np.zeros((self.framewidth * self.framescale, self.frameheight * self.framescale))

            cv2.circle(self.image, (int(cell.xcenter + cell.length / 2 * np.cos(cell.angle)),
                                    self.frameheight * self.framescale -
                                    int(cell.ycenter + cell.length / 2 * np.sin(cell.angle))),
                       int(self.cell_width / 2 - 1), 1, cv2.FILLED)

            cv2.circle(self.labeled, (int(cell.xcenter + cell.length / 2 * np.cos(cell.angle)),
                                      self.frameheight * self.framescale -
                                      int(cell.ycenter + cell.length / 2 * np.sin(cell.angle))),
                       int(self.cell_width / 2 - 1), cell.original_id, cv2.FILLED)

            cv2.circle(self.layered[cell.idnumber, :, :],
                       (int(cell.xcenter + cell.length / 2 * np.cos(cell.angle)),
                        self.frameheight * self.framescale -
                        int(cell.ycenter + cell.length / 2 * np.sin(cell.angle))),
                       int(self.cell_width / 2 - 1), 1, cv2.FILLED)

            cv2.circle(tempimg, (int(cell.xcenter + cell.length / 2 * np.cos(cell.angle)),
                                 self.frameheight * self.framescale -
                                 int(cell.ycenter + cell.length / 2 * np.sin(cell.angle))),
                       int(self.cell_width / 2 - 1), 1, 1)

            cv2.circle(self.image, (int(cell.xcenter - cell.length / 2 * np.cos(cell.angle)),
                                    self.frameheight * self.framescale -
                                    int(cell.ycenter - cell.length / 2 * np.sin(cell.angle))),
                       int(self.cell_width / 2 - 1), 1, cv2.FILLED)

            cv2.circle(self.labeled, (int(cell.xcenter - cell.length / 2 * np.cos(cell.angle)),
                                      self.frameheight * self.framescale -
                                      int(cell.ycenter - cell.length / 2 * np.sin(cell.angle))),
                       int(self.cell_width / 2 - 1), cell.original_id, cv2.FILLED)

            cv2.circle(self.layered[cell.idnumber, :, :],
                       (int(cell.xcenter - cell.length / 2 * np.cos(cell.angle)),
                        self.frameheight * self.framescale -
                        int(cell.ycenter - cell.length / 2 * np.sin(cell.angle))),
                       int(self.cell_width / 2 - 1), 1, cv2.FILLED)

            cv2.circle(tempimg, (int(cell.xcenter - cell.length / 2 * np.cos(cell.angle)),
                                 self.frameheight * self.framescale -
                                 int(cell.ycenter - cell.length / 2 * np.sin(cell.angle))),
                       int(self.cell_width / 2 - 1), 2, 1)

            x1, y1 = np.where(tempimg == 1)
            x2, y2 = np.where(tempimg == 2)
            for i in range(len(x1)):
                try:
                    cv2.line(self.image, (y1[i], x1[i]), (y2[i], x2[i]),
                             1, 3, lineType=cv2.LINE_8)

                    cv2.line(self.labeled, (y1[i], x1[i]), (y2[i], x2[i]),
                             cell.original_id, 3, lineType=cv2.LINE_8)

                    cv2.line(self.layered[cell.idnumber, :, :], (y1[i], x1[i]), (y2[i], x2[i]),
                             1, 3, lineType=cv2.LINE_8)

                except Exception:
                    pass

    def make_image3(self):
        self.image3[self.image == 1] = [255, 255, 255]

    def update_move(self):
        """
        This function update the center and angle of the cells
        """
        for cell in self.cell_list:
            cell.xcenter += int(self.flow_vec[cell.idnumber, 0])
            cell.ycenter += int(self.flow_vec[cell.idnumber, 1])
            cell.angle += int(self.angle_chg[cell.idnumber])

    def move_maker(self):
        """
        This function make array of change for flow and angle change after that you can update the
        center and angle with update_move function.
        """
        np.random.seed(self.seed)
        self.flow_vec += np.random.normal(self.moving_rate[0], self.moving_rate[1], (self.cell_number, 2)) + self.flow
        self.angle_chg += np.random.normal(self.angle_rate[0], self.angle_rate[1], self.cell_number)

    def move(self):

        self.move_maker()
        self.update_move()
        self.image = np.zeros((self.framewidth * self.framescale, self.frameheight * self.framescale), dtype=np.uint8)
        self.labeled = np.zeros((self.framewidth * self.framescale, self.frameheight * self.framescale), dtype=np.uint8)
        self.layered = np.zeros((self.cell_number, self.framewidth * self.framescale,
                                 self.frameheight * self.framescale), dtype=np.uint8)
        self.draw()

    def color_cell(self, num, color_=(255, 0, 0)):
        self.image3[self.labeled == num] = color_

    def cell_splitter(self, cell_list):
        self.split_lab = self.labeled.copy()
        self.split_img = self.image.copy()
        for c in cell_list:
            cell = self.cell_list[c]
            self.split_lab[self.labeled == c] = 0
            self.split_img[self.labeled == c] = 0
            tempimg = np.zeros((self.framewidth * self.framescale, self.frameheight * self.framescale), dtype=np.uint16)

            cv2.circle(self.split_img, (int(cell.xcenter + cell.length / 2 * np.cos(cell.angle)),
                                        self.frameheight * self.framescale -
                                        int(cell.ycenter + cell.length / 2 * np.sin(cell.angle))),
                       int(self.cell_width / 2 - 1), 1, cv2.FILLED)

            cv2.circle(self.split_img, (int(cell.xcenter + self.cell_width * np.cos(cell.angle)),
                                        self.frameheight * self.framescale -
                                        int(cell.ycenter + self.cell_width * np.sin(cell.angle))),
                       int(self.cell_width / 2 - 1), 1, cv2.FILLED)

            cv2.circle(self.split_lab, (int(cell.xcenter + cell.length / 2 * np.cos(cell.angle)),
                                        self.frameheight * self.framescale -
                                        int(cell.ycenter + cell.length / 2 * np.sin(cell.angle))),
                       int(self.cell_width / 2 - 1), cell.original_id, cv2.FILLED)

            cv2.circle(self.split_lab, (int(cell.xcenter + self.cell_width * np.cos(cell.angle)),
                                        self.frameheight * self.framescale -
                                        int(cell.ycenter + self.cell_width * np.sin(cell.angle))),
                       int(self.cell_width / 2 - 1), cell.original_id, cv2.FILLED)

            cv2.circle(tempimg, (int(cell.xcenter + cell.length / 2 * np.cos(cell.angle)),
                                 self.frameheight * self.framescale -
                                 int(cell.ycenter + cell.length / 2 * np.sin(cell.angle))),
                       int(self.cell_width / 2 - 1), 1, 1)

            cv2.circle(tempimg, (int(cell.xcenter + self.cell_width * np.cos(cell.angle)),
                                 self.frameheight * self.framescale -
                                 int(cell.ycenter + self.cell_width * np.sin(cell.angle))),
                       int(self.cell_width / 2 - 1), 2, 1)

            x1, y1 = np.where(tempimg == 1)
            x2, y2 = np.where(tempimg == 2)
            for i in range(len(x1)):
                try:
                    cv2.line(self.split_img, (y1[i], x1[i]), (y2[i], x2[i]),
                             1, 3, lineType=cv2.LINE_8)

                    cv2.line(self.split_lab, (y1[i], x1[i]), (y2[i], x2[i]),
                             cell.original_id, 3, lineType=cv2.LINE_8)

                except Exception:
                    pass

            tempimg = np.zeros((self.framewidth * self.framescale, self.frameheight * self.framescale))

            cv2.circle(self.split_img, (int(cell.xcenter - cell.length / 2 * np.cos(cell.angle)),
                                        self.frameheight * self.framescale -
                                        int(cell.ycenter - cell.length / 2 * np.sin(cell.angle))),
                       int(self.cell_width / 2 - 1), 1, cv2.FILLED)

            cv2.circle(self.split_img, (int(cell.xcenter - self.cell_width / 2 * np.cos(cell.angle)),
                                        self.frameheight * self.framescale -
                                        int(cell.ycenter - self.cell_width / 2 * np.sin(cell.angle))),
                       int(self.cell_width / 2 - 1), 1, cv2.FILLED)

            cv2.circle(self.split_lab, (int(cell.xcenter - cell.length / 2 * np.cos(cell.angle)),
                                        self.frameheight * self.framescale -
                                        int(cell.ycenter - cell.length / 2 * np.sin(cell.angle))),
                       int(self.cell_width / 2 - 1), int(cell.original_id + self.max_id), cv2.FILLED)

            cv2.circle(self.split_lab, (int(cell.xcenter - self.cell_width * np.cos(cell.angle)),
                                        self.frameheight * self.framescale -
                                        int(cell.ycenter - self.cell_width * np.sin(cell.angle))),
                       int(self.cell_width / 2 - 1), int(cell.original_id + self.max_id), cv2.FILLED)

            cv2.circle(tempimg, (int(cell.xcenter - cell.length / 2 * np.cos(cell.angle)),
                                 self.frameheight * self.framescale -
                                 int(cell.ycenter - cell.length / 2 * np.sin(cell.angle))),
                       int(self.cell_width / 2 - 1), 1, 1)

            cv2.circle(tempimg, (int(cell.xcenter - self.cell_width * np.cos(cell.angle)),
                                 self.frameheight * self.framescale -
                                 int(cell.ycenter - self.cell_width * np.sin(cell.angle))),
                       int(self.cell_width / 2 - 1), 2, 1)

            x1, y1 = np.where(tempimg == 1)
            x2, y2 = np.where(tempimg == 2)
            for i in range(len(x1)):
                try:
                    cv2.line(self.split_img, (y1[i], x1[i]), (y2[i], x2[i]),
                             1, 3, lineType=cv2.LINE_8)

                    cv2.line(self.split_lab, (y1[i], x1[i]), (y2[i], x2[i]),
                             int(cell.original_id + self.max_id), 3, lineType=cv2.LINE_8)

                except Exception:
                    pass

    def cen_vec(self):
        """
        This function finds the center and two end point of cells from the self.image and put them in related
        instances for each cell.
        """
        for cn in self.cell_list.keys():
            cell_img = np.zeros(np.shape(self.image))
            cell_img[self.labeled == cn] = 1
            wh = np.where(cell_img == 1)
            pixs = np.array((wh[0] - np.mean(wh[0]), wh[1] - np.mean(wh[1])))
            ev, eigenvectors = np.linalg.eig(np.matmul(pixs, pixs.T))
            temp_cent = (np.int(np.mean(wh[0])), np.int(np.mean(wh[1])))
            self.cell_list[cn].end1, self.cell_list[cn].end2 = end_finder(temp_cent,
                                                                          cell_img,
                                                                          eigenvectors[:, np.argmax(ev)])

            self.cell_list[cn].img_center = (np.int(np.mean(wh[1])), np.int(np.mean(wh[0])))

            self.cell_list[cn].img_length = np.linalg.norm(np.array(self.cell_list[cn].end2) -
                                                           np.array(self.cell_list[cn].end1))


class BM:
    """
    This object would get two frames and fine the parents and children for those two frames. Besides that
    we would have all information for the cost function too.
    """

    def __init__(self, fr1, fr0, lam1, lam2, lam3, lam4, epoch_num=5000, temperature=1000, decrease_ratio=.995, eps=.005):
        # first and second frame that we want to find the split which they are from class of Frame
        self.frame1 = fr1
        self.frame0 = fr0
        # lambda one to four coefficients of the cost function
        self.lambda1 = lam1
        self.lambda2 = lam2
        self.lambda3 = lam3
        self.lambda4 = lam4
        self.lambda_par1 = 5
        self.lambda_par2 = 100
        self.lambda_par3 = .01
        # number of the epoch
        self.number_of_epoch = epoch_num
        # temperature
        self.t = temperature
        # ratio to change the temprature
        self.dt = decrease_ratio
        # epsilon value to stop the process
        self.epsilon = eps
        # number of cells that has been split
        self.split_num = self.frame1.cell_number - self.frame0.cell_number
        # number of small children to consider for possible cases
        # self.small_number = 4 * self.split_num + 10
        self.small_number = max(int(self.frame1.cell_number * 2/ 3), 4 * self.split_num + 15)
        # number of pairs
        # if self.split_num * 4 + 4 < 40:
        #     self.number_of_pairs = int(4 * self.split_num + 4)
        # elif self.split_num * 3 + 4 < 40:
        #     self.number_of_pairs = int(3 * self.split_num + 4)
        # else:
        #     self.number_of_pairs = int(2 * self.split_num + 4)
        self.number_of_pairs = self.split_num

        # cost of the pairs for each step
        self.cost = []
        # all cells that may be one of the children
        self.possible_cases = []
        # first coordinate and second coordinate of self.pairs
        self.pairs_col1 = []
        self.pairs_col2 = []
        # the target pairs that we want to find
        self.pairs = []
        # all parameters that we want to store them
        self.overlaps = []
        self.ratio_cost = []
        self.distance_cost = []
        self.length_ratio_cost = []
        self.epoch_cost = []
        self.new_cost = []
        self.len_cost = []
        self.par_cost = []
        # dictionary of the choice to match for each cell
        self.choices = {}
        # dictionary of parents of children
        self.possible_parents = {}
        self.pairs_with_parent = {}
        # this is the set of all possible pairs
        self.all_pairs = []
        self.pairs_temp = []
        # this is the radius that we won't accept any cell with the center of the base cell farther than this
        self.neighbor_radius = 80
        # this variable count the number of times initial value reapeat and if it is
        # more than 5 we stop the process.
        self.while_counter = 0
        # these two are the neighbors for frame 0 and 1
        self.neigh0 = {}
        self.neigh1 = {}
        # length of the minimum cell
        self.minimum_length = min([np.linalg.norm(np.array(self.frame1.cell_list[y].end1) -
                                                  np.array(self.frame1.cell_list[y].end2))
                                   for y in self.frame1.cell_list.keys()])
        # threshold for the parameters
        self.deviate_threshold = .2
        self.distance_threshold = 30
        self.ratio_threshold = .2
        self.length_threshold = 250
        self.look_parents = 25

    def number_of_smalls(self):
        pass

    def possible_maker(self):
        """
        This function finds label of all small_number smallest cells
        :return:
        """
        s_cases = small_cell_finder(self.frame1, self.small_number)
        for case in s_cases:
            self.possible_cases.append(case[0])

    def choice_maker(self):
        """
        This function make the dictionary of choices. We do not consider neighbors here
        :return:
        """
        for c in self.possible_cases:
            self.choices[c] = []
            p_cases = self.possible_cases.copy()
            p_cases.remove(c)
            for pc in p_cases:
                if np.linalg.norm(np.array(self.frame1.cell_list[c].img_center) -
                                  np.array(self.frame1.cell_list[pc].img_center)) < self.neighbor_radius:
                    self.choices[c].append(pc)

    def choice_maker_with_neighbor(self):
        """
        This function make the dictionary of choices. We consider neighbors
        in this function.
        :return:
        """
        self.neighbor_maker()
        for c in self.possible_cases:
            self.choices[c] = []
            for neigh in self.neigh1[c]:
                if neigh in self.possible_cases:
                    self.choices[c].append(neigh)

    def all_pairs_maker(self, neighbor=True):
        """
        This function make all possible pairs.
        :param neighbor: if we want to consider neighbors as restriction for choice_maker
        it should be True. Otherwise for distance we have to change it to False
        :return:
        """
        if neighbor:
            self.choice_maker_with_neighbor()
        else:
            self.choice_maker()
        for key, val in self.choices.items():
            for cell in val:
                if key < cell:
                    penalties = self.pair_values(key, cell)
                    len1 = (self.minimum_length - np.linalg.norm(np.array(self.frame1.cell_list[key].end1) -
                                                                 np.array(self.frame1.cell_list[key].end2))) ** 2
                    len2 = (self.minimum_length - np.linalg.norm(np.array(self.frame1.cell_list[cell].end1) -
                                                                 np.array(self.frame1.cell_list[cell].end2))) ** 2
                    if len1 + len2 < self.length_threshold and penalties[0] < self.deviate_threshold and \
                            penalties[1] < self.deviate_threshold and penalties[2] < self.distance_threshold and \
                            penalties[3] < self.ratio_threshold:

                        self.possible_parents[(key, cell)] = self.which_parent(key, cell)
                        if self.possible_parents[(key, cell)]:
                            self.all_pairs.append((key, cell))

    def which_parent(self, c1, c2):
        mid_point = (np.array(self.frame1.cell_list[c1].img_center) +
                     np.array(self.frame1.cell_list[c2].img_center)) / 2

        cells = []

        for c in self.frame0.cell_list:
            if np.linalg.norm(mid_point - np.array(self.frame0.cell_list[c].img_center)) < self.look_parents:
                cells.append((c, self.parent_cost(c1, c2, c, (self.lambda_par1, self.lambda_par2, self.lambda_par3))))

        return sorted(cells, key=lambda x: x[1])[0][0] if cells else []

    def parent_cost(self, c1, c2, p, lam=None):
        """
        This function finds the cost values for relation between parents and children
        :param c1: integer - cell number for one of children in I_1
        :param c2: integer - cell number for another children in I_1
        :param p: integer - cell number for possible parent of children in I_0
        :param lam: triple of float - coefficient values for cost parameters
        :return: if lam=None returns all parameters separately, else multiply lam with parameters and add them
        """
        mid_point = (np.array(self.frame1.cell_list[c1].img_center) +
                     np.array(self.frame1.cell_list[c2].img_center)) / 2

        center_distance = np.linalg.norm(mid_point - np.array(self.frame0.cell_list[p].img_center))

        middle_vec = np.array(self.frame1.cell_list[c1].img_center) - np.array(self.frame1.cell_list[c2].img_center)
        c1_vec = np.array(self.frame1.cell_list[c1].img_center) - np.array(self.frame1.cell_list[c2].end1)
        c2_vec = np.array(self.frame1.cell_list[c2].img_center) - np.array(self.frame1.cell_list[c2].end2)
        par_vec = np.array(self.frame0.cell_list[p].end2) - np.array(self.frame0.cell_list[p].end1)

        angle_cost = np.abs(np.cross(par_vec, c1_vec)/(np.linalg.norm(par_vec) * np.linalg.norm(c1_vec))) +\
                     np.abs(np.cross(par_vec, c2_vec)/(np.linalg.norm(par_vec) * np.linalg.norm(c2_vec))) +\
                     np.abs(np.cross(middle_vec, par_vec)/(np.linalg.norm(middle_vec) * np.linalg.norm(par_vec)))

        length_cost = np.abs(np.linalg.norm(middle_vec) + np.linalg.norm(c1_vec) + np.linalg.norm(c2_vec) -
                       np.linalg.norm(par_vec))

        if not lam:
            return center_distance, angle_cost, length_cost
        else:
            return lam[0] * center_distance + lam[1] * angle_cost + lam[2] * length_cost

    def initialize_without_neighbor(self):
        """
        This function initialize values for the boltzmann machine. First make possible
        choices. After that make a random pairs
        :return:
        """
        self.possible_maker()
        self.all_pairs_maker(False)
        for _ in range(self.number_of_pairs):
            p_case = self.all_pairs.copy()
            for pair in self.pairs:
                p_case.remove(pair)
            if len(p_case) == 0:
                # print('this file has emptied p_case', self.frame1.file_name)
                break
            temp_idx = np.random.choice(len(p_case))
            self.pairs.append(p_case[temp_idx])
            self.pairs_col1.append(p_case[temp_idx][0])
            self.pairs_col2.append(p_case[temp_idx][1])

    def initialize_with_neighbor(self):
        """
        This function initialize values for the boltzmann machine. First make possible
        choices. After that make a random pairs
        :return:
        """
        self.possible_maker()
        self.all_pairs_maker()
        for _ in range(self.number_of_pairs):
            p_case = self.all_pairs.copy()
            for pair in self.pairs:
                p_case.remove(pair)
            temp_idx = np.random.choice(len(p_case))
            self.pairs.append(p_case[temp_idx])
            self.pairs_col1.append(p_case[temp_idx][0])
            self.pairs_col2.append(p_case[temp_idx][1])
            # If we had overlap self.accepted_pair would be False so we have to change
            # the choice
            flag = False
            while not self.accepted_pair(False):
                del self.pairs[-1]
                del self.pairs_col1[-1]
                del self.pairs_col2[-1]
                del p_case[temp_idx]

                if len(p_case) == 0:
                    self.while_counter += 1
                    if self.while_counter > 4:
                        print('it is increased more than 5:', self.frame0.file_name)
                    self.pairs = []
                    self.initialize_with_neighbor()
                    flag = True
                    break

                temp_idx = np.random.choice(len(p_case))
                self.pairs.append(p_case[temp_idx])
                self.pairs_col1.append(p_case[temp_idx][0])
                self.pairs_col2.append(p_case[temp_idx][1])
            if flag:
                break

    def pair_values(self, c1, c2):
        """
        This function compute some cost values for two cells c1 and c2
        :param c1: int - cell number one
        :param c2: int - cell number two
        :return: deviate penalty c1, deviate penalty c2, distance penalty two cells, ratio penalty two cells
        """
        possible_cases = np.array(list(itertools.product((self.frame1.cell_list[c1].end1,
                                                          self.frame1.cell_list[c1].end2),
                                                         (self.frame1.cell_list[c2].end1,
                                                          self.frame1.cell_list[c2].end2))))
        dis_list = []
        for i in range(len(possible_cases)):
            dis_list.append(np.linalg.norm(possible_cases[i][0] - possible_cases[i][1]))

        low_dis = possible_cases[np.argmin(dis_list)]

        dis0 = np.cross(np.array(self.frame1.cell_list[c1].img_center) - np.array(self.frame1.cell_list[c2].img_center),
                        np.array(self.frame1.cell_list[c2].img_center) - np.array(low_dis[0])) / \
               np.linalg.norm(np.array(self.frame1.cell_list[c1].img_center) -
                              np.array(self.frame1.cell_list[c2].img_center)) ** 2

        dis1 = np.cross(np.array(self.frame1.cell_list[c1].img_center) - np.array(self.frame1.cell_list[c2].img_center),
                        np.array(self.frame1.cell_list[c2].img_center) - np.array(low_dis[1])) / \
               np.linalg.norm(np.array(self.frame1.cell_list[c1].img_center) -
                              np.array(self.frame1.cell_list[c2].img_center)) ** 2

        length_ratio = abs(2 - (np.linalg.norm(np.array(self.frame1.cell_list[c1].end1) -
                                               np.array(self.frame1.cell_list[c1].end2)) /
                                np.linalg.norm(np.array(self.frame1.cell_list[c2].end1) -
                                               np.array(self.frame1.cell_list[c2].end2))) -
                           (np.linalg.norm(np.array(self.frame1.cell_list[c2].end1) -
                                           np.array(self.frame1.cell_list[c2].end2)) /
                            np.linalg.norm(np.array(self.frame1.cell_list[c1].end1) -
                                           np.array(self.frame1.cell_list[c1].end2))))

        return abs(dis0), abs(dis1), np.linalg.norm(np.array(low_dis[0]) - np.array(low_dis[1])), length_ratio

    def length_cost(self):
        return sum([(self.minimum_length - np.linalg.norm(np.array(self.frame1.cell_list[y].end1) -
                                                          np.array(self.frame1.cell_list[y].end2)))**2
                    for y in self.pairs_col2 + self.pairs_col1]) / (2 * len(self.pairs))

    def total_parent_cost(self):
        return np.sum(list(map(lambda x, y: self.parent_cost(x, y, self.possible_parents[(x, y)],
                                                             (self.lambda_par1, self.lambda_par2,
                                                              self.lambda_par3)),
                               self.pairs_col1, self.pairs_col2)))

    def compute_cost(self):
        """
        This function compute the cost w,r,t lambdas and other parameters in self
        :return: float/ cost value
        """
        overlap_cost = hmn_overlap(self.pairs_temp)
        len_cost = self.length_cost()
        pairs_cost = np.array(list(map(self.pair_values, self.pairs_col1, self.pairs_col2)))
        par_cost = self.total_parent_cost()

        return np.sum(pairs_cost[:, 0:2]) + self.lambda1 * np.sum(pairs_cost[:, 2]) + \
               self.lambda2 * np.sum(pairs_cost[:, 3]) + self.lambda3 * overlap_cost + \
               self.lambda4 * len_cost + 5 * par_cost

    def change(self, pair_num, col_num, new_cell):
        if col_num == 0:
            self.pairs_col1[pair_num] = new_cell
        else:
            self.pairs_col2[pair_num] = new_cell

    def switch_pairs(self, id, tp):
        """
        This function switch pair tp in the self.pairs_temp and self.col1 and two
        :param id: integer / index of the pair in self.pairs_temp
        :param tp: pair [c_1, c_2]/ that we wanna switch in self.pairs
        :return: switch in pairs_temp, self.col1 and self.col2
        """
        del self.pairs_temp[id]
        self.pairs_temp.insert(id, tp)
        del self.pairs_col1[id]
        self.pairs_col1.insert(id, tp[0])
        del self.pairs_col2[id]
        self.pairs_col2.insert(id, tp[1])

    def find_neighbors(self, pindex, triang):
        # This function will find neighbors of point pindex from
        # triang and return index of neighbors
        return triang.vertex_neighbor_vertices[1][
               triang.vertex_neighbor_vertices[0][pindex]:
               triang.vertex_neighbor_vertices[0][pindex + 1]]

    def neigh_without_cell_cut(self, pind, pset, lis_lab, triangle, lab):
        """
        we find neighbors of cell number pind with Delaunay triangulation and if there is any overlap between the line
        from two neighbors and another cell we remove that neighbor as a candidate for neighbors.
        :param pind: index of cell integer
        :param pset: set of all centroid
        :param triangle: Delaunay triangulation between the cells
        :param lab: labeled image for cell numbers
        :return: neighbors with nan and without nan
        """
        neig = self.find_neighbors(pind, triangle)

        neigh_set = [[x, lis_lab[x]] for x in neig]
        for neighbor in neig:
            if np.linalg.norm((pset[pind] - pset[neighbor])) > self.neighbor_radius:
                neigh_set.remove([neighbor, lis_lab[neighbor]])

            else:
                line_pixel_set = list(lab[tuple(zip(line(int(pset[pind][1]), int(pset[pind][0]),
                                                         int(pset[neighbor][1]), int(pset[neighbor][0]))))][0])

                if {0, lis_lab[pind], lis_lab[neighbor]} != (set(np.unique(line_pixel_set)) | {0}):
                    neigh_set.remove([neighbor, lis_lab[neighbor]])

        return [x[1] for x in neigh_set]

    def neighbor_maker(self):
        """
        This function make two dictionary of neigh0 and neigh1 for the neighbors of the
        cells in frame 0 and 1
        :return: neighbors in self.neigh0 and self.neigh1
        """
        cent0 = np.array([cell.img_center for cell in self.frame0.cell_list.values()])
        label_listed0 = np.array([num for num in self.frame0.cell_list.keys()])
        cent1 = np.array([cell.img_center for cell in self.frame1.cell_list.values()])
        label_listed1 = np.array([num for num in self.frame1.cell_list.keys()])

        tri0 = Delaunay(cent0)
        tri1 = Delaunay(cent1)

        for idx, key in enumerate(self.frame0.cell_list.keys()):
            self.neigh0[key] = self.neigh_without_cell_cut(idx, cent0, label_listed0,
                                                           tri0, self.frame0.labeled)
        for idx, key in enumerate(self.frame1.cell_list.keys()):
            self.neigh1[key] = self.neigh_without_cell_cut(idx, cent1, label_listed1,
                                                           tri1, self.frame1.labeled)

    def execute_first_bm(self):
        if self.split_num == 0:
            return
        self.initialize_without_neighbor()
        self.cost.append(self.compute_cost())
        self.epoch_cost.append(self.cost[0])
        self.pairs_temp = self.pairs.copy()
        for _ in range(self.number_of_epoch):
            self.overlaps.append(hmn_overlap(self.pairs))
            pairs_cost = np.array(list(map(self.pair_values, self.pairs_col1, self.pairs_col2)))
            self.ratio_cost.append(np.sum(pairs_cost[:, 0:2]))
            self.distance_cost.append(np.sum(pairs_cost[:, 2]))
            self.length_ratio_cost.append(np.sum(pairs_cost[:, 3]))
            self.len_cost.append(self.length_cost())
            for idx, pair in enumerate(self.pairs):

                p_case = self.all_pairs.copy()
                for p in self.pairs:
                    p_case.remove(p)
                for p in self.pairs_temp:
                    try:
                        p_case.remove(p)
                    except Exception:
                        pass

                if len(p_case) == 0:
                    break

                temp_pair_idx = np.random.choice(len(p_case))
                self.switch_pairs(idx, p_case[temp_pair_idx])

                new_cost = self.compute_cost()
                self.new_cost.append(new_cost)

                delta_cost = new_cost - self.cost[-1]
                if delta_cost < 0:
                    self.cost.append(new_cost)

                else:
                    u = np.random.uniform()
                    if u < np.exp(-delta_cost / self.t):
                        self.cost.append(new_cost)
                    else:
                        self.cost.append(self.cost[-1])
                        self.switch_pairs(idx, pair)

            self.pairs = self.pairs_temp.copy()
            self.epoch_cost.append(self.cost[-1])
            self.t *= self.dt
            if len(self.epoch_cost) > 100 and \
                    np.sum(np.abs(np.diff(self.epoch_cost[-30:]))) < self.epsilon * self.epoch_cost[-100]:
                break

        self.parent_finder()

    def accepted_pair(self, temp=True):
        if temp:
            pair = self.pairs_temp
        else:
            pair = self.pairs
        if hmn_overlap(pair) > 0:
            return False
        else:
            return True

    def execute_with_neighbor_bm_match_finder(self):
        self.initialize_with_neighbor()
        self.cost.append(self.compute_cost())
        self.epoch_cost.append(self.cost[0])
        self.pairs_temp = self.pairs.copy()
        for _ in tqdm(range(self.number_of_epoch)):
            self.overlaps.append(hmn_overlap(self.pairs))
            pairs_cost = np.array(list(map(self.pair_values, self.pairs_col1, self.pairs_col2)))
            self.ratio_cost.append(np.sum(pairs_cost[:, 0:2]))
            self.distance_cost.append(np.sum(pairs_cost[:, 2]))
            self.length_ratio_cost.append(np.sum(pairs_cost[:, 3]))
            self.par_cost.append(self.total_parent_cost())
            self.len_cost.append(self.length_cost())
            for idx, pair in enumerate(self.pairs):
                second_pair_list = list(range(len(self.pairs)))
                second_pair_list.remove(idx)
                idx_second = np.random.choice(second_pair_list)
                second_pair = self.pairs[idx_second]
                p_case = self.all_pairs.copy()
                for p in self.pairs:
                    p_case.remove(p)
                for p in self.pairs_temp:
                    try:
                        p_case.remove(p)
                    except Exception:
                        pass

                if len(p_case) == 0:
                    break

                temp_pair_idx = np.random.choice(len(p_case))
                self.switch_pairs(idx, p_case[temp_pair_idx])
                # here we put two big numbers max+1 and max+2 for second pair
                max_num = np.max(self.possible_cases)
                self.switch_pairs(idx_second, [max_num + 1, max_num + 2])
                flag = False
                while not self.accepted_pair(True):
                    del p_case[temp_pair_idx]
                    if len(p_case) == 0:
                        self.switch_pairs(idx, pair)
                        self.switch_pairs(idx_second, second_pair)
                        flag = True
                        break
                    temp_pair_idx = np.random.choice(len(p_case))
                    self.switch_pairs(idx, p_case[temp_pair_idx])
                # here we find the second pair the same as the first one
                if len(p_case) == 0:
                    self.switch_pairs(idx_second, second_pair)
                    self.switch_pairs(idx, pair)
                    break

                del p_case[temp_pair_idx]

                temp_pair_idx = np.random.choice(len(p_case))
                self.switch_pairs(idx_second, p_case[temp_pair_idx])
                while not self.accepted_pair(True):
                    del p_case[temp_pair_idx]
                    if len(p_case) == 0:
                        self.switch_pairs(idx, pair)
                        self.switch_pairs(idx_second, second_pair)
                        flag = True
                        break
                    temp_pair_idx = np.random.choice(len(p_case))
                    self.switch_pairs(idx_second, p_case[temp_pair_idx])
                # if we the length of the p_case is zero we jumped out and we leave the loop
                if flag:
                    break

                new_cost = self.compute_cost()
                self.new_cost.append(new_cost)

                delta_cost = new_cost - self.cost[-1]
                if delta_cost < 0:
                    self.cost.append(new_cost)

                else:
                    u = np.random.uniform()
                    if u < np.exp(-delta_cost / self.t):
                        self.cost.append(new_cost)
                    else:
                        self.cost.append(self.cost[-1])
                        self.switch_pairs(idx, pair)
                        self.switch_pairs(idx_second, second_pair)
            self.pairs = self.pairs_temp.copy()
            self.epoch_cost.append(self.cost[-1])
            self.t *= self.dt
            if len(self.epoch_cost) > 100 and \
                    np.sum(np.abs(np.diff(self.epoch_cost[-30:]))) < self.epsilon * self.epoch_cost[-100]:
                break

    def children_center(self):
        """
        This function finds the center of each pair is list of pairs
        :return: list of tuples - centers of pairs
        """
        centers = []
        for pair in self.pairs:
            centers.append((np.array(self.frame1.cell_list[pair[0]].img_center) +
                            np.array(self.frame1.cell_list[pair[1]].img_center)) / 2)

        return centers

    def parent_finder(self):
        for pair in self.pairs:
            self.pairs_with_parent[self.possible_parents[pair]] = pair

    # def major_vote(self, hmn):
    #     votes = []
    #     for _ in range(hmn):

    def color_pairs(self):
        colors = prb.static_generate_colors(self.number_of_pairs)
        for idx, pair in enumerate(self.pairs):
            self.frame1.image3[self.frame1.labeled == pair[0]] = colors[idx]
            self.frame1.image3[self.frame1.labeled == pair[1]] = colors[idx]


def full_path_json(json_f):
    with open(json_f) as f:
        json_file = json.load(f)

    fr354 = Frame(json_file, 354)
    fr354.draw()
    fr360 = Frame(json_file, 360)
    fr360.draw()

    truemap = {}
    for i in range(1, 1 + fr354.cell_number):
        truemap[i] = [i]

    one_res = prb.complete_bm(100000, 100000, 100000000,
                              fr354.image, fr360.image,
                              fr354.labeled, fr360.labeled,
                              truemap, rel_window=55, g_rate=1.3,
                              epoch=2000, T=1000, dT=.99, epsilon=.005,
                              nei_win=100, draw_=False)
    return one_res


def true_map_split(i1_fr):
    """
    This function makes true_map for accuracy for I_1.
    :param i1_fr: output of class frame for I_1.
    :return: dictionary of true mapping.
    """
    truemap = {}
    for key in i1_fr.cell_list.keys():
        truemap[key] = [i1_fr.cell_list[key].parent]

    return truemap


def label_img(img_lab, img_bin, color_=(0, 0, 255), lab=None):
    if lab is None:
        labeled_img = label(img_bin, connectivity=1)
    else:
        labeled_img = lab

    # print("Number of cells: ", np.max(labeled_img) + 1)
    for i in np.unique(labeled_img):
        if i == 0:
            continue
        cord_ = np.where(labeled_img == i)
        y_center = int(np.sum(cord_[0]) / len(cord_[0])) + 1
        x_center = int(np.sum(cord_[1]) / len(cord_[1])) - 1
        cv2.putText(img_lab, str(i), (x_center, y_center),
                    cv2.FONT_HERSHEY_SIMPLEX, 5 / 10, color_, 1)
    return img_lab


# def cell_splitter(img, ilab, cell_list, cell_width):
#     im = img.copy()
#     centr0, vector0 = vf2.cen_vec(im, save_name="vector0", lab_img=ilab)
#     for cell in cell_list:
#         im[ilab == cell] = 0
#         ilab[ilab == cell] = 0


def hmn_overlap(pair):
    _, occurrence = np.unique(pair, return_counts=True)
    return sum(occurrence) - len(occurrence)


def labtolab(img, ilab):
    transfer_dic = {}
    transfer_dic_reverse = {}
    labeled_img = label(img, connectivity=1)
    for cell in range(1, np.max(labeled_img) + 1):
        x = np.where(labeled_img == cell)[0][0]
        y = np.where(labeled_img == cell)[1][0]
        transfer_dic[cell] = ilab[x, y]
        transfer_dic_reverse[ilab[x, y]] = cell

    return labeled_img, transfer_dic, transfer_dic_reverse


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
    n0 = len(true_dic)
    for nu in range(1, n0 + 1):
        if transfer0[nu] == true_dic[transfer1[pred_dic[nu][0]]]:
            truenum += 1
        elif transfer0[nu] > i0_thresh and transfer0[nu] - i0_thresh == true_dic[transfer1[pred_dic[nu][0]]]:
            truenum += 1
        else:
            wrong_list.append(nu)

    return truenum / n0, wrong_list


def long_cell_finder(frame, hmn):
    """
    This function looks in the frame and find the 'hmn' of longest cells in the frame.
    :param frame: one of the frame of the movie.
    :param hmn: how many long cell you want. integer
    :return: returns list of 'hmn' long cells in the list.
    """
    cell_length_list = [(x, frame.cell_list[x].img_length) for x in frame.cell_list.keys()]

    length_sorted = sorted(cell_length_list, reverse=True, key=itemgetter(1))

    return length_sorted[:hmn]


def small_cell_finder(frame, hmn):
    """
    This function looks in the frame and find the 'hmn' of shortes cells in the frame.
    :param frame: one of the frame of the movie.
    :param hmn: how many long cell you want. integer
    :return: returns list of 'hmn' long cells in the list.
    """
    cell_length_list = [(x, frame.cell_list[x].img_length) for x in frame.cell_list.keys()]

    length_sorted = sorted(cell_length_list, key=itemgetter(1))

    return length_sorted[:hmn]


def color_some_cell(lab, lis, colors=None):
    img = np.zeros((np.shape(lab)[0], np.shape(lab)[1], 3))
    img[lab > 0] = [255, 255, 255]
    if colors:
        for idx, li in enumerate(lis):
            img[lab == li] = colors
    else:
        colors = prb.static_generate_colors(len(lis))
        for idx, li in enumerate(lis):
            img[lab == li] = colors[idx]

    return img


def pair_accuracy(bol, frame):
    """
    This function finds the the ratio of correct children finds in set of pairs to actual children from
    boltzmann machine algorithm.
    :param bol: object of children finder after boltzmann machine optimization
    :param frame: frame that we want to find the children. we can use bol.frame1 too
    :return: ratio of correct children in set of pairs to actual children
    """
    s = 0
    for key, val in frame.hierarchy.items():
        if (key, val[0]) in bol.pairs or (val[0], key) in bol.pairs:
            s += 1
    return s / len(frame.hierarchy)


def parent_accuracy(bol):
    """
    This function finds the accuracy of parents
    :param bol: output of boltzmann machine algorithm to find children
    :return: ratio of children to pairs found in pairs_with_parent, and actual children
    """
    s = 0
    for key, val in bol.frame1.hierarchy.items():
        if key in bol.pairs_with_parent:
            if key in bol.pairs_with_parent[key] and val[0] in bol.pairs_with_parent[key]:
                s += 1

    return 0 if len(bol.pairs_with_parent) == 0 else s / len(bol.pairs_with_parent), \
           0 if len(bol.frame1.hierarchy) == 0 else s/len(bol.frame1.hierarchy)


def real_pair_accuracy(pair, frame):
    s = 0
    for key, val in frame.hierarchy.items():
        if [key, val[0]] in pair or [val[0], key] in pair:
            s += 1
    return s / len(frame.hierarchy)

# with open('Images/james_sample/with_split/ws.json') as f:
#     json_file = json.load(f)
# # fr354 = Frame(json_file, 354)
# # fr360 = Frame(json_file, 360)
#
# fr = []
# for i in range(2):
#     fr.append(Frame(json_file, 354 + i, hierarchy_lookback=0))
#     fr[i].draw()
#     fr[i].cen_vec()


# plt.imshow(label_img(ms.for_show(fr[1].image), fr[1].image, lab=fr[1].labeled))
# plt.show()
#
# # fr[4].cell_splitter(list(fr[5].hierarchy.keys()))
# # fr[4].cell_splitter(long_cell_finder(fr[4], fr[5].cell_number - fr[4].cell_number)[0])
# fr[4].cell_splitter([1340, 1334, 1260, 1324, 1326, 1328, 1323])
# plt.imshow(ms.for_show(fr[0].image))
# plt.show()
# plt.imshow(ms.for_show(fr[1].image))
# plt.show()
# plt.imshow(ms.for_show(fr[1].split_img))
# plt.show()
#
# trmp = true_map_split(fr[5])
#
# lab0, tr0, rtr0 = labtolab(fr[4].split_img, fr[4].split_lab)
# lab1, tr1, trt1 = labtolab(fr[5].image, fr[5].labeled)
def bm_func1(l=(.01, .01, .01, .01)):
    """
    this function return the accuracy of bm for given lambda for file ws
    :param l: lambdas for boltzmann machine
    :return: accuracy of the matching
    """
    with open('Images/james_sample/with_split/ws.json') as f:
        json_file = json.load(f)

    fr = []
    for i in range(2):
        fr.append(Frame(json_file, 354 + i, 'Images/james_sample/with_split/ws.json', hierarchy_lookback=0))
        fr[i].draw()
        fr[i].cen_vec()
    acc = []
    for i in range(7):
        bm = BM(fr[1], fr[0], l[0], l[1], l[2], l[3])
        bm.execute_first_bm()
        acc.append(pair_accuracy(bm, fr[1]))
    acc.extend(l)
    return acc


def bm_func(fi, l=(.01, .0001, 500, .05)):
    """
    This function finds the accuracy of the matching pairs for all 5 pairs of images for frame in address
    'fi'. so we find 7 accuracy for each 5 pairs of images.
    :param fi: address of the frames
    :param l: lambdas for boltzmann machine
    :return: 7 by 5 accuracy of the matching of pairs of cell
    """
    with open(fi) as f:
        json_file = json.load(f)
    # fr354 = Frame(json_file, 354)
    # fr360 = Frame(json_file, 360)

    fr = []
    for i in range(6):
        fr.append(Frame(json_file, 354 + i, fi, hierarchy_lookback=0))
        fr[i].draw()
        fr[i].cen_vec()
    total_acc = []
    total_par0 = []
    total_par1 = []
    for j in range(1, 6):
        acc = []
        par0 = []
        par1 = []
        for i in range(7):
            bolt = BM(fr[j], fr[j-1], l[0], l[1], l[2], l[3])
            if len(fr[j].hierarchy) == 0:
                acc.append(np.inf)
                par0.append(np.inf)
                par1.append(np.inf)
            else:
                bolt.execute_first_bm()
                # bol.color_pairs()
                acc.append(pair_accuracy(bolt, fr[j]))
                par = parent_accuracy(bolt)
                par0.append(par[0])
                par1.append(par[1])

        total_acc.append(acc)
        total_par0.append(par0)
        total_par1.append(par1)
    return total_acc, total_par0, total_par1, bolt.frame1.file_name


def bm_func_vote(fi, l=(.01, .00001, .01, .001)):
    """
    This function gets the frames and finds the boltzmann machine for each one of 5 pairs find the
    result of boltzmann machine n = 7 times and returns the voting for each pairs of images
    :param fi: address of the json file (output of the result of simulation of James Winkle's algorithm
    :param l: lambdas for boltzmnn machine lagorithm
    :return: list of accuracy of voting for 5 pairs of images
    """
    with open(fi) as f:
        json_file = json.load(f)
    # fr354 = Frame(json_file, 354)
    # fr360 = Frame(json_file, 360)

    fr = []
    for i in range(6):
        fr.append(Frame(json_file, 354 + i, fi, hierarchy_lookback=0))
        fr[i].draw()
        fr[i].cen_vec()
    total_acc = []
    for j in tqdm(range(1, 6)):
        acc = []
        for i in range(7):
            acc = []
            bm = BM(fr[j], fr[j - 1], l[0], l[1], l[2], l[3])
            pairs = []
            if len(fr[j].hierarchy) == 0:
                acc.append(np.inf)
            else:
                for i in range(7):
                    bm.execute_first_bm()
                    pairs += bm.pairs
                pair_tuple = map(tuple, pairs)
                pair_counter = Counter(pair_tuple)
                pair_final = sorted(pair_counter.items(), key=lambda x: x[1], reverse=True)
                pair_final = [list(x[0]) for x in pair_final[:bm.number_of_pairs]]
                acc.append(real_pair_accuracy(pair_final, fr[j]))
        total_acc.append(acc)

    return total_acc


def cost_dist_compare(bol, al=True):
    actual_pairs = []
    for key, val in bol.frame1.hierarchy.items():
        actual_pairs.append([key, val[0]])
    if al:
        pairs = bol.all_pairs.copy()
        for pair in actual_pairs:
            try:
                pairs.remove(pair)
            except Exception:
                pass
    else:
        pairs = actual_pairs.copy()

    costs = []
    for pair in pairs:
        c = list(bol.pair_values(pair[0], pair[1]))
        c.append(sum([(bol.minimum_length - np.linalg.norm(np.array(bol.frame1.cell_list[y].end1) -
                                                           np.array(bol.frame1.cell_list[y].end2)))**2
                      for y in pair]))
        costs.append(c)

    return costs


def video_maker_teenager(frames, address, teens=3, cut=False, label_=False, start=0, stop=390):
    """
    This function makes all frames of json file 'frames' and save it in 'address' but it colors children in
    blue and teenager for 'teens' frame before in red
    :param frames: address of the json file for video
    :param address: address to save frames
    :param teens: number of frames that we want to consider teenager children
    :return: save images in 'address' folder
    """
    children = {}
    with open(frames) as f:
        j_file = json.load(f)

    for idx in tqdm(range(start, min(stop, len(j_file['frames'])))):
        frame = Frame(j_file, idx, frames, hierarchy_lookback=0)
        children[idx] = list(frame.hierarchy.keys())
        children[idx].extend([x[0] for x in frame.hierarchy.values()])

        frame.draw()
        frame.cen_vec()
        frame.make_image3()

        if cut:
            for rdx in frame.cell_list:
                if np.linalg.norm(np.array(frame.cell_list[rdx].end1) -
                                  np.array(frame.cell_list[rdx].end2)) < 56:
                    frame.image3[frame.labeled == rdx] = [0, 0, 0]

        if idx > teens + 1:
            for jdx in range(1, teens):
                for x in children[idx - jdx]:
                    frame.image3[frame.labeled == x] = [255, 0, 0]

        for x in children[idx]:
            frame.image3[frame.labeled == x] = [0, 0, 255]
        if label_:
            frame.image3 = label_img(frame.image3, frame.image, color_=(0, 0, 255), lab=frame.labeled)

        cv2.imwrite(address + '{}.png'.format(idx), frame.image3)


def bm_func_parent(fi):
    with open(fi) as f:
        json_file = json.load(f)

    fr = []
    for i in range(6):
        fr.append(Frame(json_file, 354 + i, fi, hierarchy_lookback=0))
        fr[i].draw()
        fr[i].cen_vec()

    parent = []
    actual = []
    for j in range(1, 6):
        bm = BM(fr[j], fr[j - 1], .01, .0001, .01, .05)
        if len(fr[j].hierarchy) == 0:
            actual.append(np.inf)
            parent.append(np.inf)
        else:
            bm.execute_first_bm()
            bm.parent_finder()
            acc_parent, acc_actual = parent_accuracy(bm)
            parent.append(acc_parent)
            actual.append(acc_actual)

    return parent, actual, fi


def end_finder(cent, img, vec):
    if img[cent] != 1:
        print('center is not in the cell, look at function end_finder for definition of Frame')
        return

    end1 = cent
    i = 1
    while True:
        temp = (cent + i * vec).astype(np.uint16)
        if img[temp[0], temp[1]] != 1:
            break
        end1 = temp
        i += 1

    end2 = cent
    i = 1
    while True:
        temp = (cent - i * vec).astype(np.uint16)
        if img[temp[0], temp[1]] != 1:
            break
        end2 = temp
        i += 1

    return (int(end1[1]), int(end1[0])), (int(end2[1]), int(end2[0]))


def vote_algorithm_maker(fi, hmn=7, l=(.01, .0001, 500, .05)):
    with open(fi) as f:
        jfile = json.load(f)
    fra = []
    for i in range(6):
        fra.append(Frame(jfile, 354 + i, fi, hierarchy_lookback=0))
        fra[i].draw()
        fra[i].cen_vec()

    acc0 = []
    acc1 = []

    for i in range(5):
        pairs_with_parents = []
        for _ in range(hmn):
            bol = BM(fra[i + 1], fra[i], l[0], l[1], l[2], l[3])
            # bol.execute_with_neighbor_bol_match_finder()
            bol.execute_first_bm()
            pairs_with_parents.append(bol.pairs_with_parent)
        if bol.split_num == 0:
            acc0.append(np.inf)
            acc1.append(np.inf)
            continue

        tuple_representation = []
        for x in pairs_with_parents:
            for k, v in x.items():
                tuple_representation.append((k, v[0], v[1]))

        counted = Counter(tuple_representation).most_common(len(pairs_with_parents[0]))

        bol.pairs_with_parent = {}
        bol.pairs = []
        bol.pairs_col1 = []
        bol.pairs_col2 = []
        for c in counted:
            bol.pairs_with_parent[c[0][0]] = (c[0][1], c[0][2])
            bol.pairs.append((c[0][1], c[0][2]))
            bol.pairs_col1.append(c[0][1])
            bol.pairs_col2.append(c[0][2])

        acc = parent_accuracy(bol)
        acc0.append(acc[0])
        acc1.append(acc[1])

    return acc0, acc1, fi




# # with open('Images/james_sample/with_split/100/17_51_551591749604_0-0_jsonData.json') as f:
with open('Images/james_sample/with_split/ws.json') as f:
    json_file = json.load(f)
fr = []
for i in range(6):
    fr.append(Frame(json_file, 354 + i, 'Images/james_sample/with_split/ws.json', hierarchy_lookback=1))
    fr[i].draw()
    fr[i].cen_vec()

bm = BM(fr[1], fr[0], .01, .0001, 500, .05)
# bm.execute_with_neighbor_bm_match_finder()
# bm.execute_first_bm()

# plt.imshow(fr[1].image3)
# plt.show()
# plt.plot(bm.epoch_cost)
# plt.show()
# actual_pairs = []
# for key, val in fr[1].hierarchy.items():
#     actual_pairs.append([key, val[0]])
##################################################################
####################### multiprocessing over files ###############
# list_of_files = glob.glob('Images/james_sample/with_split/100/*.json')
# results = []
# with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
#     # for _ in tqdm(pool.istarmap(foo, iterable),
#     #                    total=len(iterable)):
#     #     pass
#
#     print(multiprocessing.cpu_count())
#     # results = pool.imap_unordered(BM_implement, itertools.product(ll, ll, ll))
#
#     with tqdm(total=len(list_of_files)) as pbar:
#         for i, res in enumerate(pool.imap_unordered(bm_func, list_of_files)):
#             pbar.update()
#             results.append(res)
#
# np.save('/home/sorena/servers/storage/SS/Images/Samples/same_lam/new_children_acc.npy', results)
##################################################################
###################### multi processing ##########################
# ll = [.0001, .005, .001, .05, .01]
# results = []
# with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
#     # for _ in tqdm(pool.istarmap(foo, iterable),
#     #                    total=len(iterable)):
#     #     pass
#
#     print(multiprocessing.cpu_count())
#     # results = pool.imap_unordered(BM_implement, itertools.product(ll, ll, ll))
#
#     with tqdm(total=6 ** 4) as pbar:
#         for i, res in enumerate(pool.imap_unordered(bm_func1, itertools.product(ll, ll, ll, ll))):
#             pbar.update()
#             results.append(res)
#
# np.save('/home/sorena/servers/storage/SS/Images/Samples/same_lam/split_lambdas.npy', results)
######################################################################
#
# fr[4].cell_splitter([1340, 1334, 1260, 1324, 1326, 1328, 1323])
# trmp = true_map_split(fr[5])
#
# lab0, tr0, rtr0 = labtolab(fr[4].split_img, fr[4].split_lab)
# lab1, tr1, trt1 = labtolab(fr[5].image, fr[5].labeled)


# one_res = prb.combm_split(100000, 100000, 100000000,
#                           fr[4].split_img, fr[5].image,
#                           lab0, lab1,
#                           trmp, tr0, tr1, fr[4].max_id,
#                           rel_window=55, g_rate=1.05,
#                           epoch=2000, T=1000, dT=.99, epsilon=.005,
#                           nei_win=100, draw_=False)


# list_of_files = glob.glob('Images/james_sample/with_split/100/*.json')
# ratio = []
# ratio_split_total = []
# ratio2 = []
# diff_list = []
# diff2_list = []
# s_size = []
# s_list = []
# s2_size = []
# r_s = []
# r_s2 = []
# lens = []
# for address in tqdm(list_of_files):
#     with open(address) as f:
#         json_file = json.load(f)
#     # fr354 = Frame(json_file, 354)
#     # fr360 = Frame(json_file, 360)
#
#     fr = []
#     for i in range(6):
#         fr.append(Frame(json_file, 354 + i, hierarchy_lookback=0))
#         fr[i].draw()
#     # plt.imshow(label_img(ms.for_show(fr[4].image), fr[4].image, lab=fr[4].labeled))
#     # plt.show()
#
#     for j in range(1, len(fr) - 1):
#         s_list.append(len(fr[j + 1].hierarchy))
#         ratio_split_total.append(len(fr[j + 1].hierarchy)/fr[j].cell_number)
#         if s_list[-1] == 0:
#             pass
#         else:
#             split_length = [fr[j].cell_list[x].length for x in fr[j + 1].hierarchy.keys()]
#
#             s_size.append(np.min(split_length))
#             hmn_index = fr[j].cell_number - sorted([x.length for x in fr[j].cell_list.values()]).index(s_size[-1])
#             diff_list.append(hmn_index - s_list[-1])
#             r_s.append(diff_list[-1]/s_list[-1])
#             ratio.append(diff_list[-1]/fr[j].cell_number)
#         split2_length = [fr[j].cell_list[x[0]].length for x in fr[j].hierarchy.values()] + [fr[j].cell_list[x].length for x
#                                                                                                 in fr[j].hierarchy.keys()]
#         if not split2_length:
#             pass
#         else:
#             s2_size.append(np.max(split2_length))
#             hmn2_index = fr[j].cell_number - sorted([x.length for x in fr[j].cell_list.values()], reverse=True).index(s2_size[-1])
#             diff2_list.append(hmn2_index - 2 * len(fr[j].hierarchy))
#             r_s2.append(diff2_list[-1]/(2 * len(fr[j].hierarchy)))
#             ratio2.append(diff2_list[-1] / fr[j].cell_number)

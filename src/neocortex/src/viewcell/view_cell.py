#!/usr/bin/env python
from numpy import append
import rospy
from termios import VSTART
import numpy as np
# Sklearn
from sklearn.metrics.pairwise import cosine_similarity
from utils.pairwiseDescriptor import pairwiseDescriptors
# Scipy
from scipy import sparse


class ViewCell(object):
    "A ViewCell object is used to store the information of a single view cell"
    
    _ID = 0

    def __init__(self, template, n_image, x_gc, y_gc, th_gc):
        self.id = ViewCell._ID
        self.template = template
        self.imgs = []
        self.imgs.append(n_image)
        self.x_gc = x_gc
        self.y_gc = y_gc
        self.th_gc = th_gc
        self.first = True
        self.vt_active_decay = rospy.get_param('vt_active_decay')

        ViewCell._ID += 1

class ViewCells(object):
    def __init__(self):
        "Initializes the Local View Match"
        self.size = 0
        self.cells = []
        self.prev_cell = None
        
        self.image_cell = []
        self.id_cell = None
        self.vt_relative_rad = 0
        self.map = []
        self.scores = 0

        self.n_interval = 0
        self.prev_interval = 0
        self.intervals_htm_map = []
        self.prev_intervals_htm_map = []
        self.interval = {}
        self.list_interval = []
        self.interval_cell = []

        self.theta_alpha = rospy.get_param('theta_alpha', 384)
        self.theta_rho = rospy.get_param('theta_rho', 3)
        self.score_interval = rospy.get_param('score_interval', 470)
        self.vt_start = rospy.get_param('vt_start', 0)
        self.vt_match_threshold = rospy.get_param('vt_match_threshold', 0.75)
        self.vt_active_decay = rospy.get_param('vt_active_decay', 0.1)

    def _create_template(self, feature):
        return feature

    def _compare_cells(self, feature, map, bin):
        if bin == 1:
            S = pairwiseDescriptors(feature, map)
            S = np.array(S)
        else:
            S = cosine_similarity(feature, map)
        return S

    def _score(self, template, map, bin):
        #Compute the similarity of a given template with all view cells.
        #scores = []
        #for cell in self.cells:
        #    cell.vt_active_decay -= 0.1 #VT_GLOBAL_DECAY
        #    if cell.vt_active_decay < 0:
        #        cell.vt_active_decay = 0
        #    s = self._compare_cells(template, cell.template, bin)
        #    scores.append(s)
        s = self._compare_cells(template, map, bin)

        return s

    def create_cell(self, template, n_image, x_gc, y_gc, th_gc):
        # Create a new View Cell and register it into the View Cell module
        cell = ViewCell(template, n_image, x_gc, y_gc, th_gc )
        self.id_cell = cell.id
        self.cells.append(cell)
        self.size += 1
        return cell 

    def on_image(self, feature, map, bin, n_image, gc):
        
        x_gc = gc[0][0]
        y_gc = gc[1][0]
        th_gc = gc[2][0]
        template = self._create_template(feature)
        self.scores = self._score(template, map, bin)

        rospy.loginfo("n_image: %d vt start: %d", n_image, self.vt_start)
        if (n_image <= self.vt_start):
            cell = self.create_cell(template, n_image, x_gc, y_gc, th_gc)
        else:
            scores_compare = self.scores[0][:-self.vt_start]
            match = np.nonzero(scores_compare > self.vt_match_threshold)
            if (not np.any(match)):
                cell = self.create_cell(template, n_image, x_gc, y_gc, th_gc)
            else:
                i = np.argmax(scores_compare)
                j = self.image_cell[i]
                #rospy.loginfo("Loop Closure: %s - %s", n_image, i)
                #print("Cell:", j)
                cell = self.cells[j]
                cell.imgs.append(n_image)
                cell.first = False
                #cell.vt_active_decay += self.vt_active_decay
        
        self.image_cell.append(cell.id)

        #img = cell.imgs[0]
        #cell.id = self.image_cell[img]

        #self.prev_cell = cell
        #self.id_cell = cell.id
        return cell

    def on_image_map(self, featureInt, bin, n_image):
        feature = featureInt.astype(dtype=np.bool)
        
        # ****************************************
        # Create Intervals
        # ****************************************
        if (n_image == 0):
            self.interval = {
                "InitEnd": [n_image,n_image],
                #"anchor": feature,
                #"descriptors": feature,
                "global": feature
            }
            self.list_interval.append(self.interval)
            #self.intervals_htm_map = feature
        else:
            # ****************************************
            # Intervals: alpha
            # ****************************************
            if bin == 1:
                anchor = self.list_interval[self.n_interval]["global"]
                S = (anchor.astype(dtype=np.int)).dot((feature.astype(dtype=np.int)).transpose())
                S = S.toarray()
                #S = pairwiseDescriptors(anchor, feature)
                #S = np.array(S)
                alpha = S[0][0]
            
            # ****************************************
            # Intervals: Distance
            # ****************************************
            o_distance = self.list_interval[self.n_interval]["InitEnd"][1] - \
                self.list_interval[self.n_interval]["InitEnd"][0]

            # ****************************************
            # Intervals: create?
            # ****************************************
            if (alpha >= self.theta_alpha) and (o_distance < self.theta_rho):
                self.list_interval[self.n_interval]["InitEnd"][1] = \
                    self.list_interval[self.n_interval]["InitEnd"][1] + 1
                self.list_interval[self.n_interval]["global"] = \
                    self.list_interval[self.n_interval]["global"] + \
                        feature
            else:
                self.interval = {
                    "InitEnd": [n_image, n_image],
                    #"anchor": feature,
                    #"descriptors": feature,
                    "global": feature
                }
                self.list_interval.append(self.interval)
                self.n_interval = self.n_interval + 1

        # ****************************************
        # Map HTM - Intervals
        # ****************************************
        if self.n_interval == 0:
            self.intervals_htm_map = featureInt
        else:
            if (self.prev_interval == self.n_interval):
                self.intervals_htm_map = sparse.vstack([self.prev_intervals_htm_map,\
                    self.list_interval[self.n_interval]["global"]])
            else:
                self.prev_intervals_htm_map = self.intervals_htm_map
                self.intervals_htm_map = sparse.vstack([self.prev_intervals_htm_map,\
                    self.list_interval[self.n_interval]["global"]])
        
        rospy.loginfo("Interval (%d): %s", self.n_interval,\
             self.list_interval[self.n_interval]["InitEnd"])

        # ****************************************
        # Loop Closure
        # ****************************************
        values = (self.intervals_htm_map.astype(dtype=np.int)).dot \
            (featureInt.transpose())
        values_array = values.toarray()
        #print(values_array[:,0])
        
        scores_compare_intervals = values_array[:-3,0]
        #print(scores_compare_intervals)

        if (n_image == 0):
            cell = self.create_cell(0, n_image, 0, 0, 0)
            self.interval_cell.append(cell.id)
        else: 
            match = np.nonzero(scores_compare_intervals > self.score_interval) # 470 480
            #print(match)
            if (not np.any(match)):
                if (self.prev_interval != self.n_interval):
                    cell = self.create_cell(0, n_image, 0, 0, 0)
                else:
                    cell = self.prev_cell
            else:
                rospy.loginfo("Loop Closure")
                i = np.argmax(scores_compare_intervals)
                #print(i)
                j = self.interval_cell[i]
                cell = self.cells[j]
                cell.first = False

        if (self.prev_interval != self.n_interval):
            self.interval_cell.append(cell.id)
        #print(self.interval_cell)

        if (self.n_interval == 0):
            self.interval_map = values_array
        else:
            if (self.prev_interval != self.n_interval):
                self.interval_map = np.pad(self.interval_map, (0, 1), 'constant')
            self.interval_map[:,self.n_interval] = values_array[:,0]
        #print(self.interval_map)

        self.prev_cell = cell
        self.prev_interval = self.n_interval
       
        return self.interval_map, cell

    def get_current_vt(self):
        return self.id_cell
    
    def get_relative_rad(self):
        return self.vt_relative_rad

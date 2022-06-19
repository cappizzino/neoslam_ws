#!/usr/bin/env python
import rospy
from termios import VSTART
import numpy as np
# Sklearn
from sklearn.metrics.pairwise import cosine_similarity
from utils.pairwiseDescriptor import pairwiseDescriptors

VT_START = rospy.get_param('VT_START')
VT_ACTIVE_DECAY = rospy.get_param('VT_ACTIVE_DECAY')
VT_MATCH_THRESHOLD = rospy.get_param('VT_MATCH_THRESHOLD')

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
        self.decay = VT_ACTIVE_DECAY

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
        #    cell.decay -= 0.1 #VT_GLOBAL_DECAY
        #    if cell.decay < 0:
        #        cell.decay = 0
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

        if (n_image <= VT_START):
            cell = self.create_cell(template, n_image, x_gc, y_gc, th_gc)
        else:
            scores_compare = self.scores[0][:-VT_START]
            match = np.nonzero(scores_compare > VT_MATCH_THRESHOLD)
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
                #cell.decay += self.VT_ACTIVE_DECAY
        
        self.image_cell.append(cell.id)

        #img = cell.imgs[0]
        #cell.id = self.image_cell[img]

        #self.prev_cell = cell
        #self.id_cell = cell.id
        return cell

    def on_image_map(self, feature, bin, n_image):
        template = self._create_template(feature)
        scores = self._score(template, self.map, bin)

    def get_current_vt(self):
        return self.id_cell
    
    def get_relative_rad(self):
        return self.vt_relative_rad

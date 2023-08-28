#!/usr/bin/env python
import rospy
import numpy as np


class Interval(object):
    "A Interval object is used to store the information of a interval"
    
    _ID = 0

    def __init__(self, template, n_image):
        self.id = Interval._ID
        self.start = n_image
        self.end = n_image
        self.template = []
        self.template.append(template)
        self.global_rep = template
        self.first = True

        Interval._ID += 1

class Intervals(object):
    def __init__(self):
        "Initializes the Intervals"
        self.size = 0

    def _alpha(self, feature):
        return feature
    
    def rho(self, u, v):
        distance = u - v 
        return distance

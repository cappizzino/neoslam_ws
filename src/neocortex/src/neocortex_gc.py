#!/usr/bin/env python
import rospy
import message_filters
from cv_bridge import CvBridge
# ROS message
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import ByteMultiArray as BIN
from neocortex.msg import ViewTemplate
from std_srvs.srv import Empty
# Numpy
import numpy as np
# Pytorch
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models
from PIL import Image as PIL_Image
# OpenCV
import cv2
# Scipy
from scipy import sparse
# HTM
from nupic.algorithms.temporal_memory import TemporalMemory
from htmresearch.algorithms.apical_tiebreak_temporal_memory import ApicalTiebreakPairMemory as TM
# Other modules
from viewcell.view_cell import ViewCells
from gridcell.grid_cell import GridCells
from experience.experience_map import ExperienceMap
# Other functions
from utils.getLSBH import get_lsbh
from utils.globals import *
import os
import subprocess as sp
import time

class NeoCortex(object):
    'Neocortec Class'
    def __init__(self):

        # ****************************************
        # ROS paramenters
        # ****************************************
        bag_file = rospy.get_param('bag_file')
        bag_filename = rospy.get_param('bag_filename')
        bagfile_folder_path = rospy.get_param('bagfile_folder_path')
        bag_file_start = rospy.get_param('bag_file_start')
        bag_file_duration = rospy.get_param('bag_file_duration')

        bag_topic_01 = rospy.get_param('bag_topic_01')
        bag_topic_02 = rospy.get_param('bag_topic_02')

        self.tempory_memory = rospy.get_param('tempory_memory')

        # ****************************************
        # Load CNN
        # ****************************************
        print ("Load")
        original_model = models.alexnet(pretrained=True)
        class AlexNetConv3(nn.Module):
            'AlexNet Class'
            def __init__(self):
                super(AlexNetConv3, self).__init__()
                self.features = nn.Sequential(
                    # stop at conv3
                    *list(original_model.features.children())[:7]
                )
            def forward(self, x):
                'Forward method'
                x = self.features(x)
                return x

        self.model = AlexNetConv3()
        self.model.eval()

        # ****************************************
        # Load Temporal Memory
        # ****************************************
        self.tm = TemporalMemory(
            # Must be the same dimensions as t-->he SP
            columnDimensions=(2048,),
            # How many cells in each mini-column.
            cellsPerColumn=32,
            # A segment is active if it has >= activationThreshold connected synapses
            # that are active due to infarrActiveState
            activationThreshold=4,
            initialPermanence=0.55,
            connectedPermanence=0.5,
            # Minimum number of active synapses for a segment to be considered during
            # search for the best-matching segments.
            minThreshold=1,
            # The max number of synapses added to a segment during learning
            maxNewSynapseCount=20,
            permanenceIncrement=0.01,
            permanenceDecrement=0.01,
            predictedSegmentDecrement=0.0005,
            maxSegmentsPerCell=100,
            maxSynapsesPerSegment=100,
            seed=42
            )

        # ****************************************
        # Load Apical Temporal Memory
        # ****************************************
        self.externalSize = 180000
        self.externalOnBits = self.externalSize*0.1
        self.bottomUpOnBits = 512

        self.tm_apical = TM(columnCount = 2048,
                basalInputSize = self.externalSize,
                cellsPerColumn=4,
                initialPermanence=0.4,
                connectedPermanence=0.5,
                minThreshold= self.externalOnBits,
                sampleSize=40,
                permanenceIncrement=0.1,
                permanenceDecrement=0.00,
                activationThreshold=int(0.75*(self.externalOnBits+self.bottomUpOnBits)),
                basalPredictedSegmentDecrement=0.00,
                seed = 42
                )    

        # ****************************************
        # Load Matrix - Dimension Reduction and binarizarion
        # ****************************************
        print ("Dimension Reduction")
        createMatrix = 0
        out_dir = os.path.dirname(__file__)
        out_dir = os.path.join(out_dir, 'data')
        if createMatrix == 1:
            self.matrix_p = np.random.randn(64896, 1024)
            self.matrix_p = normc(self.matrix_p)
            np.save(os.path.join(out_dir, 'randomMatrix.npy'), self.matrix_p)
        else:
            self.matrix_p = np.load(os.path.join(out_dir, 'randomMatrix.npy'))

        # ****************************************
        # Load Grid Cells
        # ****************************************
        self.grid_cells = GridCells()
        self.x_gc, self.y_gc, self.th_gc = self.grid_cells.active
        self.gc = [[self.grid_cells.x_gc], [self.grid_cells.y_gc], [self.grid_cells.z_gc]]
        #print(self.gc)

        # ****************************************
        # Load Experience Map
        # ****************************************
        #self.experience_map = ExperienceMap()

        # ****************************************
        # openCV brigde
        # ****************************************
        self.bridge = CvBridge()

        # ****************************************
        # Timer
        # ****************************************
        self.rate = rospy.Rate(1)
        self.rate_image = rospy.Rate(0.5)
        self.rate_odo = rospy.Rate(20)

        # ****************************************
        # Create service client
        # ****************************************
        self.client_camera = rospy.ServiceProxy('/image_saver/save', Empty)
        rospy.loginfo("Waiting for image saver server...")
        wait_camera = self.client_camera.wait_for_service
        if not wait_camera:
            rospy.logerr("Image Saver not available!")
            return
        rospy.loginfo("Connected to Image server")
        self.srv_camera = Empty()

        # ****************************************
        # Define Subscribers
        # ****************************************
        #self.image_sub = message_filters.Subscriber(bag_topic_01, Image)
        #self.odom_sub = message_filters.Subscriber(bag_topic_02, Odometry)
        #ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.odom_sub], #queue_size=1, slop=2)
        #ts.registerCallback(self.image_cb)
        
        self.image_sub = rospy.Subscriber(bag_topic_01, Image, self.image_cb, queue_size=1)
        #self.odom_sub = rospy.Subscriber(bag_topic_02, Odometry, self.odometry_cb)
        #self.gps_sub = rospy.Subscriber('/gps/fix', NavSatFix, self.gps_cb)

        # ****************************************
        # Define Publishers
        # ****************************************
        self.pub_image = rospy.Publisher('/husky_hwu/processed_image', Image, queue_size=1)
        self.features_lsbh = rospy.Publisher('/feats_lsbh', BIN, queue_size=1)
        self.features_htm = rospy.Publisher('/feats_htm', BIN, queue_size=1)
        self.pub_vt = rospy.Publisher('/husky_hwu/LocalView/Template', ViewTemplate, queue_size=1)
        #self.pub_odom = rospy.Publisher('/husky_hwu/odom', Odometry, queue_size=1)
        #self.pub_gps = rospy.Publisher('/husky_hwu/gps', NavSatFix, queue_size=1)

        # ****************************************
        # Variables
        # ****************************************  
        self.feats_lsbh = BIN()
        self.feats_htm = BIN()
        self.feats_htm_map = []
        self.vt_output = ViewTemplate()
        self.odom_output = Odometry()
        self.gps_output = NavSatFix()
        self.image_output = Image()
        self.vtrans = 0
        self.vrot = 0
        self.lat = 0
        self.lon = 0
        self.count = 0
        self.tm_ready = 0

        # ****************************************
        # Neocortex status
        # ****************************************  
        self.rate.sleep()
        rospy.loginfo("Neocortex OK")

        # ****************************************
        # Run Rosbag
        # ****************************************  
        #if bag_file==1:
        #    sp.Popen(['rosbag', 'play', bag_filename, "-s", bag_file_start, "-u", bag_file_duration, "--topics", bag_topic_01, bag_topic_02, "/gps/fix"], cwd = bagfile_folder_path)

        #if bag_file==1:
        #    sp.Popen(['rosbag', 'play', bag_filename, "-s", bag_file_start, "-u", bag_file_duration, "/odometry/filtered:=/husky_hwu/odom"], cwd = bagfile_folder_path)

        # ****************************************
        # Spin
        # ****************************************  
        rospy.spin()

    def odometry_cb(self, odo):
        "Odometry"
        # ****************************************
        # Move Action Pattern
        # ****************************************
        self.vtrans = odo.twist.twist.linear.x
        self.vrot = odo.twist.twist.angular.z
        #actives = self.grid_cells(self.vtrans, self.vrot)
        #self.gc = [[self.grid_cells.x_gc], [self.grid_cells.y_gc], [self.grid_cells.z_gc]]
        #print(self.gc)
        self.rate_odo.sleep()

    def gps_cb(self, gps):
        self.lat = gps.latitude
        self.lon = gps.longitude

    def image_cb(self, msg):
    #def image_cb(self, msg, odo):
        "Image"
        start = time.time()
  
        self.client_camera()

        #self.vtrans = odo.twist.twist.linear.x
        #self.vrot = odo.twist.twist.angular.z

        self.count += 1
        #self.publish_image(msg)

        # ****************************************
        # Preprocessing
        # ****************************************
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        imga = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        imga = cv2.cvtColor(imga, cv2.COLOR_BGR2RGB)
        imga = cv2.GaussianBlur(imga,(5,5),0)
        img = PIL_Image.fromarray(imga)
        input_tensor = preprocess(img)

        # ****************************************
        # CNN
        # ****************************************
        # create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0)

        # move the input and model to GPU for speed if availa/odomble
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.model.to('cuda')

        with torch.no_grad():
            output = self.model(input_batch)

        output_flatten = output[0].numpy()
        output_flatten = output_flatten.flatten('C')
        #output_flatten = output_flatten / np.linalg.norm(output_flatten)
        features_vector = np.array(output_flatten, ndmin=2)

        # ****************************************
        # sLSBH (binarized descriptors)
        # ****************************************
        d1_slsbh = get_lsbh(features_vector, self.matrix_p, 0.25)
        #print (d1_slsbh)
        non_zero_lsbh = np.nonzero(d1_slsbh[0])
        non_zero_lsbh_list = non_zero_lsbh[0].tolist()

        # Send sLSBH features
        self.feats_lsbh.data = d1_slsbh[0]
        self.publish_lsbh(self.feats_lsbh)

        # ****************************************
        # Lateral prediction from Grid Cells
        # ****************************************
        i_gc = self.grid_cells.cells.flatten('C')
        i_gc_sorted = np.sort(i_gc)
        i_gc_sdr = i_gc > i_gc_sorted[self.grid_cells.total_gcells - self.grid_cells.numbers_one - 1]

        # ****************************************
        # HTM
        # ****************************************
        activeColumnIndices = non_zero_lsbh_list

        if self.tempory_memory == "apical":
            self.tm_apical.compute(activeColumnIndices, i_gc_sdr, learn=True)
            activeCells = self.tm_apical.getWinnerCells()
        elif self.tempory_memory == "distal":
            self.tm.compute(activeColumnIndices, learn=True)
            activeCells = self.tm.getWinnerCells()

        d1_htm_sparse = sparse.lil_matrix((1,(self.tm.columnDimensions[0]*self.tm.cellsPerColumn)), dtype=np.int)
        d1_htm_sparse[0,activeCells] = 1
        d1_htm = d1_htm_sparse.toarray()
            
        # Map HTM
        if self.count == 1:
            self.feats_htm_map = d1_htm_sparse
        else:
            self.feats_htm_map = sparse.vstack([self.feats_htm_map, d1_htm_sparse])

        # Send HTM features
        self.feats_htm.data = d1_htm[0]
        self.publish_htm(self.feats_htm)        

        # ****************************************
        # Visual Template
        # ****************************************
        cell_vc = lv.on_image(feature=d1_htm_sparse, map=self.feats_htm_map, bin=1, n_image=self.count-1, gc = self.gc)

        self.vt_output.header.stamp = rospy.Time.now()
        self.vt_output.header.seq += 1
        self.vt_output.current_id = cell_vc.id ## enviar a primeira visual cell correspondente
        self.vt_output.relative_rad = lv.get_relative_rad()
        self.publish_vt(self.vt_output)

        # ****************************************
        # Odometry
        # ****************************************
        #self.odom_output.header.stamp = rospy.Time.now()
        #self.odom_output.twist.twist.linear.x = self.vtrans
        #self.odom_output.twist.twist.angular.z = self.vrot
        #self.publish_odom(self.odom_output)

        # ****************************************
        # GPS
        # ****************************************
        #self.gps_output.header.stamp = rospy.Time.now()
        #self.gps_output.latitude = self.lat
        #self.gps_output.longitude = self.lon
        #self.publish_gps(self.gps_output)

        # ****************************************
        # Image
        # ****************************************
        #self.image_output.header.stamp = rospy.Time.now()
        #self.image_output.height = msg.height
        #self.image_output.width = msg.width
        #self.image_output.encoding = msg.encoding
        #self.image_output.is_bigendian = msg.is_bigendian
        #self.image_output.step = msg.step
        #self.image_output.data = msg.data
        #self.publish_image(self.image_output)
        #self.publish_image(msg)

        # ****************************************
        # Experience Map
        # ****************************************
        #self.experience_map(cell_vc, self.vtrans, self.vrot, self.x_gc, self.y_gc, self.th_gc)

        # ****************************************
        # Time elapsed
        # ****************************************
        end = time.time()
        print (end - start)

        self.rate_image.sleep()

    def publish_image(self, data):
        'Copy and visualize images'
        self.pub_image.publish(data)

    def publish_lsbh(self, data):
        'Publish LSBH features'
        self.features_lsbh.publish(data)

    def publish_htm(self, data):
        'Publish LSBH features'
        self.features_htm.publish(data)

    def publish_vt(self, data):
        'Publish View Template'
        self.pub_vt.publish(data)

    def publish_odom(self, data):
        'Publish Odometry'
        self.pub_odom.publish(data)

    def publish_gps(self, data):
        'Publish GPS'
        self.pub_gps.publish(data)

if __name__ == '__main__':

    # Initialize node
    rospy.init_node("neocortex", anonymous=True)

    # Initilize LocalViewMatch
    lv = ViewCells()

    # Initialize neocortex
    NeoCortex()

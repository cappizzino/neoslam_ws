#! /usr/bin/env python
from __future__ import division
import rospy
import time
import actionlib
from neocortex.msg import NeocortexViewCellAction, NeocortexViewCellFeedback, NeocortexViewCellResult
from std_msgs.msg import ByteMultiArray as BIN
from neocortex.msg import ViewTemplate, infoExp
from viewcell.view_cell import ViewCells
from gridcell.grid_cell import GridCells
# Numpy
import numpy as np
# Image
from cv_bridge import CvBridge
# OpenCV
import cv2
# Pytorch
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models
from PIL import Image as PIL_Image
# Scipy
from scipy import sparse
# HTM
from nupic.algorithms.temporal_memory import TemporalMemory
from htmresearch.algorithms.apical_tiebreak_temporal_memory import ApicalTiebreakPairMemory as TM
# Other functions
from utils.getLSBH import get_lsbh
from utils.globals import *
import os
import matplotlib.pyplot as plt


class ActionServer():

    def __init__(self):
        # ****************************************
        # Load Parameters
        # ****************************************
        self.tempory_memory = rospy.get_param('tempory_memory')
        self.image_filter = rospy.get_param('image_filter')
        self.plot_test = rospy.get_param('plot_test')
        self.plot_test_cnn = rospy.get_param('plot_test_cnn')
        topic_local_view = rospy.get_param('topic_local_view')


        # ****************************************
        # Preprocessing
        # ****************************************
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        # ****************************************
        # Load CNN
        # ****************************************
        rospy.loginfo("Load CNN")
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
        rospy.loginfo("Dimension Reduction")
        createMatrix = 1
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

        # ****************************************
        # openCV brigde
        # ****************************************
        self.bridge = CvBridge()

        # ****************************************
        # Variables
        # ****************************************  
        self.feats_lsbh = BIN()
        self.feats_htm = BIN()
        self.feats_htm_map = []
        self.vt_output = ViewTemplate()
        self.lv = ViewCells()
        self.info_exp = infoExp()
        self.count = 0

        # ****************************************
        # Load Server
        # ****************************************
        self.a_server = actionlib.SimpleActionServer(
            "neocortex_s", NeocortexViewCellAction, execute_cb=self.execute_cb, auto_start=False)
        self.a_server.start()

        # ****************************************
        # Define Publishers
        # ****************************************
        self.pub_vt = rospy.Publisher(topic_local_view, ViewTemplate, queue_size=1)
        self.features_lsbh = rospy.Publisher('/feats_lsbh', BIN, queue_size=1)
        self.features_htm = rospy.Publisher('/feats_htm', BIN, queue_size=1)
        self.info = rospy.Publisher('/info', infoExp, queue_size=1)

    def execute_cb(self, goal):
        start = time.time()
        self.count += 1
        success = True

        feedback = NeocortexViewCellFeedback()
        result = NeocortexViewCellResult()
        rate = rospy.Rate(10)

        if self.a_server.is_preempt_requested():
            self.a_server.set_preempted()
            success = False

        # ****************************************
        # Preprocessing (filter)
        # ****************************************
        img_origin = self.bridge.imgmsg_to_cv2(goal.image, "bgr8")

        if self.image_filter == 'gauss':
            imga = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
            imga = cv2.GaussianBlur(imga,(5,5),0)
        elif self.image_filter == 'clahe':
            hsv_img = cv2.cvtColor(img_origin, cv2.COLOR_BGR2HSV)
            h, s, v = hsv_img[:,:,0], hsv_img[:,:,1], hsv_img[:,:,2]
            v = self.clahe.apply(v)
            hsv_img = np.dstack((h,s,v))
            imga = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        else:
            imga = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)

        if self.plot_test == 'yes':
            fig1, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
            img_origin_rgb = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
            ax1.imshow(img_origin_rgb)
            ax2.imshow(imga)
            plt.show()

        # ****************************************
        # Preprocessing (transform)
        # ****************************************
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
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

        if self.plot_test_cnn == 'yes':
            fig2, axs = plt.subplots(1, 3)
            for i in range(0,3):
                axs[i].imshow(output[0,i,:,:])
            plt.show()

        output_flatten = output[0].numpy()
        output_flatten = output_flatten.flatten('C')
        #output_flatten = output_flatten / np.linalg.norm(output_flatten)
        features_vector = np.array(output_flatten, ndmin=2)

        # ****************************************
        # sLSBH (binarized descriptors)
        # ****************************************
        d1_slsbh = get_lsbh(features_vector, self.matrix_p, 0.25)
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
        # Send HTM features
        self.feats_htm.data = d1_htm[0]
        self.publish_htm(self.feats_htm)    

        # Test with SDR
        if self.count == 1:
            self.Du = d1_htm[0]
        else:
            self.Du = self.Du + d1_htm[0]
        
        n_perc = ((np.count_nonzero(self.Du))/d1_htm.shape[1])*100
        print(n_perc)

        # ****************************************
        # Map HTM
        # ****************************************
        if self.count == 1:
            self.feats_htm_map = d1_htm_sparse
        else:
            self.feats_htm_map = sparse.vstack([self.feats_htm_map, d1_htm_sparse])

        # ****************************************
        # Visual Template
        # ****************************************
        cell_vc = self.lv.on_image(feature=d1_htm_sparse, map=self.feats_htm_map, bin=1, n_image=self.count-1, gc = self.gc)
        rospy.loginfo("View Cell ID: %d, Image: %s", cell_vc.id, cell_vc.imgs)
        rospy.loginfo("Image: %d, View Cell ID: %d", self.count-1, cell_vc.id)

        # ****************************************
        # Time elapsed
        # ****************************************
        end = time.time()
        time_exec = end - start

        # ****************************************
        # Send message
        # ****************************************
        self.vt_output.header.stamp = rospy.Time.now()
        self.vt_output.header.seq += 1
        self.vt_output.current_id = cell_vc.id
        self.vt_output.relative_rad = self.lv.get_relative_rad()
        self.publish_vt(self.vt_output)

        # ****************************************
        # Send info
        # ****************************************
        self.info_exp.current_img = self.count-1
        self.info_exp.current_view_cell = cell_vc.imgs
        self.info_exp.time_exec = time_exec
        self.publish_info(self.info_exp)

        # ****************************************
        # Feedback
        # ****************************************
        feedback.time_exec = goal.image.height
        self.a_server.publish_feedback(feedback)

        # ****************************************
        # Result
        # ****************************************
        last_view_cell = goal.image.height
        result.view_cell = last_view_cell
        if success:
            self.a_server.set_succeeded(result)

        rospy.loginfo(time_exec)

        rate.sleep()

    def publish_lsbh(self, data):
        'Publish LSBH features'
        self.features_lsbh.publish(data)

    def publish_htm(self, data):
        'Publish LSBH features'
        self.features_htm.publish(data)

    def publish_vt(self, data):
        'Publish View Template'
        self.pub_vt.publish(data)

    def publish_info(self, data):
        'Publish Information'
        self.info.publish(data)

if __name__ == "__main__":
    rospy.init_node("neocortex_server")
    s = ActionServer()
    rospy.spin()

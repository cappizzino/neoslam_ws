#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from std_srvs.srv import Empty

import actionlib
from neocortex.msg import NeocortexViewCellAction, NeocortexViewCellGoal

import sys
import os
from os import listdir
from os.path import isfile, join

import cv2
from cv_bridge import CvBridge, CvBridgeError

class VisualEye(object):
    'Eye Class'
    def __init__(self):
        # ****************************************
        # Load Parameters
        # ****************************************
        # General
        self._image_files = rospy.get_param('image_files', False)
        self._waiting_rate = rospy.get_param('waiting_rate', 1.0)
        self._image_topic = rospy.get_param('image_topic')
        # Via topic
        self.image_saver = rospy.get_param('image_saver', False)
        # Via files
        self._sort_files = rospy.get_param('~sort_files', True)
        self._frame_id = rospy.get_param('~frame_id', 'camera')
        self._loop = rospy.get_param('~loop', 1)
        self._image_folder = rospy.get_param('~image_folder', '')

        # ****************************************
        # Bridge
        # ****************************************
        self._cv_bridge = CvBridge()

        # ****************************************
        # Subscriber
        # ****************************************
        self._image_sub = rospy.Subscriber(self._image_topic, Image, self.callback, queue_size=1)

        # ****************************************
        # Publisher (visualization)
        # ****************************************
        self._image_publisher = rospy.Publisher(self._image_topic, Image, queue_size=1)
        
        # ****************************************
        # Create service client
        # ****************************************
        self.client_camera = rospy.ServiceProxy('/image_saver/save', Empty)
        rospy.loginfo("Waiting for Image Saver server...")
        wait_camera = self.client_camera.wait_for_service
        if not wait_camera:
            rospy.logerr("Image Saver not available!")
            return
        rospy.loginfo("Connected to Image Saver server")
        self.srv_camera = Empty()
        self.imag0 = 0

        # ****************************************
        # Create action client
        # ****************************************
        self.goal = NeocortexViewCellGoal()
        self.client = actionlib.SimpleActionClient('neocortex_s', NeocortexViewCellAction)
        rospy.loginfo("Waiting for Neocortex server...")
        wait_neocortex = self.client.wait_for_server()
        if not wait_neocortex:
            rospy.logerr("Neocortex Server not available!")
            return
        rospy.loginfo("Connected to Neocortex server")

        # ****************************************
        # Image folder function
        # ****************************************
        if self._image_files == True:
            if self._image_folder == '' or not os.path.exists(self._image_folder) or not os.path.isdir(self._image_folder):
                rospy.logfatal("Invalid Image folder: %s", self._image_folder)
                sys.exit(0)
            rospy.loginfo("Reading images from %s", self._image_folder)
    
        # ****************************************
        # Create neocortex handler
        # ****************************************
        self.data = None
        self.neocortex_visual_handler()

    def callback(self, msg):
        self.imag0 += 1
        rospy.loginfo("image: %d", self.imag0)
        self.data = msg

    def neocortex_visual_handler(self):
        rate = rospy.Rate(self._waiting_rate)

        if self._image_files == False:
            while not rospy.is_shutdown():
                if self.data != None:
                    # Save image
                    if self.image_saver == 1:
                        self.client_camera()
                    # Send to view cell
                    self.goal.image = self.data
                    self.client.send_goal(self.goal)
                    self.client.wait_for_result()
                    self.client.get_result()
                    # Erase data
                    self.data = None
                rate.sleep()
        else:
            files_in_dir = [f for f in listdir(self._image_folder) if isfile(join(self._image_folder, f))]
            if self._sort_files:
                files_in_dir.sort()
            try:
                while self._loop != 0:
                    for f in files_in_dir:
                        if not rospy.is_shutdown():
                            if isfile(join(self._image_folder, f)):
                                cv_image = cv2.imread(join(self._image_folder, f))
                                if cv_image is not None:
                                    ros_msg = self._cv_bridge.cv2_to_imgmsg(cv_image, "bgr8")
                                    # Send to view cell
                                    self.goal.image = ros_msg
                                    self.client.send_goal(self.goal)
                                    # Send to visualize
                                    ros_msg.header.frame_id = self._frame_id
                                    ros_msg.header.stamp = rospy.Time.now()
                                    self._image_publisher.publish(ros_msg)
                                    rospy.loginfo("Published %s", join(self._image_folder, f))
                                    self.client.wait_for_result()
                                    self.client.get_result()
                                else:
                                    rospy.loginfo("Invalid image file %s", join(self._image_folder, f))
                                rate.sleep()
                        else:
                            return
                    self._loop = self._loop - 1
            except CvBridgeError as e:
                rospy.logerr(e)

if __name__ == '__main__':
    # Initialize node
    rospy.init_node('visual_cortex', anonymous=True)
    try:
        # Initialize Eye
        visual_eye = VisualEye()
    except rospy.ROSInternalException:
        pass

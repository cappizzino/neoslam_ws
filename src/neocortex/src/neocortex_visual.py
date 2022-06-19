#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from std_srvs.srv import Empty

import actionlib
from neocortex.msg import NeocortexViewCellAction, NeocortexViewCellGoal

class VisualEye(object):
    'Eye Class'
    def __init__(self):
        # ****************************************
        # Load Parameters
        # ****************************************
        bag_topic_01 = rospy.get_param('bag_topic_01')

        # ****************************************
        # Subscriber
        # ****************************************
        self.image_sub = rospy.Subscriber(bag_topic_01, Image, self.callback, queue_size=1)
        
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
        self.client = actionlib.SimpleActionClient('neocortex_s', NeocortexViewCellAction)
        rospy.loginfo("Waiting for Neocortex server...")
        wait_neocortex = self.client.wait_for_server()
        if not wait_neocortex:
            rospy.logerr("Neocortex Server not available!")
            return
        rospy.loginfo("Connected to Neocortex server")
        self.goal = NeocortexViewCellGoal()

        rospy.spin()

    def callback(self, msg):
        imag = msg.header.stamp.secs - self.imag0
        if imag != 0:
            # Save image
            rospy.loginfo(msg.header.stamp.secs)
            self.client_camera()
            # Send to view cell
            self.goal.image = msg
            self.client.send_goal(self.goal)
            self.client.wait_for_result()
            self.client.get_result()
            # Return
            self.imag0 = msg.header.stamp.secs

if __name__ == '__main__':
    # Initialize node
    rospy.init_node('visual_cortex', anonymous=True)
    # Initialize Eye
    VisualEye()

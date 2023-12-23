#!/usr/bin/env python
import rospy
import numpy as np
import math as m
from std_msgs.msg import ByteMultiArray as BIN
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
import time
#from utils.pairwiseDescriptor import pairwiseDescriptors
#from neocortex.msg import ViewTemplate
#from ratslam_ros.msg import ViewTemplate

class Plot:
    def __init__(self):
        # ROS
        self.loop_rate = rospy.Rate(1)

        #Sub
        rospy.Subscriber('odom', Odometry, self._odom, queue_size=1)
        rospy.Subscriber('gps/fix', NavSatFix, self.husky_gps, queue_size=1)
        rospy.Subscriber('feats_htm', BIN, self.plotHeatMap, queue_size=1)
        #rospy.Subscriber('LocalView/Template', ViewTemplate, self.plotViewCells, queue_size=1)
        
        # Variables
        self.x = []
        self.y = []
        self.x_gps = []
        self.y_gps = []
        self.feats_htm = []
        self.feats_htm_map = []
        self.viewcells = []

        self.first_odom = 1
        self.first_gps = 1
        self.first_htm = 1
        self.start = 0
        self.flagIgnore = 0

    def _odom(self, msg):
        "Odometry Data"
        if self.first_odom==1:
            self.x0 = msg.pose.pose.position.x
            self.y0 = msg.pose.pose.position.y
            self.first_odom = 0
        self.x.append(msg.pose.pose.position.x - self.x0)
        self.y.append(msg.pose.pose.position.y - self.y0)

    def husky_gps(self, gps):
        "GPS Data"
        if self.first_gps==1:
            self.lat0 = gps.latitude
            self.lon0 = gps.longitude
            self.first_gps = 0
        lat1 = gps.latitude
        lon1 = gps.longitude
        X, Y = self.latlon_to_XY(lat1, lon1)
        self.x_gps.append(X)
        self.y_gps.append(Y)

    def latlon_to_XY(self, lat1, lon1):
        '''Convert latitude and longitude to global X, Y coordinates,
	    using an equirectangular projection.'''
        R_earth = 6371000 # meters
        delta_lat = m.radians(lat1 - self.lat0)
        delta_lon = m.radians(lon1 - self.lon0)
        lat_avg = 0.5 * ( m.radians(lat1) + m.radians(self.lat0))
        X = R_earth * delta_lon * m.cos(lat_avg)
        Y = R_earth * delta_lat
        return X, Y

    def plotHeatMap(self, msg):
        "Similiarity"
        self.feats_htm = msg.data
        self.feats_htm_map.append(msg.data)

    def plotViewCells(self, msg):
        "View Cells"
        self.viewcells.append(msg.current_id)

    def plot(self):
        # Time
        end = time.time()
        elapsed = (int)(end - self.start)

        # if(len(self.x) == 0):
        #     return
        temp = elapsed
        timeSec = np.arange(0,elapsed,temp)

        # # First plot
        # # Format data
        # x = np.array(self.x)
        # y = np.array(self.y)
        # t = np.array(timeSec)
        # fig = plt.figure()
        # ax1 = fig.add_subplot(1,1,1)
        # #ax2 = fig.add_subplot(2,1,2)
        # # Plot
        # ax1.plot(x, y, label="Odometry")
        # #ax1.set(xlabel ='(m)', ylabel ='(m)', xlim =(-10, 250), ylim =(-110, 110))
        # ax1.set(xlabel ='(m)', ylabel ='(m)')
        # #ax1.plot(x1, y1, label = "GPS")
        # ax1.legend()
        # ax1.grid(True)
        # ax1.set_title('Position')
        # fig.savefig('odom_filtered.png')

        # Second plot
        fig2 = plt.figure()
        ax1_fig2 = fig2.add_subplot(1,1,1)
        # Plot
        try:
            self.S_htm = self.pairwiseDescriptors(np.array(self.feats_htm_map), np.array(self.feats_htm_map))
            ax1_fig2 = sns.heatmap(self.S_htm)
            ax1_fig2.set_title('Heat Map HTM')
            fig2.savefig('heatMap.png')
        except:
            print("pairwiseDescriptors does not work.")

        # # Third plot
        # # Format data
        # x_gps = np.array(self.x_gps)
        # y_gps = np.array(self.y_gps)
        # fig3 = plt.figure()
        # ax1_fig3 = fig3.add_subplot(1,1,1)
        # #ax2 = fig.add_subplot(2,1,2)
        # # Plot
        # ax1_fig3.plot(x_gps, y_gps, label="GPS")
        # #ax1.set(xlabel ='(m)', ylabel ='(m)', xlim =(-10, 250), ylim =(-110, 110))
        # ax1_fig3.set(xlabel ='(m)', ylabel ='(m)')
        # #ax1.plot(x1, y1, label = "GPS")
        # ax1_fig3.legend()
        # ax1_fig3.grid(True)
        # ax1_fig3.set_title('Husky Position - GPS')
        # fig3.savefig('gps.png')

        # # Fourth Plot
        # scores = self.S_htm[-1][:]
        # len_scores = len(scores)
        # n_elements = 40
        # if len_scores < n_elements:
        #     scores = []
        # else:
        #     scores = scores[:-n_elements]
        # fig4 = plt.figure()
        # ax1_fig4 = fig4.add_subplot(1,1,1)
        # ax1_fig4.plot(scores)
        # ax1_fig4.set(xlabel ='image', ylabel ='Overlap value', ylim =(0, 1))
        # ax1_fig4.grid(True)
        # ax1_fig4.set_title('Overlap')
        # fig4.savefig('hist_scores.png')

        # # Fifth Plot
        # fig5 = plt.figure()
        # viewcell = np.array(self.viewcells)
        # ax1_fig5 = fig5.add_subplot(1,1,1)
        # ax1_fig5.plot(viewcell, 'bo')
        # ax1_fig5.set(xlabel ='Images', ylabel ='View Cells')
        # ax1_fig5.grid(True)
        # ax1_fig5.set_title('View Cells')
        # fig5.savefig('view_cells.png')


        plt.close('all')

    def pairwiseDescriptors(self, D1, D2):
        # Pairwise comparison
        if sparse.issparse(D1):
            S = D1.dot(D2.transpose())
            D1 = D1.toarray()
            D2 = D2.toarray()
        else:
            S = D1.dot(np.transpose(D2))
        
        nOnes_D1 = np.sum(D1, axis=1)
        nOnes_D2 = np.sum(D2, axis=1)
        D1t = np.transpose(np.vstack((np.ones(len(nOnes_D1)),nOnes_D1)))
        D2t = np.vstack((nOnes_D2,np.ones(len(nOnes_D2))))
        mean_nOnes = D1t.dot(D2t)/2
        S = S / mean_nOnes
        return S

    def waitForData(self):
        count = 0
        self.start = time.time()
        
        while not rospy.is_shutdown():
            count += 1
            # rospy.loginfo(count)
            if(count == 5):
                self.plot()
                count = 0
            self.loop_rate.sleep()

if __name__ == '__main__':
    try:
        # ROS
        rospy.init_node('neoslam_plots', anonymous=False)
        plot = Plot()
        plot.waitForData()
       
    except rospy.ROSInterruptException:
        pass

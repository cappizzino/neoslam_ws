#!/usr/bin/env python
import rospy
import numpy as np
import itertools
from nav_msgs.msg import Odometry
from std_msgs.msg import ByteMultiArray as BIN

class ViewCell(object):
    # A ViewCell object is used to store the information of a single view cell.
    _ID = 0

    def __init__(self, template, image):
        self.images = []
        self.VT_ACTIVE_DECAY = 1.0
        self.id = ViewCell._ID
        self.template = template
        self.images.append(image)
        self.decay = self.VT_ACTIVE_DECAY
        self.first = True

        ViewCell._ID += 1

class GridCells(object):
    'GridCell Class'

    def __init__(self):

        self.cells = np.zeros([PC_DIM_XY, PC_DIM_XY, PC_DIM_TH])
        self.active = a, b, c = [PC_DIM_XY/2, PC_DIM_XY/2, PC_DIM_TH/2]
        self.cells[a, b, c] = 1

        # local excitation - PC_le = PC elements * PC weights
        self.cells = self.compute_activity_matrix(PC_E_XY_WRAP, 
                                                  PC_E_TH_WRAP, 
                                                  PC_W_E_DIM, 
                                                  PC_W_EXCITE)
        
        # local inhibition - PC_li = PC_le - PC_le elements * PC weights
        self.cells = self.cells-self.compute_activity_matrix(PC_I_XY_WRAP, 
                                                             PC_I_TH_WRAP, 
                                                             PC_W_I_DIM, 
                                                             PC_W_INHIB) 
        
        # local global inhibition - PC_gi = PC_li elements - inhibition
        self.cells[self.cells < PC_GLOBAL_INHIB] = 0
        self.cells[self.cells >= PC_GLOBAL_INHIB] -= PC_GLOBAL_INHIB
        
        # normalization
        total = np.sum(self.cells)
        self.cells = self.cells/total

        # Gridcell -> SDR
        self.i_gc_sdr = BIN()
        t_GCN = 0.1
        self.total_sdr = PC_DIM_XY*PC_DIM_XY*PC_DIM_TH
        self.numbers_one = int(round(t_GCN*self.total_sdr))
        self.i_gc_sdr.data = np.zeros((self.total_sdr), dtype=np.bool)

        self.odom_gc = Odometry()
        
        # Define subscribers
        rospy.Subscriber('/husky_hwu/odom', Odometry, self.odometryCb)

        # Define publishers
        self.pub_odom_gc = rospy.Publisher('/husky_hwu/odom_gc', Odometry,queue_size=10)
        self.pub_i_gc = rospy.Publisher('/husky_hwu/i_gc', BIN, queue_size=1)

        # Frequency = 1Hz
        self.rate = rospy.Rate(1)

        rospy.loginfo("Node OK")

        # Spin
        rospy.spin()

    def odometryCb(self, msg):
        vtrans = msg.twist.twist.linear.x
        vrot = msg.twist.twist.angular.z
        vtrans = vtrans*POSECELL_VTRANS_SCALING

        # Path Integration
        # vtrans affects xy direction
        # shift in each th given by the th
        for dir_pc in xrange(PC_DIM_TH): 
            direction = np.float64(dir_pc-1) * PC_C_SIZE_TH
            # N,E,S,W are straightforward
            if (direction == 0):
                self.cells[:,:,dir_pc] = \
                    self.cells[:,:,dir_pc] * (1.0 - vtrans) + \
                    np.roll(self.cells[:,:,dir_pc], 1, 1)*vtrans

            elif direction == np.pi/2:
                self.cells[:,:,dir_pc] = \
                    self.cells[:,:,dir_pc]*(1.0 - vtrans) + \
                    np.roll(self.cells[:,:,dir_pc], 1, 0)*vtrans

            elif direction == np.pi:
                self.cells[:,:,dir_pc] = \
                    self.cells[:,:,dir_pc]*(1.0 - vtrans) + \
                    np.roll(self.cells[:,:,dir_pc], -1, 1)*vtrans

            elif direction == 3*np.pi/2:
                self.cells[:,:,dir_pc] = \
                    self.cells[:,:,dir_pc]*(1.0 - vtrans) + \
                    np.roll(self.cells[:,:,dir_pc], -1, 0)*vtrans

            else:
                pca90 = np.rot90(self.cells[:,:,dir_pc], 
                              int(np.floor(direction *2/np.pi)))
                dir90 = direction - int(np.floor(direction*2/np.pi)) * np.pi/2

                # extend the Posecells one unit in each direction (max supported at the moment)
                # work out the weight contribution to the NE cell from the SW, NW, SE cells 
                # given vtrans and the direction
                # weight_sw = v * cos(th) * v * sin(th)
                # weight_se = (1 - v * cos(th)) * v * sin(th)
                # weight_nw = (1 - v * sin(th)) * v * sin(th)
                # weight_ne = 1 - weight_sw - weight_se - weight_nw
                # think in terms of NE divided into 4 rectangles with the sides
                # given by vtrans and the angle
                pca_new = np.zeros([PC_DIM_XY+2, PC_DIM_XY+2])   
                pca_new[1:-1, 1:-1] = pca90 
                
                weight_sw = (vtrans**2) *np.cos(dir90) * np.sin(dir90)
                weight_se = vtrans*np.sin(dir90) - \
                            (vtrans**2) * np.cos(dir90) * np.sin(dir90)
                weight_nw = vtrans*np.cos(dir90) - \
                            (vtrans**2) *np.cos(dir90) * np.sin(dir90)
                weight_ne = 1.0 - weight_sw - weight_se - weight_nw
          
                pca_new = pca_new*weight_ne + \
                          np.roll(pca_new, 1, 1) * weight_nw + \
                          np.roll(pca_new, 1, 0) * weight_se + \
                          np.roll(np.roll(pca_new, 1, 1), 1, 0) * weight_sw

                pca90 = pca_new[1:-1, 1:-1]
                pca90[1:, 0] = pca90[1:, 0] + pca_new[2:-1, -1]
                pca90[1, 1:] = pca90[1, 1:] + pca_new[-1, 2:-1]
                pca90[0, 0] = pca90[0, 0] + pca_new[-1, -1]

                #unrotate the pose cell xy layer
                self.cells[:,:,dir_pc] = np.rot90(pca90, 
                                                   4 - int(np.floor(direction * 2/np.pi)))

        # Path Integration - Theta
        # Shift the pose cells +/- theta given by vrot
        if vrot != 0: 
            weight = (np.abs(vrot)/PC_C_SIZE_TH)%1
            if weight == 0:
                weight = 1.0

            shift1 = int(np.sign(vrot) * int(np.floor(abs(vrot)/PC_C_SIZE_TH)))
            shift2 = int(np.sign(vrot) * int(np.ceil(abs(vrot)/PC_C_SIZE_TH)))
            self.cells = np.roll(self.cells, shift1, 2) * (1.0 - weight) + \
                             np.roll(self.cells, shift2, 2) * (weight)
        
        self.active = self.get_pc_max(PC_AVG_XY_WRAP, PC_AVG_TH_WRAP)
        
        self.pub_i_gc.publish(self.i_gc_sdr)

        self.odom_gc.pose.pose.position.x = self.active[0]
        self.odom_gc.pose.pose.position.y = self.active[1]
        self.odom_gc.pose.pose.position.z = self.active[2]
        self.pub_odom_gc.publish(self.odom_gc)

        #print(self.active[2])


    def compute_activity_matrix(self, xywrap, thwrap, wdim, pcw): 
        'Compute the activation'
        
        pca_new = np.zeros([PC_DIM_XY, PC_DIM_XY, PC_DIM_TH])
        
        # for nonzero posecell values  
        indices = np.nonzero(self.cells)

        for i,j,k in itertools.izip(*indices):
            pca_new[np.ix_(xywrap[i:i+wdim], 
                           xywrap[j:j+wdim],
                           thwrap[k:k+wdim])] += self.cells[i,j,k]*pcw
         
        return pca_new

    def get_pc_max(self, xywrap, thwrap):
        'Find the x, y, th center of the activity in the network.'

        self.getSDR(self.cells)
        #print(self.i_gc_sdr.data)
        
        x, y, z = np.unravel_index(np.argmax(self.cells), self.cells.shape)
        #print(x, y, z)
          
        z_posecells = np.zeros([PC_DIM_XY, PC_DIM_XY, PC_DIM_TH]) 
      
        zval = self.cells[np.ix_(
            xywrap[x:x+PC_CELLS_TO_AVG*2], 
            xywrap[y:y+PC_CELLS_TO_AVG*2], 
            thwrap[z:z+PC_CELLS_TO_AVG*2]
        )]
        z_posecells[np.ix_(
            PC_AVG_XY_WRAP[x:x+PC_CELLS_TO_AVG*2], 
            PC_AVG_XY_WRAP[y:y+PC_CELLS_TO_AVG*2], 
            PC_AVG_TH_WRAP[z:z+PC_CELLS_TO_AVG*2]
        )] = zval
        
        # get the sums for each axis
        x_sums = np.sum(np.sum(z_posecells, 2), 1) 
        y_sums = np.sum(np.sum(z_posecells, 2), 0)
        th_sums = np.sum(np.sum(z_posecells, 1), 0)
        th_sums = th_sums[:]
        
        # now find the (x, y, th) using population vector decoding to handle 
        # the wrap around 
        x = (np.arctan2(np.sum(PC_XY_SUM_SIN_LOOKUP*x_sums), 
                        np.sum(PC_XY_SUM_COS_LOOKUP*x_sums)) * \
            PC_DIM_XY/(2*np.pi)) % (PC_DIM_XY)
            
        y = (np.arctan2(np.sum(PC_XY_SUM_SIN_LOOKUP*y_sums), 
                        np.sum(PC_XY_SUM_COS_LOOKUP*y_sums)) * \
            PC_DIM_XY/(2*np.pi)) % (PC_DIM_XY)
            
        th = (np.arctan2(np.sum(PC_TH_SUM_SIN_LOOKUP*th_sums), 
                         np.sum(PC_TH_SUM_COS_LOOKUP*th_sums)) * \
             PC_DIM_TH/(2*np.pi)) % (PC_DIM_TH)

        # print x, y, th
        return (x, y, th)

    def getSDR(self, cells):
        i_gc = cells.flatten('C')
        i_gc_sorted = np.sort(i_gc)
        self.i_gc_sdr.data [:] = 0
        self.i_gc_sdr.data  = i_gc > i_gc_sorted[self.total_sdr - self.numbers_one - 1]

def create_pc_weights(dim, var):
    dim_center = int(np.floor(dim/2.))
    
    weight = np.zeros([dim, dim, dim])
    for x, y, z in itertools.product(xrange(dim), xrange(dim), xrange(dim)):
        dx = -(x-dim_center)**2
        dy = -(y-dim_center)**2
        dz = -(z-dim_center)**2
        weight[x, y, z] = 1.0/(var*np.sqrt(2*np.pi))*np.exp((dx+dy+dz)/(2.*var**2))

    weight = weight/np.sum(weight)
    return weight

if __name__ == '__main__':
    
    PC_DIM_XY = 100
    PC_DIM_TH = 18
    TOTAL_SDR = PC_DIM_XY*PC_DIM_XY*PC_DIM_TH

    PC_CELLS_TO_AVG = 3

    PC_AVG_XY_WRAP = range(PC_DIM_XY-PC_CELLS_TO_AVG, PC_DIM_XY) + range(PC_DIM_XY) + range(PC_CELLS_TO_AVG)
    PC_AVG_TH_WRAP = range(PC_DIM_TH-PC_CELLS_TO_AVG, PC_DIM_TH) + range(PC_DIM_TH) + range(PC_CELLS_TO_AVG)

    PC_XY_SUM_SIN_LOOKUP = np.sin(np.multiply(range(1, PC_DIM_XY+1), (2*np.pi)/PC_DIM_XY))
    PC_XY_SUM_COS_LOOKUP = np.cos(np.multiply(range(1, PC_DIM_XY+1), (2*np.pi)/PC_DIM_XY))
    PC_TH_SUM_SIN_LOOKUP = np.sin(np.multiply(range(1, PC_DIM_TH+1), (2*np.pi)/PC_DIM_TH))
    PC_TH_SUM_COS_LOOKUP = np.cos(np.multiply(range(1, PC_DIM_TH+1), (2*np.pi)/PC_DIM_TH))

    PC_W_E_VAR = 1
    PC_W_E_DIM = 7
    PC_W_I_VAR = 2
    PC_W_I_DIM = 5

    PC_W_EXCITE = create_pc_weights(PC_W_E_DIM, PC_W_E_VAR)
    PC_W_INHIB = create_pc_weights(PC_W_I_DIM, PC_W_I_VAR)

    PC_GLOBAL_INHIB = 0.00002

    PC_W_E_DIM_HALF = int(np.floor(PC_W_E_DIM/2.))
    PC_W_I_DIM_HALF = int(np.floor(PC_W_I_DIM/2.))

    PC_C_SIZE_TH = (2.*np.pi)/PC_DIM_TH

    PC_E_XY_WRAP = range(PC_DIM_XY-PC_W_E_DIM_HALF, PC_DIM_XY) + range(PC_DIM_XY) + range(PC_W_E_DIM_HALF)
    PC_E_TH_WRAP = range(PC_DIM_TH-PC_W_E_DIM_HALF, PC_DIM_TH) + range(PC_DIM_TH) + range(PC_W_E_DIM_HALF)
    PC_I_XY_WRAP = range(PC_DIM_XY-PC_W_I_DIM_HALF, PC_DIM_XY) + range(PC_DIM_XY) + range(PC_W_I_DIM_HALF)
    PC_I_TH_WRAP = range(PC_DIM_TH-PC_W_I_DIM_HALF, PC_DIM_TH) + range(PC_DIM_TH) + range(PC_W_I_DIM_HALF)

    POSECELL_VTRANS_SCALING = 1./3.
    ODO_ROT_SCALING         = np.pi/180./7.

    # Initialize node
    rospy.init_node("gridcell", anonymous=True)
    # Initialize neocortex
    GridCells()

#!/usr/bin/python

import rospy
import time
import sys
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point32
from sensor_msgs.msg import PointCloud
from sensor_msgs.msg import ChannelFloat32



class CloudAggregator:
    
    
    
    def __init__(self):
        
        rospy.init_node("cloud_aggregator_node", anonymous=True)
        
        self.read_params()
        self.topics_init()
        self.wait_loop()
        
        if all(self.callback_ready):
            self.map_compute()
    
    
    
    def read_params(self):
        
        self.prefix = rospy.get_param("~prefix", "d435i")
        
        self.tf = [float(x) for x in rospy.get_param("~tf", "0 0 0 0 0 0 1").split(" ")]
        self.publish_tf = rospy.get_param("~publish_tf", "True")
        
        self.freq = float(rospy.get_param("~freq", "0.2"))
        self.freq_min = float(rospy.get_param("~freq", "0.04"))
        
        self.callback_ready = [False, False]
        self.code_ready = True
    
    
    
    def topics_init(self):
        
        rospy.Subscriber("/" + self.prefix + "/cloud/points", PointCloud, self.callback_pcl)
        rospy.Subscriber("/ekf/odom", Odometry, self.callback_odom)
        
        self.pub = rospy.Publisher("/" + self.prefix + "/cloud/map", PointCloud, queue_size=10)
    
    
    
    def wait_loop(self):
        
        sys.stdout.write("Waiting for local cloud publication...\n")
        sys.stdout.flush()
        
        while not rospy.is_shutdown() and not all(self.callback_ready):
            time.sleep(1e-3)
        
        if all(self.callback_ready):
            sys.stdout.write("Local cloud subscribed.  Publishing map cloud at %.1fHz\n" % self.freq)
            sys.stdout.flush()
    
    
    
    def map_compute(self):
        
        msg = PointCloud()
        msg.header.frame_id = self.odom_child_frame
        msg.channels = [ChannelFloat32(),ChannelFloat32(),ChannelFloat32()]
        msg.channels[0].name = "r"
        msg.channels[1].name = "g"
        msg.channels[2].name = "b"
        
        T_ros = 1.0/self.freq
        k_max = 1e3/self.freq_min
        
        while not rospy.is_shutdown():
            
            t = time.time()
            t_ros = rospy.Time.now().to_sec()
            
            self.code_ready = False
            
            if all(self.callback_ready):
            
                p_odom = (self.odom_position.x, self.odom_position.y, self.odom_position.z)
                q_odom = (self.odom_orientation.x, self.odom_orientation.y, self.odom_orientation.z, self.odom_orientation.w)
                p_tf = tuple(self.tf[:3])
                q_tf = tuple(self.tf[3:])
                
                p_cloud = pp_sum(qp_mult(q_odom,p_tf),p_odom)
                q_cloud = qq_mult(q_odom,q_tf)
                
                self.map_points = [tuple2point(pp_sum(qp_mult(q_cloud,(p.x,p.y,p.z)),p_cloud)) for p in self.cloud_points]
                
                msg.header.seq += 1
                msg.header.stamp = self.cloud_stamp
                msg.points += self.map_points
                msg.channels[0].values += self.cloud_channels[0].values
                msg.channels[1].values += self.cloud_channels[1].values
                msg.channels[2].values += self.cloud_channels[2].values
            
            self.pub.publish(msg)
            
            if self.publish_tf:
                msg_tf = tf.TransformBroadcaster()
                msg_tf.sendTransform( tuple(self.tf[:3]),
                                      tuple(self.tf[3:]),
                                      self.cloud_stamp,
                                      self.cloud_frame,
                                      self.odom_child_frame)
            
            self.callback_ready = [False, False]
            self.code_ready = True
            
            sys.stdout.write("2: Global map with %d points computed in %.0fms\n" % (len(msg.points), 1000*(time.time()-t)))
            sys.stdout.flush()
            
            k = 0
            while not rospy.is_shutdown() and rospy.Time.now().to_sec()-t_ros < T_ros and k < k_max:
                k += 1
                time.sleep(1e-3)
    
    
    
    def callback_pcl(self, msg):
        if self.code_ready:
            self.cloud_stamp = msg.header.stamp
            self.cloud_frame = msg.header.frame_id
            self.cloud_points = msg.points
            self.cloud_channels = msg.channels
            self.callback_ready[0] = True
    
    
    
    def callback_odom(self, msg):
        if self.code_ready:
            self.odom_stamp = msg.header.stamp
            self.odom_frame = msg.header.frame_id
            self.odom_child_frame = msg.child_frame_id
            self.odom_position = msg.pose.pose.position
            self.odom_orientation = msg.pose.pose.orientation
            self.callback_ready[1] = True



def qq_mult(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return x, y, z, w

def q_conjugate(q):
    x, y, z, w = q
    return -x, -y, -z, w

def qp_mult(q1, p1):
    q2 = p1 + (0.0,)
    return qq_mult(qq_mult(q1, q2), q_conjugate(q1))[:-1]

def pp_sum(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x = x1 + x2
    y = y1 + y2
    z = z1 + z2
    return x, y, z

def p_conjugate(p):
    x, y, z = p
    return -x, -y, -z

def tuple2point(t):
    p = Point32()
    p.x = t[0]
    p.y = t[1]
    p.z = t[2]
    return p



if __name__ == "__main__":
    CloudAggregator()
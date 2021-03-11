#!/usr/bin/python

import rospy
import time
import sys
import tf
from numpy import pi, sign
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point32, Quaternion
from sensor_msgs.msg import PointCloud, ChannelFloat32
from tf.transformations import euler_from_quaternion, quaternion_from_euler



class Cloud2Map:
    
    
    
    def __init__(self):
        
        rospy.init_node("cloud_to_map_node", anonymous=True)
        
        self.read_params()
        self.topics_init()
        self.wait_loop()
        
        if self.interp_ready:
            self.map_compute()
    
    
    
    def read_params(self):
        
        self.prefix = rospy.get_param("~prefix", "d435i")
        
        self.tf = [float(x) for x in rospy.get_param("~tf", "0 0 0 0 0 0 1").split(" ")]
        self.publish_tf = rospy.get_param("~publish_tf", "True")
        
        self.freq = float(rospy.get_param("~freq", "0.5"))
        self.rate = float(rospy.get_param("~rate", "1.0"))
        
        self.cloud_ready  = False
        self.odom_ready   = False
        self.interp_ready = False
        self.code_ready   = True
        
        self.time = [[0.0, 0.0], 0.0, 0.0]
    
    
    
    def topics_init(self):
        
        rospy.Subscriber("/" + self.prefix + "/cloud/points", PointCloud, self.callback_pcl)
        rospy.Subscriber("/ekf/odom", Odometry, self.callback_odom)
        
        self.pub = rospy.Publisher("/" + self.prefix + "/cloud/map", PointCloud, queue_size=10)
    
    
    
    def wait_loop(self):
        
        sys.stdout.write("\n")
        sys.stdout.write("%-17s %.02fHz\n" % ("Map cloud rate:", self.freq))
        sys.stdout.write("Waiting for local cloud publication...\n")
        sys.stdout.flush()
        
        while not rospy.is_shutdown() and not self.interp_ready:
            time.sleep(1e-3)
    
    
    
    def map_compute(self):
        
        msg = PointCloud()
        msg.header.frame_id = self.odom_child_frame
        msg.channels = [ChannelFloat32(),ChannelFloat32(),ChannelFloat32()]
        msg.channels[0].name = "r"
        msg.channels[1].name = "g"
        msg.channels[2].name = "b"
        
        msg_tf = tf.TransformBroadcaster()
        
        T_ros = 1.0/self.freq
        k_max = 2e3/(self.rate*self.freq)
        
        while not rospy.is_shutdown():
            
            t0 = time.time()
            t_ros = rospy.Time.now().to_sec()
            
            self.code_ready = False
            
            if self.interp_ready:
            
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
            
            self.cloud_ready  = False
            self.interp_ready = False
            self.code_ready   = True
            
            self.time[1] = 1e3*(time.time()-t0)
            
            t0 = time.time()
            
            if self.publish_tf:
                msg_tf.sendTransform( tuple(self.tf[:3]),
                                      tuple(self.tf[3:]),
                                      self.cloud_stamp,
                                      self.cloud_frame,
                                      self.odom_child_frame)
            self.pub.publish(msg)
            
            self.time[2] = 1e3*(time.time()-t0)
            
            sys.stdout.write("[  MAP]: %7d points | Subscribing: %4.0fms | Computing: %4.0fms | Publishing: %4.0fms\n" %
                             (len(msg.points), sum(self.time[0]), self.time[1], self.time[2]))
            sys.stdout.flush()
            
            k = 0
            while not rospy.is_shutdown() and rospy.Time.now().to_sec()-t_ros < T_ros and k < k_max:
                k += 1
                time.sleep(1e-3)
    
    
    
    def callback_pcl(self, msg):
        
        t0 = time.time()
        
        if self.code_ready and self.odom_ready:
            
            self.cloud_stamp = msg.header.stamp
            self.cloud_frame = msg.header.frame_id
            self.cloud_points = msg.points
            self.cloud_channels = msg.channels
            self.cloud_ready = True
            
        self.time[0][0] = 1e3*(time.time()-t0)
    
    
    
    def callback_odom(self, msg):
        
        t0 = time.time()
        
        if self.code_ready and self.cloud_ready:
            
            t = [self.odom_msg.header.stamp.to_sec(), self.cloud_stamp.to_sec(), msg.header.stamp.to_sec()]
            p = [self.odom_msg.pose.pose.position,    Point32(),                 msg.pose.pose.position]
            q = [self.odom_msg.pose.pose.orientation, Quaternion(),              msg.pose.pose.orientation]
            o = [quat_to_rpy(q[0]),                   Point32(),                 quat_to_rpy(q[2])]
            
            k = ((t[1]-t[0])/(t[2]-t[0]))
            
            if abs(o[2].x - o[0].x) > pi: o[2].x += -sign(o[2].x)*2*pi
            if abs(o[2].y - o[0].y) > pi: o[2].y += -sign(o[2].y)*2*pi
            if abs(o[2].z - o[0].z) > pi: o[2].z += -sign(o[2].z)*2*pi
            
            p[1].x = p[0].x + k * (p[2].x - p[0].x)
            p[1].y = p[0].y + k * (p[2].y - p[0].y)
            p[1].z = p[0].z + k * (p[2].z - p[0].z)
            o[1].x = o[0].x + k * (o[2].x - o[0].x)
            o[1].y = o[0].y + k * (o[2].y - o[0].y)
            o[1].z = o[0].z + k * (o[2].z - o[0].z)
            
            if abs(o[1].x) > pi: o[1].x += -sign(o[1].x)*2*pi
            if abs(o[1].y) > pi: o[1].y += -sign(o[1].y)*2*pi
            if abs(o[1].z) > pi: o[1].z += -sign(o[1].z)*2*pi
            
            q[1] = rpy_to_quat(o[1])
            
            self.odom_child_frame = msg.header.frame_id
            self.odom_position = p[1]
            self.odom_orientation = q[1]
            
            self.cloud_ready = False
            self.interp_ready = True
            
        self.odom_msg = msg
        self.odom_ready = True
        
        self.time[0][1] = 1e3*(time.time()-t0)



def quat_to_rpy(q):
    r = euler_from_quaternion((q.x,q.y,q.z,q.w),'sxyz')
    o = Point32()
    o.x = r[0]
    o.y = r[1]
    o.z = r[2]
    return o

def rpy_to_quat(o):
    r = quaternion_from_euler(o.x,o.y,o.z,'sxyz')
    q = Quaternion()
    q.x = r[0]
    q.y = r[1]
    q.z = r[2]
    q.w = r[3]
    return q
    



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
    Cloud2Map()

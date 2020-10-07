#!/usr/bin/python

import rospy
from copy import copy
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Point32
from sensor_msgs.msg import PointCloud
from sensor_msgs.msg import ChannelFloat32



class CloudAggregator:
    
    
    
    def __init__(self):
        
        rospy.init_node("cloud_aggregator_node", anonymous=True)
        
        self.read_params()
        self.wait_cloud()
        self.wait_tfs()
        
        if all(self.callback_bool):
            self.cloud_transform()
    
    
    
    def read_params(self):
        
        self.fixed_frame = rospy.get_param("~fixed_frame", "map")
        
        self.freq = float(rospy.get_param("~freq", "0.2"))
        
        self.tf_frames = []
        self.tf_values = []
        
        self.callback_bool = [False, False]
    
    
    
    def wait_cloud(self):
        
        rospy.Subscriber("/cloud/points", PointCloud, self.callback_cloud)
        
        rate = rospy.Rate(10)
        t = 0.0
        
        while not self.callback_bool[0] and not rospy.is_shutdown():
            
            if t >= 1.0:
                t = 0.0;
                rospy.loginfo("Waiting for cloud publications...")
            else:
                t += 0.1
            rate.sleep()
        
        if self.callback_bool[0]:
            rospy.loginfo("Local cloud subscribed!")
    
    
    
    def wait_tfs(self):
            
        rospy.Subscriber("/tf",        TFMessage,  self.callback_tf)
        rospy.Subscriber("/tf_static", TFMessage,  self.callback_tf)
        
        rate = rospy.Rate(10)
        t = 0.0
        
        while not self.callback_bool[1] and not rospy.is_shutdown():
            
            if t >= 1.0:
                t = 0.0;
                rospy.loginfo("Listening TF data...")
            else:
                t += 0.1
            rate.sleep()
        
        if self.callback_bool[1]:
            rospy.loginfo("Relationship between %s and %s were found!", self.cloud_frame, self.fixed_frame)
            rospy.loginfo("Publishing global map in %.1fHz...", self.freq)
    
    
    
    def cloud_transform(self):
        
        self.pub_map = rospy.Publisher("/cloud/map", PointCloud, queue_size=10)
        
        p_global = Point32()
        self.num_points = 0
            
        self.map_msg = PointCloud()
        self.map_msg.header.frame_id = self.fixed_frame
        self.map_msg.channels = [ChannelFloat32()]
        self.map_msg.channels[0].name = self.cloud_channels[0].name
        
        rate = rospy.Rate(self.freq)
        
        while not rospy.is_shutdown():
            
            t = rospy.Time.now().to_sec()
            
            v = (0.0, 0.0, 0.0)
            q = (0.0, 0.0, 0.0, 1.0)
            
            for k in self.tf_seq:
                v = vv_sum(qv_mult(self.tf_values[k][1], v), self.tf_values[k][0])
                q = qq_mult(self.tf_values[k][1], q)
            
            points_temp = self.cloud_points
            channels_temp = self.cloud_channels
            
            self.map_msg.header.seq += 1
            self.map_msg.header.stamp = self.cloud_stamp
            
            for k in range(len(points_temp)):
                p_local = (points_temp[k].x, points_temp[k].y, points_temp[k].z)
                p_global.x, p_global.y, p_global.z = vv_sum(qv_mult(q, p_local), v)
                self.map_msg.points += [copy(p_global)]
            
            self.map_msg.channels[0].values += channels_temp[0].values
            
            self.pub_map.publish(self.map_msg)
            
            self.num_points = len(self.map_msg.points)
            
            rospy.loginfo("Published map with %d points in %.1f seconds!", self.num_points, rospy.Time.now().to_sec()-t)
            
            rate.sleep()
    
    
    
    def callback_cloud(self, msg):
        
        self.cloud_points = msg.points
        self.cloud_channels = msg.channels
        self.cloud_frame = msg.header.frame_id
        self.cloud_stamp = msg.header.stamp
        self.callback_bool[0] = True
    
    
    
    def callback_tf(self, msg):
        
        for k in range(len(msg.transforms)):
            
            tf_frm = [msg.transforms[k].header.frame_id,
                      msg.transforms[k].child_frame_id]
            
            tf_val = [(msg.transforms[k].transform.translation.x,
                       msg.transforms[k].transform.translation.y,
                       msg.transforms[k].transform.translation.z),
                      (msg.transforms[k].transform.rotation.x,
                       msg.transforms[k].transform.rotation.y,
                       msg.transforms[k].transform.rotation.z,
                       msg.transforms[k].transform.rotation.w)]
            
            if not tf_frm in self.tf_frames:
                self.tf_frames += [tf_frm]
                self.tf_values += [tf_val]
            else:
                self.tf_values[self.tf_frames.index(tf_frm)] = tf_val
        
        if not self.callback_bool[1]:
            
            tf_curr = self.cloud_frame
            self.tf_seq = []
            while tf_curr in [x[1] for x in self.tf_frames]:
                self.tf_seq += [[x[1] for x in self.tf_frames].index(tf_curr)]
                tf_curr = self.tf_frames[self.tf_seq[-1]][0]
            
            if tf_curr == self.fixed_frame:
                self.callback_bool[1] = True



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

def qv_mult(q1, v1):
    q2 = v1 + (0.0,)
    return qq_mult(qq_mult(q1, q2), q_conjugate(q1))[:-1]

def vv_sum(v1, v2):
    x1, y1, z1 = v1
    x2, y2, z2 = v2
    x = x1 + x2
    y = y1 + y2
    z = z1 + z2
    return x, y, z



if __name__ == "__main__":
    CloudAggregator()
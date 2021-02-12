#!/usr/bin/python

import rospy
import numpy
import cv2
from time import time
from cv_bridge import CvBridge
from geometry_msgs.msg import Point32
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from sensor_msgs.msg import ChannelFloat32, PointCloud



class Depth2Cloud:
    
    
    
    def __init__(self):
        
        rospy.init_node("depth_to_cloud_node", anonymous=True)
        
        self.read_params()
        self.topics_init()
        self.wait_loop()
        
        if all(self.callback_bool):
            self.cloud_compute()
    
    
    
    def read_params(self):
        
        self.prefix = rospy.get_param("~prefix", "d435i")
        
        self.image_size = int(rospy.get_param("~image_size", "480"))
        
        self.min_range = int(1000 * float(rospy.get_param("~min_range", "0.001")))
        self.max_range = int(1000 * float(rospy.get_param("~max_range", "10.0")))
        
        self.freq = float(rospy.get_param("~freq", "5.0"))
        
        self.bridge = CvBridge()
        self.callback_bool = [False, False, False, False]
    
    
    
    def topics_init(self):
        
        rospy.Subscriber("/" + self.prefix + "/aligned_depth_to_color/image_raw",                 Image,           self.callback_img_depth)
#        rospy.Subscriber("/" + self.prefix + "/aligned_depth_to_color/image_raw/compressedDepth", CompressedImage, self.callback_cimg_depth)
        rospy.Subscriber("/" + self.prefix + "/aligned_depth_to_color/camera_info",               CameraInfo,      self.callback_info_depth)
        rospy.Subscriber("/" + self.prefix + "/color/image_raw",                                  Image,           self.callback_img_color)
        rospy.Subscriber("/" + self.prefix + "/color/image_raw/compressed",                       CompressedImage, self.callback_cimg_color)
        rospy.Subscriber("/" + self.prefix + "/color/camera_info",                                CameraInfo,      self.callback_info_color)
        
        self.pub = rospy.Publisher("/" + self.prefix + "/cloud/points", PointCloud, queue_size=10)
    
    
    
    def wait_loop(self):
        
        rospy.loginfo("Waiting for depth image publications...")
        rate = rospy.Rate(1000)
        while not all(self.callback_bool) and not rospy.is_shutdown():
            rate.sleep()
        
        if all(self.callback_bool):
            rospy.loginfo("RGB-D images subscribed!")
            rospy.loginfo("Publishing cloud image in %.1fHz...", self.freq)
    
    
    
    def cloud_compute(self):
        
        global x,y,z,rgb,msg
        
        rate = rospy.Rate(self.freq)
        
        while not rospy.is_shutdown():
            
            t = time()
            
            if len(self.img_depth[0]) > self.image_size:
                k_depth = 1.0 * self.image_size / len(self.img_depth[0])
                h_depth = int(k_depth * len(self.img_depth))
                w_depth = self.image_size
                fx_depth = k_depth * self.fx_depth
                fy_depth = k_depth * self.fy_depth
                img_depth = cv2.resize(self.img_depth, (w_depth, h_depth), interpolation=cv2.INTER_NEAREST)
            else:
                h_depth = len(self.img_depth)
                w_depth = len(self.img_depth[0])
                fx_depth = self.fx_depth
                fy_depth = self.fy_depth
                img_depth = self.img_depth
            
            if len(self.img_color[0]) > self.image_size:
                k_color = 1.0 * self.image_size / len(self.img_color[0])
                h_color = int(k_color * len(self.img_color))
                w_color = self.image_size
                fx_color = k_color* self.fx_color
                fy_color = k_color * self.fy_color
                img_color = cv2.resize(self.img_color, (w_color, h_color), interpolation=cv2.INTER_NEAREST)
            else:
                h_color = len(self.img_color)
                w_color = len(self.img_color[0])
                fx_color = self.fx_color
                fy_color = self.fy_color
                img_color = self.img_color
            
            stamp = rospy.Time.from_sec(max([self.stamp_color, self.stamp_depth]))
            
            z_depth = 1e-3 * img_depth.flatten()
            z_depth[z_depth==0.0] = -1.0
            
            i_depth = numpy.arange(0, h_depth*w_depth) / w_depth
            j_depth = numpy.arange(0, h_depth*w_depth) % w_depth
            
            u_depth = j_depth - w_depth / 2.0 - 0.5
            v_depth = i_depth - h_depth / 2.0 - 0.5
            
            x_depth = u_depth * z_depth / fx_depth
            y_depth = v_depth * z_depth / fy_depth
            
            u_color = fx_color * x_depth / z_depth
            v_color = fy_color * y_depth / z_depth
            i_color = (v_color + h_color / 2.0 + 1.0).astype(int)
            j_color = (u_color + w_color / 2.0 + 1.0).astype(int)
            
            idx = (i_color >= 0) & (i_color < h_color) & (j_color >= 0) & (j_color < w_color) & \
                  (z_depth > 1e-3*self.min_range) & (z_depth < 1e-3*self.max_range)
            
            x = x_depth[idx]
            y = y_depth[idx]
            z = z_depth[idx]
            
            rgb = img_color.reshape(h_color*w_color, 3)[idx].astype(int)
            
            msg = PointCloud()
            msg.header.seq += 1
            msg.header.stamp = stamp
            msg.header.frame_id = self.frame_id
            
            msg.points = [array2point(x[k],y[k],z[k]) for k in range(len(z))]
            
            msg.channels = [ChannelFloat32(),ChannelFloat32(),ChannelFloat32()]
            msg.channels[0].name = "r"
            msg.channels[1].name = "g"
            msg.channels[2].name = "b"
            msg.channels[0].values = 1.0 * rgb[:,0] / 255
            msg.channels[1].values = 1.0 * rgb[:,1] / 255
            msg.channels[2].values = 1.0 * rgb[:,2] / 255
            
            rospy.loginfo("Cloud computed in %.0fms...", 1000*(time()-t))
            
            self.pub.publish(msg)
            rate.sleep()
    
    
    
    def callback_img_depth(self, msg):
        self.img_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.stamp_depth = msg.header.stamp.to_sec()
        self.frame_id = msg.header.frame_id
        self.callback_bool[1] = True
    
#    def callback_cimg_depth(self, msg):
#        self.img_depth = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
#        self.stamp_depth = msg.header.stamp.to_sec()
#        self.frame_id = msg.header.frame_id
#        self.callback_bool[1] = True
    
    def callback_info_depth(self, msg):
        self.fx_depth = msg.K[0] * 1.5
        self.fy_depth = msg.K[4] * 1.5
        self.callback_bool[3] = True
    
    def callback_img_color(self, msg):
        self.img_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.stamp_color = msg.header.stamp.to_sec()
        self.callback_bool[0] = True
    
    def callback_cimg_color(self, msg):
        self.img_color = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.stamp_color = msg.header.stamp.to_sec()
        self.callback_bool[0] = True
    
    def callback_info_color(self, msg):
        self.fx_color = msg.K[0]
        self.fy_color = msg.K[4]
        self.callback_bool[2] = True



def array2point(x,y,z):
    p = Point32()
    p.x = x
    p.y = y
    p.z = z
    return p



if __name__ == "__main__":
    Depth2Cloud()
#!/usr/bin/python

import rospy
import numpy
import struct
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
        
        self.min_range = int(1000 * float(rospy.get_param("~min_range", "0.001")))
        self.max_range = int(1000 * float(rospy.get_param("~max_range", "10.0")))
        
        self.freq = float(rospy.get_param("~freq", "5.0"))
        
        self.bridge = CvBridge()
        self.callback_bool = [False, False, False, False]
    
    
    
    def topics_init(self):
        
        rospy.Subscriber("/rgb/image_raw",              Image,           self.callback_img_rgb)
        rospy.Subscriber("/depth/image_raw",            Image,           self.callback_img_depth)
        rospy.Subscriber("/rgb/image_raw/compressed",   CompressedImage, self.callback_cimg_rgb)
        rospy.Subscriber("/depth/image_raw/compressed", CompressedImage, self.callback_cimg_depth)
        rospy.Subscriber("/rgb/camera_info",            CameraInfo,      self.callback_info_rgb)
        rospy.Subscriber("/depth/camera_info",          CameraInfo,      self.callback_info_depth)
        
        self.pub = rospy.Publisher("/cloud/points", PointCloud, queue_size=10)
    
    
    
    def wait_loop(self):
        
        rate = rospy.Rate(10)
        t = 0.0
        
        while not all(self.callback_bool) and not rospy.is_shutdown():
            
            if t >= 1.0:
                t = 0.0;
                rospy.loginfo("Waiting for image publications...")
            else:
                t += 0.1
            rate.sleep()
        
        if all(self.callback_bool):
            rospy.loginfo("RGB-D images subscribed!")
            rospy.loginfo("Publishing cloud image in %.1fHz...", self.freq)
    
    
    
    def cloud_compute(self):
        
        rate = rospy.Rate(self.freq)
        
        while not rospy.is_shutdown():
            
            rgb = self.img_rgb
            depth = self.img_depth
            stamp = rospy.Time.from_sec(max([self.stamp_rgb, self.stamp_depth]))
                
            h = len(depth)
            w = len(depth[0])
            
            z_depth = 1e-3 * depth.flatten()
            z_depth[z_depth==0.0] = -1.0
            
            i_depth = numpy.arange(0,h*w) / w
            j_depth = numpy.arange(0,h*w) % w
            
            u_depth = j_depth - w / 2.0 - 0.5
            v_depth = i_depth - h / 2.0 - 0.5
            
            x_depth = u_depth * z_depth / self.fx_depth
            y_depth = v_depth * z_depth / self.fy_depth
            
            u_rgb = self.fx_rgb * x_depth / z_depth
            v_rgb = self.fy_rgb * y_depth / z_depth
            i_rgb = (v_rgb + h / 2.0 + 1.0).astype(int)
            j_rgb = (u_rgb + w / 2.0 + 1.0).astype(int)
            
            idx = (i_rgb >= 0) & (i_rgb < h) & (j_rgb >= 0) & (j_rgb < w) & \
                  (z_depth > 1e-3*self.min_range) & (z_depth < 1e-3*self.max_range)
            
            x = x_depth[idx]
            y = y_depth[idx]
            z = z_depth[idx]
            n = len(z)
            
            color_rgb = rgb.reshape(h*w,3)[idx].astype(int)
            color_hex = (color_rgb[:,0]<<16) + (color_rgb[:,1]<<8) + (color_rgb[:,2])
            
            msg = PointCloud()
            msg.header.seq += 1
            msg.header.stamp = stamp
            msg.header.frame_id = self.frame_id
            msg.channels = [ChannelFloat32()]
            msg.channels[0].name = "rgb" #"intensity"
            msg.channels[0].values = [0.0] * n
            
            x = x_depth[idx]
            y = y_depth[idx]
            z = z_depth[idx]
            n = len(z)
            
            msg.points = [None] * n
            
            for k in range(n):
                
                msg.points[k] = Point32()
                msg.points[k].x = x[k]
                msg.points[k].y = y[k]
                msg.points[k].z = z[k]
                
                color_float = struct.unpack("f", struct.pack("i", color_hex[k]))[0]
                msg.channels[0].values[k] = color_float
            
            self.pub.publish(msg)
            rate.sleep()
    
    
    
    def callback_img_rgb(self, msg):
        self.img_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.stamp_rgb = msg.header.stamp.to_sec()
        self.callback_bool[0] = True
    
    def callback_img_depth(self, msg):
        self.img_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.stamp_depth = msg.header.stamp.to_sec()
        self.frame_id = msg.header.frame_id
        self.callback_bool[1] = True
    
    def callback_cimg_rgb(self, msg):
        self.img_rgb = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.stamp_rgb = msg.header.stamp.to_sec()
        self.callback_bool[0] = True
    
    def callback_cimg_depth(self, msg):
        self.img_depth = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.stamp_depth = msg.header.stamp.to_sec()
        self.frame_id = msg.header.frame_id
        self.callback_bool[1] = True
    
    def callback_info_rgb(self, msg):
        self.fx_rgb = msg.K[0]
        self.fy_rgb = msg.K[4]
        self.callback_bool[2] = True
    
    def callback_info_depth(self, msg):
        self.fx_depth = msg.K[0] * 1.5
        self.fy_depth = msg.K[4] * 1.5
        self.callback_bool[3] = True



if __name__ == "__main__":
    Depth2Cloud()
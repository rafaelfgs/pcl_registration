#!/usr/bin/python

import rospy
import numpy
import time
import sys
import cv2
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
        
        if all(self.callback_ready):
            self.cloud_compute()
    
    
    
    def read_params(self):
        
        self.prefix = rospy.get_param("~prefix", "d435i")
        
        self.image_size = int(rospy.get_param("~image_size", "240"))
        
        self.min_range = int(1000 * float(rospy.get_param("~min_range", "0.1")))
        self.max_range = int(1000 * float(rospy.get_param("~max_range", "9.9")))
        
        self.freq = float(rospy.get_param("~freq", 0.6e6/self.image_size**2))
        
        self.callback_ready = [False, False, False, False]
        self.code_ready = True
        
        self.bridge = CvBridge()
    
    
    
    def topics_init(self):
        
        rospy.Subscriber("/" + self.prefix + "/aligned_depth_to_color/image_raw",                 Image,           self.callback_img_depth)
        rospy.Subscriber("/" + self.prefix + "/color/image_raw",                                  Image,           self.callback_img_color)
#        rospy.Subscriber("/" + self.prefix + "/aligned_depth_to_color/image_raw/compressedDepth", CompressedImage, self.callback_cimg_depth)
        rospy.Subscriber("/" + self.prefix + "/color/image_raw/compressed",                       CompressedImage, self.callback_cimg_color)
        rospy.Subscriber("/" + self.prefix + "/aligned_depth_to_color/camera_info",               CameraInfo,      self.callback_info_depth)
        rospy.Subscriber("/" + self.prefix + "/color/camera_info",                                CameraInfo,      self.callback_info_color)
        
        self.pub = rospy.Publisher("/" + self.prefix + "/cloud/points", PointCloud, queue_size=10)
    
    
    
    def wait_loop(self):
        
        sys.stdout.write("Waiting for depth image publication...\n")
        sys.stdout.flush()
        
        while not rospy.is_shutdown() and not all(self.callback_ready):
            time.sleep(1e-3)
        
        if all(self.callback_ready):
            sys.stdout.write("RGB-D  image subscribed. Publishing local cloud at %.1fHz\n" % self.freq)
            sys.stdout.flush()
    
    
    
    def cloud_compute(self):
        global img_depth, img_color
        rate = rospy.Rate(self.freq)
        
        while not rospy.is_shutdown():
            
            while not all(self.callback_ready):
                time.sleep(1e-3)
            
            t = time.time()
            
            self.code_ready = False
            
            if len(self.img_depth) > self.image_size:
                k_depth = 1.0 * self.image_size / len(self.img_depth)
                h_depth = self.image_size
                w_depth = int(k_depth * len(self.img_depth[0]))
                fx_depth = k_depth * self.fx_depth
                fy_depth = k_depth * self.fy_depth
                img_depth = cv2.resize(self.img_depth, (w_depth, h_depth), interpolation=cv2.INTER_NEAREST)
            else:
                h_depth = len(self.img_depth)
                w_depth = len(self.img_depth[0])
                fx_depth = self.fx_depth
                fy_depth = self.fy_depth
                img_depth = self.img_depth
            
            if len(self.img_color) > self.image_size:
                k_color = 1.0 * self.image_size / len(self.img_color)
                h_color = self.image_size
                w_color = int(k_color * len(self.img_color[0]))
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
            
            x = +z_depth[idx]
            y = -x_depth[idx]
            z = -y_depth[idx]
            n = len(z)
            
            rgb = img_color.reshape(h_color*w_color, 3)[idx].astype(int)
            
            msg = PointCloud()
            msg.header.seq += 1
            msg.header.stamp = stamp
            msg.header.frame_id = self.frame_id[:-13] + "frame"
            
            msg.points = [array2point(x[k],y[k],z[k]) for k in range(n)]
            
            msg.channels = [ChannelFloat32(),ChannelFloat32(),ChannelFloat32()]
            msg.channels[0].name = "r"
            msg.channels[1].name = "g"
            msg.channels[2].name = "b"
            msg.channels[0].values = 1.0 * rgb[:,0] / 255
            msg.channels[1].values = 1.0 * rgb[:,1] / 255
            msg.channels[2].values = 1.0 * rgb[:,2] / 255
            
            self.pub.publish(msg)
            
            self.callback_ready = [False, False, False, False]
            self.code_ready = True
            
            sys.stdout.write("1: %6d local points computed in %.0fms\n" % (n, 1000*(time.time()-t)))
            sys.stdout.flush()
            
            rate.sleep()
    
    
    
    def callback_img_depth(self, msg):
        if self.code_ready:
            self.img_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.stamp_depth = msg.header.stamp.to_sec()
            self.frame_id = msg.header.frame_id
            self.callback_ready[0] = True
    
    def callback_img_color(self, msg):
        if self.code_ready:
            self.img_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            self.stamp_color = msg.header.stamp.to_sec()
            self.callback_ready[1] = True
    
#    def callback_cimg_depth(self, msg):
#        if self.code_ready:
#            self.img_depth = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
#            self.stamp_depth = msg.header.stamp.to_sec()
#            self.frame_id = msg.header.frame_id
#            self.callback_ready[0] = True
    
    def callback_cimg_color(self, msg):
        if self.code_ready:
            self.img_color = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="rgb8")
            self.stamp_color = msg.header.stamp.to_sec()
            self.callback_ready[1] = True
    
    def callback_info_depth(self, msg):
        if self.code_ready:
            self.fx_depth = msg.K[0] * 1.5
            self.fy_depth = msg.K[4] * 1.5
            self.callback_ready[2] = True
    
    def callback_info_color(self, msg):
        if self.code_ready:
            self.fx_color = msg.K[0]
            self.fy_color = msg.K[4]
            self.callback_ready[3] = True



def array2point(x,y,z):
    p = Point32()
    p.x = x
    p.y = y
    p.z = z
    return p



if __name__ == "__main__":
    Depth2Cloud()
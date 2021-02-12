#!/usr/bin/python

import rospy
import numpy
import cv2
import tf
from time import time
from math import tan, pi
from cv_bridge import CvBridge
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo, CompressedImage, Image



class Stereo2Depth:
    
    
    
    def __init__(self):
        
        rospy.init_node("stereo_to_depth_node", anonymous=True)
        
        self.read_params()
        self.topics_init()
        self.wait_loop()
        
        if all(self.callback_bool):
            self.depth_init()
            self.depth_compute()
    
    
    
    def read_params(self):
        
        self.show_window = rospy.get_param("~show_window", "False")
        
        self.prefix = rospy.get_param("~prefix", "t265")
        
        self.image_size = int(rospy.get_param("~image_size", "320"))
        self.min_range = int(1000 * float(rospy.get_param("~min_range", "0.001")))
        self.max_range = int(1000 * float(rospy.get_param("~max_range", "10.0")))
        
        self.publish_tf = rospy.get_param("~publish_tf", "True")
        
        self.freq = float(rospy.get_param("~freq", 0.6e6/self.image_size**2))
        
        self.R = numpy.array([[ 0.9999756813050,  0.004391118884090,  0.005417240317910],
                              [-0.0043891328387,  0.999990284443000, -0.000378685304895],
                              [-0.0054188510403,  0.000354899209924,  0.999985337257000]])
        self.T = numpy.array( [-0.0641773045063,  0.000311704527121, -0.000004761783202])
        
        self.bridge = CvBridge()
        self.callback_bool = [False, False, False, False]
    
    
    
    def topics_init(self):
        
        rospy.Subscriber("/" + self.prefix + "/fisheye1/image_raw",            Image,           self.callback_img_left)
        rospy.Subscriber("/" + self.prefix + "/fisheye2/image_raw",            Image,           self.callback_img_right)
        rospy.Subscriber("/" + self.prefix + "/fisheye1/image_raw/compressed", CompressedImage, self.callback_cimg_left)
        rospy.Subscriber("/" + self.prefix + "/fisheye2/image_raw/compressed", CompressedImage, self.callback_cimg_right)
        rospy.Subscriber("/" + self.prefix + "/fisheye1/camera_info",          CameraInfo,      self.callback_info_left)
        rospy.Subscriber("/" + self.prefix + "/fisheye2/camera_info",          CameraInfo,      self.callback_info_right)
        
        self.pub_depth_image = rospy.Publisher("/" + self.prefix + "/aligned_depth_to_color/image_raw",   Image,      queue_size=10)
        self.pub_color_image = rospy.Publisher("/" + self.prefix + "/color/image_raw",                    Image,      queue_size=10)
        self.pub_depth_info  = rospy.Publisher("/" + self.prefix + "/aligned_depth_to_color/camera_info", CameraInfo, queue_size=10)
        self.pub_color_info  = rospy.Publisher("/" + self.prefix + "/color/camera_info",                  CameraInfo, queue_size=10)
    
    
    
    def wait_loop(self):
        
        rospy.loginfo("Waiting for stereo image publications...")
        rate = rospy.Rate(1000)
        while not all(self.callback_bool) and not rospy.is_shutdown():
            rate.sleep()
        
        if all(self.callback_bool):
            rospy.loginfo("Stereo images subscribed!")
            rospy.loginfo("Publishing depth image in %.1fHz...", self.freq)
    
    
    
    def depth_init(self):
        
        window_size = 5
        min_disp = 0
        num_disp = 112 - min_disp
        max_disp = min_disp + num_disp
        
        self.stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
                                            numDisparities = num_disp,
                                            blockSize = 16,
                                            P1 = 8*3*window_size**2,
                                            P2 = 32*3*window_size**2,
                                            disp12MaxDiff = 1,
                                            uniquenessRatio = 10,
                                            speckleWindowSize = 100,
                                            speckleRange = 32)
        
        fov = pi/2
        h = self.image_size
        w = h + max_disp
        
        fx = h/2 / tan(fov/2)
        cx = (h - 1)/2.0 + max_disp
        cy = (h - 1)/2.0
        
        self.f = fx
        self.c = cy
        self.m = max_disp
        
        Kl = self.K_left
        Dl = self.D_left
        Rl = numpy.eye(3)
        Pl = numpy.array([[ fx,  0, cx,  0],
                          [  0, fx, cy,  0],
                          [  0,  0,  1,  0]])
        
        Kr = self.K_right
        Dr = self.D_right             
        Rr = self.R
        Pr = numpy.array([[ fx,  0, cx, fx * self.T[0]],
                          [  0, fx, cy, fx * self.T[1]],
                          [  0,  0,  1, fx * self.T[2]]])
        
        (lm1, lm2) = cv2.fisheye.initUndistortRectifyMap(Kl, Dl, Rl, Pl, (w, h), cv2.CV_32FC1)
        (rm1, rm2) = cv2.fisheye.initUndistortRectifyMap(Kr, Dr, Rr, Pr, (w, h), cv2.CV_32FC1)
        
        self.undistort_rectify = {"left" : (lm1, lm2), "right" : (rm1, rm2)}
    
    
    
    def depth_compute(self):
        
        msg_header = Header()
        
        msg_info = CameraInfo()
        msg_info.height = self.image_size
        msg_info.width = self.image_size
        msg_info.distortion_model = "plumb_bob"
        msg_info.K = [self.f, 0, self.c, 0, self.f, self.c, 0, 0, 1]
        msg_info.D = [0, 0, 0, 0, 0]
        msg_info.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        msg_info.P = [self.f, 0, self.c, 0, 0, self.f, self.c, 0, 0, 0, 1, 0]
        
        if self.show_window:
            cv2.namedWindow("Depth Results", cv2.WINDOW_NORMAL)
        
        rate = rospy.Rate(self.freq)
        
        while not rospy.is_shutdown():
            
            t = time()
            
            self.center_undistorted = {"left" : cv2.remap(src = self.img_left,
                                                map1 = self.undistort_rectify["left"][0],
                                                map2 = self.undistort_rectify["left"][1],
                                                interpolation = cv2.INTER_LINEAR),
                                       "right": cv2.remap(src = self.img_right,
                                                map1 = self.undistort_rectify["right"][0],
                                                map2 = self.undistort_rectify["right"][1],
                                                interpolation = cv2.INTER_LINEAR)}
            
            self.disparity = self.stereo.compute(self.center_undistorted["left"], self.center_undistorted["right"]) / 16.0
            
            self.depth = numpy.uint16(self.f * 1000.0*abs(self.T[0]) / (self.disparity[:,self.m:]+0.001))
            self.depth[self.depth < self.min_range] = 0.0
            self.depth[self.depth > self.max_range] = 0.0
            
            self.color = cv2.cvtColor(self.center_undistorted["left"][:,self.m:], cv2.COLOR_GRAY2RGB)
            
            msg_depth = self.bridge.cv2_to_imgmsg(self.depth, encoding="passthrough")
            msg_color = self.bridge.cv2_to_imgmsg(self.color, encoding="rgb8")
            
            msg_header.seq += 1
            msg_header.stamp = rospy.Time.from_sec(max([self.stamp_left, self.stamp_right]))
            msg_header.frame_id = self.prefix + "_aligned_depth_to_color_optical_frame"
            
            msg_depth.header = msg_header
            msg_color.header = msg_header
            msg_info.header = msg_header
            
            self.pub_depth_image.publish(msg_depth)
            self.pub_depth_info.publish(msg_info)
            self.pub_color_image.publish(msg_color)
            self.pub_color_info.publish(msg_info)
            
            if self.publish_tf:
                self.msg_tf = tf.TransformBroadcaster()
                self.msg_tf.sendTransform( (0.0, 0.0, 0.0),
                                           (0.0, 0.0, 0.0, 1.0),
                                           rospy.Time.from_sec(self.stamp_left),
                                           self.prefix + "_aligned_depth_to_color_optical_frame",
                                           self.child_frame_id)
            
            if self.show_window:
                img_window = numpy.hstack((self.color[:,:,0], numpy.uint8(255.0*self.depth/numpy.max(self.depth))))
                cv2.imshow("Depth Results", img_window)
                cv2.waitKey(1)
                if cv2.getWindowProperty("Depth Results", cv2.WND_PROP_VISIBLE) < 1:
                    self.mode = False
            
            rospy.loginfo("Depth computed in %.0fms...", 1000*(time()-t))
            
            rate.sleep()
    
    
    
    def tf_broadcaster(self):
        self.msg_tf = tf.TransformBroadcaster()
        self.msg_tf.sendTransform( (0.0, 0.0, 0.0),
                                   (0.0, 0.0, 0.0, 1.0),
                                   self.stamp_left,
                                   self.prefix + "_aligned_depth_to_color_optical_frame",
                                   self.child_frame_id)
    
    
    
    def callback_img_left(self, msg):
        self.img_left = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.stamp_left = msg.header.stamp.to_sec()
        self.child_frame_id = msg.header.frame_id
        self.callback_bool[0] = True
    
    def callback_img_right(self, msg):
        self.img_right = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.stamp_right = msg.header.stamp.to_sec()
        self.callback_bool[1] = True
    
    def callback_cimg_left(self, msg):
        self.img_left = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.stamp_left = msg.header.stamp.to_sec()
        self.child_frame_id = msg.header.frame_id
        self.callback_bool[0] = True
    
    def callback_cimg_right(self, msg):
        self.img_right = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.stamp_right = msg.header.stamp.to_sec()
        self.callback_bool[1] = True
    
    def callback_info_left(self, msg):
        self.K_left = numpy.reshape(msg.K,(3,3))
        self.D_left = numpy.array(msg.D[0:4])
        self.callback_bool[2] = True
    
    def callback_info_right(self, msg):
        self.K_right = numpy.reshape(msg.K,(3,3))
        self.D_right = numpy.array(msg.D[0:4])
        self.callback_bool[3] = True



if __name__ == "__main__":
    Stereo2Depth()
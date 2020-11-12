#!/usr/bin/python

import rospy
import sys
import os
import time
from copy import copy
from sensor_msgs.msg import PointCloud


num = 0


def callback(data):
    
    if subscribe_data:
        
        global xyz, rgb, num
        
        xyz = copy(data.points)
        rgb = copy(data.channels[0].values)
        num = len(data.points)
        
        sys.stdout.write(str(num) + " Points Subscribed...\n")
        sys.stdout.flush()


def main_function():
    
    global subscribe_data
    
    rospy.init_node("cloud_to_file_node", anonymous=True)
    rospy.Subscriber("/cloud/points", PointCloud, callback)
    file_name = rospy.get_param("~file_name", "pointcloud.pcd")
    
    if os.path.exists(file_name):
        if raw_input("\nFile already exists. Remove it? (y/n): ") == "y":
            os.remove(file_name)
    
    else:
        
        sys.stdout.write("Subscribing PointCloud...\n")
        sys.stdout.flush()
        
        subscribe_data = True
        while not rospy.is_shutdown(): pass
        subscribe_data = False
            
        if raw_input("\nSave " + file_name + "? (y/n): ") == "y" and num > 0:
            
            f = open(file_name,"w")
            
            f.write("# .PCD v0.7 - Point Cloud Data file format\n")
            f.write("VERSION 0.7\n")
            f.write("FIELDS x y z rgb\n")
            f.write("SIZE 4 4 4 4\n")
            f.write("TYPE F F F F\n")
            f.write("COUNT 1 1 1 1\n")
            f.write("WIDTH 1\n")
            f.write("HEIGHT %s\n" % num)
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write("POINTS %s\n" % num)
            f.write("DATA ascii")
            
            k = 0
            while k < num:
                
                f.write("\n%s %s %s %s" % (xyz[k].x, xyz[k].y, xyz[k].z, rgb[k]))
                k += 1
                sys.stdout.write("\rSaving PointCloud File... %5.1f%%" % (100.0*k/num))
                sys.stdout.flush()
                time.sleep(0.0001)
                
            sys.stdout.write("\n")
            sys.stdout.flush()
            f.close()


if __name__ == "__main__":
    try:
        main_function()
    except rospy.ROSInterruptException:
        pass
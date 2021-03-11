#!/usr/bin/python

import rospy
import time
import sys
import os
from sensor_msgs.msg import PointCloud



class Map2File:
    
    
    
    def __init__(self):
        
        rospy.init_node("map_to_file_node", anonymous=True)
        
        self.prefix = rospy.get_param("~prefix", "d435i")
        self.file_name = rospy.get_param("file_name", "/mnt/WD500/UFMG/DISSERTACAO/bags/registration_cloud.pcd")
        
        if os.path.exists(self.file_name):
            rospy.loginfo("File already exists")
            sys.exit()
        else:
            rospy.Subscriber("/" + self.prefix + "/cloud/map", PointCloud, self.callback_pcl)
        
        while not rospy.is_shutdown():
            time.sleep(1e-3)
    
    
    
    def callback_pcl(self, msg):
        
        t = time.time()
        n = len(msg.points)
        
        self.f = open(self.file_name,"w")
        
        self.f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        self.f.write("VERSION 0.7\n")
        self.f.write("FIELDS x y z r g b\n")
        self.f.write("SIZE 4 4 4 4 4 4\n")
        self.f.write("TYPE F F F F F F\n")
        self.f.write("COUNT 1 1 1 1 1 1\n")
        self.f.write("WIDTH 1\n")
        self.f.write("HEIGHT %s\n" % n)
        self.f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        self.f.write("POINTS %s\n" % n)
        self.f.write("DATA ascii")
        
        [self.f.write("\n%f %f %f %f %f %f" % (msg.points[k].x, msg.points[k].y, msg.points[k].z,
                                               msg.channels[0].values[k], msg.channels[1].values[k],
                                               msg.channels[2].values[k])) for k in range(n)]
        
        self.f.close()
        
        sys.stdout.write("3: %d points written to file in %.0fms\n" % (n, 1000*(time.time()-t)))
        sys.stdout.flush()



if __name__ == "__main__":
    Map2File()

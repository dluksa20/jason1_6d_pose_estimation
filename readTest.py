#!/usr/bin/env python3
#
import cv2
import sys
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

dmap = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
print(dmap)

## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

import time

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 90)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

smooth_alpha = 0.1 
smooth_delta = 40
persistency_index = 8
temporal = rs.temporal_filter(smooth_alpha,smooth_delta,persistency_index)
hole_filling = rs.hole_filling_filter()


# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #in meters
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

fps_real = 0

# Streaming loop
try:
    start = time.time()
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        # segun recomendacion los filtros van antes de la alineacion
        # Temporal filter
        frames = temporal.process(frames).as_frameset()
        # Hole filling filter
        frames = hole_filling.process(frames).as_frameset()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth frame
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        end = time.time()
        time_passed = (end-start)
        fps_real = 1/time_passed
        start = time.time()        

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        width = aligned_depth_frame.get_width()
        height = aligned_depth_frame.get_height()
        dist_center = aligned_depth_frame.get_distance(int(width/2),int(height/2))
        print('distance to center: ',"{:.2f}".format(dist_center),'m','  FPS:',int(fps_real),end='\r')

        # Remove background v0.3: with temporal and hole filling filters
        bg_removed = (depth_image < clipping_distance) * depth_image
        _,thresh = cv2.threshold(cv2.convertScaleAbs(bg_removed),255,255,cv2.THRESH_TRUNC)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(bg_removed), cv2.COLORMAP_JET)
        cv2.circle(depth_colormap,(320,240),2,(0,255,0),2)
        cv2.circle(color_image,(320,240),2,(0,0,255),2)
        
        # # PROCESSING
        # morphological transformation for closing small holes: DILATION -> EROTION
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(15,15))
        dilate = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        denoise = cv2.medianBlur(dilate,71)
        cv2.imshow('denoise',denoise)
        
        # find the contours
        contours, hierarchy = cv2.findContours(denoise,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # draw the contours
        cv2.drawContours(color_image,contours,-1,(0,255,0),2)

        # loop over all contour coordinates
        for i,c in enumerate(contours):
            if i!=0:
                continue
            if cv2.contourArea(c) < 14000:
                continue
            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box)
            box = np.array(box,dtype='int')
            (tl,tr,br,bl) = box # in the [x,y] coordinates form
            #print(tl,tr,bl,br,' ',end='\r')
            #cv2.circle(depth_colormap,tl,9,(255,0,0),4)
            #cv2.circle(depth_colormap,tr,9,(0,255,0),4)
            #cv2.circle(depth_colormap,bl,9,(0,0,255),4)
            #cv2.circle(depth_colormap,br,9,(255,255,255),4)
            cv2.drawContours(color_image,[box],-1,(255,0,0),2)

            x1 = tl[0]
            x2 = tr[0]
            x_center = int((x1+x2)/2)
            y1 = tl[1]
            y2 = bl[1]
            y_center = int((y1+y2)/2)
            cv2.circle(color_image,(x_center,y_center),9,(255,255,255),4)
            

            # for (x,y) in box:
            #     (tl,tr,bl,br) = box



        # Render images:
        images = np.hstack((color_image,depth_colormap))
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
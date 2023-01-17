#!/usr/bin/env python3
import  time
import cv2
import depthai as dai
import numpy as np
############oak相机内参数######################
f=857.06
baseline=60
distance=True

extended_disparity = True
subpixel = False
lr_check = True

# Create pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)##修改深度图尺寸
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)


depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

depth.setLeftRightCheck(lr_check)
depth.setExtendedDisparity(extended_disparity)
depth.setSubpixel(subpixel)


monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)


xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("disparity")
depth.disparity.link(xout.input)


# Connect to device and start pipeline
if __name__=="__main__":
    with dai.Device(pipeline) as device:
        qDisp = device.getOutputQueue(name="disparity", maxSize=2, blocking=False)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255,255,255)
        thickness = 1
        stereo_disp_multiplier = 255.0 / depth.initialConfig.getMaxDisparity()

        while True:
            start=time.time()

            disp_output = qDisp.get().getFrame()
    ##########计算xyz#########################
            if distance:
                disp=np.array(disp_output)
                disp=np.where(disp==0,-0.00001,disp)
                z=0.001*baseline*f/(disp)
            # print('深度图帧率为{}'.format(1/(time.time()-start))+'fps')
            stereo_disp = (disp_output * stereo_disp_multiplier).astype(np.uint8)
            stereo_disp_vis = cv2.applyColorMap(stereo_disp, cv2.COLORMAP_INFERNO)
            stereo_disp_vis = cv2.putText(stereo_disp_vis, "Stereo Disp"+'      {}'.format(int(1/(time.time()-start)))+'Fps', (400, 600), font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow("Disparity", stereo_disp_vis)

            if cv2.waitKey(1) == ord('q'):
                break
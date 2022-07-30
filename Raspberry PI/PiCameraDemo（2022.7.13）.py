import signal
import sys
import cv2
from picamera import PiCamera, Color
from time import sleep

#######################################
#Allow Ctrl-C in case something locks the screen
#######################################

def emergency(signal, frame):
    print('Something went wrong!')
    sys.exit(0)

signal.signal(signal.SIGINT, emergency)

#######################################
#Initialization
#######################################

demoCamera = PiCamera()


#######################################
#Simple Preview
#######################################


demoCamera.start_preview()
sleep(5)
demoCamera.stop_preview()



########################################
#Rotation
########################################

"""
demoCamera.rotation = 180
demoCamera.start_preview()
sleep(10)
demoCamera.stop_preview()
"""


########################################
#Take picture
########################################



demoCamera.rotation = 180
demoCamera.start_preview()
# user = raw_input('Enter to take photo')
str = input("Enter your input: ")
print ("Received input is : ", str)


# 常用命令：
# 两秒钟（时间单位为毫秒）延迟后拍摄一张照片，并保存为 image.jpg
# raspistill -t 2000 -o Desktop/image.jpg

# 获取一张照片并保存为一个文件
# raspistill -t 2000 -o - > my_file.jpg

demoCamera.capture('/home/pi/Desktop/sample.jpg')
demoCamera.stop_preview()




########################################
#Record Video
#Can use omxplayer on RPi to play
########################################

"""
demoCamera.rotation = 180
demoCamera.start_preview()
demoCamera.start_recording('/home/pi/Desktop/sampleVideo.h264')
sleep(3)
demoCamera.stop_recording()
demoCamera.stop_preview()
"""

########################################
#Annotation
########################################


demoCamera.rotation = 180
demoCamera.start_preview()
demoCamera.annotate_background = Color('white')
demoCamera.annotate_foreground = Color('red')
demoCamera.annotate_text = " SWS3009B - 2021"
sleep(5)
demoCamera.capture('/home/pi/Desktop/classPhoto.jpg')
demoCamera.stop_preview()


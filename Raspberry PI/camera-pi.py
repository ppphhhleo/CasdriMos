# import the necessary packages
import RPi.GPIO as GPIO
from picamera.array import PiRGBArray
from picamera import PiCamera, Color
import time
from pymongo import MongoClient
import datetime


GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# import cv2
# initialize the camera and grab a reference to the raw camera capture

class client_ca:
    def __init__(self):
        self.window_name = "camera"
        self.upload_time = 0
        self.query_time = 0
        self.now = 0

    def upload_image(self, frame):

        size = (frame.shape[1], frame.shape[0])
        img_bytes = frame.tobytes()
        # img_bytes = base64.b64encode(img_bytes).decode("utf-8")
        self.now = datetime.datetime.now()
        thetime = datetime.datetime.utcnow()
        timestr = datetime.datetime.strftime(thetime, '%d:%m:%Y:%H:%M:%S')

        pic_doc = {"timestamp": timestr, "images": img_bytes, "size": size}
        pictures.insert_one(pic_doc)
        self.upload_time = datetime.datetime.now()
        print("Successfull insert on document to the DB, ", (self.upload_time-self.now).microseconds)
        return self.upload_time

    def get_response(self, upt):
        while (True):
            query_res = respo.find({}, {"_id": 0, "timestamp": 1, "response": 1}).sort("timestamp", -1).limit(1)
            try:
                tmp_res = query_res[0]["response"]
            except:
                time.sleep(5)
                print("find no response")
                continue

            if tmp_res == "":
                self.now = datetime.datetime.now()
                print("find no response, continue to request, since upload: ", (self.now - upt).microseconds)
                continue
            else:
                self.now = datetime.datetime.now()
                print("find a response: {:s} since upload: ".format(tmp_res), (self.now - upt).microseconds)
                # break
                return tmp_res

    def upload_get_response(self, frame):

        # size = (frame.shape[1], frame.shape[0])
        # img_bytes = frame.tobytes()
        #
        #
        # thetime = datetime.datetime.utcnow()
        # timestr = datetime.datetime.strftime(thetime, '%d:%m:%Y:%H:%M:%S')
        #
        # pic_doc = {"timestamp": timestr, "images": img_bytes, "size": size, "response": ""}
        # pictures.insert_one(pic_doc)
        # upload_time = datetime.datetime.now()
        # print("Successfull insert on document to the DB, upload time: ", (upload_time - thetime).seconds)

        upt = self.upload_image(frame)
        res = self.get_response(upt)
        return res
        # while(True):
        #     query_res = respo.find({}, {"_id":0, "timestamp": 1, "response":1}).sort("timestamp", -1).limit(1)
        #     tmp_res = query_res[0]["response"]
        #     if tmp_res == "":
        #         query_time = datetime.datetime.now()
        #         print("find no response, since upload: ", (query_time - upload_time).seconds)
        #         continue
        #     else:
        #         query_time = datetime.datetime.now()
        #         print("find a response, since upload: ", tmp_res, (query_time - upload_time).seconds)
        #         # break
        #         return tmp_res

    def setup_a(self,a,b,c):
        # state number = a + b*2 + c*4
        if a:
            GPIO.setup(11, GPIO.OUT)
            GPIO.output(11, GPIO.HIGH)
        else:
            GPIO.setup(11, GPIO.OUT)
            GPIO.output(11, GPIO.LOW)
        if b:
            GPIO.setup(13, GPIO.OUT)
            GPIO.output(13, GPIO.HIGH)
        else:
            GPIO.setup(13, GPIO.OUT)
            GPIO.output(13, GPIO.LOW)
        if c:
            GPIO.setup(15, GPIO.OUT)
            GPIO.output(15, GPIO.HIGH)
        else:
            GPIO.setup(15, GPIO.OUT)
            GPIO.output(15, GPIO.LOW)

    def actuators(self, res):
        if res == "face" :
            self.setup_a(0,0,0)
        if res == "smoke":
            self.setup_a(1, 0, 0)
        if res == "drink":
            self.setup_a(1, 1, 0)
        if res == "phone":
            self.setup_a(0, 0, 1)


    def capture_image(self, cap):
        cv2.namedWindow(self.window_name)  # camera's name
        # cap = cv2.VideoCapture(camera_idx)
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow(self.window_name, frame)
            c = cv2.waitKey(1)

            if c & 0xFF == ord('q'):
                break
            if c == ord("n"):
                print(frame.shape)
                # self.upload_image(frame)
                self.upload_get_response(frame)
                # self.predicta(frame)
            if c == ord("m"):
                print(frame.shape)
                ret, buffer = cv2.imencode(".jpg", frame)
                print("ret: ", ret)
                print("buffer: ", buffer)

    def capture_stream(self,cap):
        cv2.namedWindow(self.window_name)  # camera's name
        # cap = cv2.VideoCapture(camera_idx)
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow(self.window_name, frame)
            c = cv2.waitKey(1)
            res = self.upload_get_response(frame)
            # self.actuators(res)
            if c & 0xFF == ord('q'):
                break


    def pi_caputure(self, show=False): # PI 视频流
        print("Initialize Pi Camera")
        camera = PiCamera(resolution='640x480', framerate=24)
        rawCapture = PiRGBArray(camera)
        # allow the camera to warmup
        time.sleep(1)
        # grab an image from the camera
        # pi capture a frame
        print("Capture a frame")
        camera.capture(rawCapture, format="bgr")
        image = rawCapture.array
        print("BGR array: ", image.shape)  # (480, 640, 3)
        # pi upload the image
        # print("upload the image")
        # upload_image(image)

        # if show == True:
        #     cv2.imshow("Image", image)
        #     cv2.waitKey(0)

        print("test to upload and get response, with a frame")
        # testc = client_ca()
        # 测试
        db_response = self.upload_get_response(image)
        print("pi get response:", db_response)

        print("PI begins to upload continous frames.")
        while(True):
            time.sleep(1)
            camera.capture(rawCapture, format="bgr")
            image = rawCapture.array
            start_time = datetime.datetime.now()
            db_response = self.upload_get_response(image)
            respon_time = datetime.datetime.now()
            print("Get response: ", db_response, "since: ", (respon_time - start_time).microseconds)

        # camera
        # camera = PiCamera()
        # camera.resolution = (640, 480)
        # camera.framerate = 32
        # rawCapture = PiRGBArray(camera, size=(640, 480))
        # time.sleep(0.1)
        # for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        #     image = frame.array
        #     print("BGR array: ", image.shape)  # (480, 640, 3)
        #     # if show == True:
        #     #     cv2.imshow("Frame", image)
        #     #     key = cv2.waitKey(1) & 0xFF
        #
        #     start_time = datetime.datetime.now()
        #     db_response = self.upload_get_response(image)
        #     respon_time = datetime.datetime.now()
        #     print("Get response: ", db_response, "since: ", (respon_time - start_time).microseconds)


    def run1(self): # PI 视频流
        self.pi_caputure()

    def run2(self): # 测试版
        cap = cv2.VideoCapture(0)
        self.capture_image(cap)
        cap.release()
        cv2.destroyWindow(self.window_name)

    def run3(self):  # 视频流
        cap = cv2.VideoCapture(0)
        self.capture_stream(cap)
        cap.release()
        cv2.destroyWindow(self.window_name)


if __name__ == "__main__":
    print("Connect to mongoDB")
    mongodbUri = 'mongodb://user2:myadmin@101.35.252.209:27017/admin'
    # user1，myadmin
    # user2，myadmin
    # user3，myadmin
    client = MongoClient(mongodbUri)
    mydb = client['images']
    pictures = mydb["pictures"]
    respo = mydb["responses"]
    print("Successfully connect to mongoDB")

    test_c = client_ca()
    # test_c.pi_caputure(show=False)
    # test_c.pi_caputure(show=True)
    # test_c.run2()
    test_c.run3()

    # # pi connect to a camera
    # print("Initialize Pi Camera")
    # camera = PiCamera(resolution='640x480', framerate=24)
    # rawCapture = PiRGBArray(camera)
    # # allow the camera to warmup
    # time.sleep(1)
    # # grab an image from the camera
    #
    # # pi capture a frame
    # print("Capture a frame")
    # camera.capture(rawCapture, format="bgr")
    # image = rawCapture.array
    # print("BGR array: ", image.shape) # (480, 640, 3)
    #
    # # pi upload the image
    # print("upload the image")
    # # upload_image(image)
    # print("test to upload and get response, with a frame")
    # testc = client_ca()
    # db_response = testc.upload_get_response(image)

    # display the image on screen and wait for a keypress
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)

    # camera
    # for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    #     image = frame.array
    #     start_time = datetime.datetime.now()
    #     db_response = testc.upload_get_response(image)
    #     respon_time = datetime.datetime.now()
    #     print("Get response: ", db_response, "since: ", (respon_time-start_time).seconds)


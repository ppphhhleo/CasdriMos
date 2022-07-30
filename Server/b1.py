from hashlib import sha512
from pymongo import MongoClient
from flask import Flask, render_template, Response, request,flash, redirect,url_for
from PIL import Image # Pillow library
from base64 import b64decode
import cv2
import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, \
    set_logging
from utils.torch_utils import select_device, time_synchronized
import base64
from numpy import random
import datetime
import requests
import json


app = Flask('__name__')


def check_register(user, pw):
    if user == "a" or len(pw) < 6:
        return 0
    return 1

USER_OK = 0
USER_NOT_EXIST = 1
USER_BAD_PASSWORD = 2

def finduser(userid):
    return users.find_one({"userid": userid})

def check_pw(userid, passwd):
    user_data = finduser(userid)
    if user_data is None:
        return USER_NOT_EXIST
    pw = sha512(passwd.encode('utf-8')).hexdigest()
    if pw == user_data["password"]:
        return USER_OK
    else:
        return USER_BAD_PASSWORD

def legal_pw(pw):
    if len(pw) < 6:
        return "Password too short, please use more than six digits"
    else:
        return 1


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def predicta(frame):

    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    # Set Dataloader & Run inference
    img = letterbox(frame, new_shape=imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    # pred = model(img, augment=opt.augment)[0]
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt_conf_thres, opt_iou_thres)
    #
    # # Process detections
    ret = []
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]}'
                prob = round(float(conf) * 100, 2)  # round 2
                ret_i = [label, prob, xyxy]
                ret.append(ret_i)
    # return
    # label  'face' 'smoke' 'drink' 'phone'
    # prob
    # xyxy position
    if ret == []:
        # print("detect no driver!")
        return "no driver"
    else:
        # print("prediction:", ret)
        # print("prediction result:", ret[0][0])
        return ret[0][0]

def upload_actuators(resp):
    thetime = datetime.datetime.utcnow()
    timestr = datetime.datetime.strftime(thetime, '%d:%m:%Y:%H:%M:%S')
    res_doc = {"timestamp": timestr, "response": resp}
    respo.insert_one(res_doc)
    res_time = datetime.datetime.now()
    print("Server upload a response: ", (res_time - thetime).microseconds)

def detect_get_response(image):
    timea = datetime.datetime.now()
    resul = predicta(image)
    timef = datetime.datetime.now()
    print(resul, " is the result of prediction, consume: ", (timef-timea).microseconds)
    return resul, 200

def check_num(res):

    if res == "smoke":
        info[0] += 1
    if res == "drink":
        info[1] += 1
    if res == "smoke":
        info[2] += 1
    if res == "face":
        info[3] += 1

#     return "Dandelion", 200
@app.route('/')
def root():
    return render_template("index2.html", situation = "no driver")


@app.route("/")
def index():
    return render_template("index1.html")

@app.route("/cool_form", methods=["GET", "POST"])
def cool_form():
    if request.method == "POST":
        if ("login" in request.form):
            return redirect(url_for("driver_login"))
        if ("register" in request.form):
            return redirect(url_for("driver_register"))
        return redirect(url_for("video_home"))
    return render_template("cool_form.html")

@app.route("/login",methods = ["GET", "POST"])
def driver_login():
    hint = "Please Login"
    return render_template("login.html", hint=hint)



@app.route("/get_login", methods = ["POST"])
def driver_login_a():
    print(request.form)
    user_name = request.form["name"]
    user_pw = request.form["password"]
    check_login = check_pw(user_name, user_pw)
    if check_login == USER_NOT_EXIST:
        hint = "User Not Exist! Please check your UserID or Register"
        return render_template("login.html", hint = hint)
    if check_login == USER_BAD_PASSWORD:
        hint = "User Exists, but WRONG password, please try again"
        return render_template("login.html", hint = hint)
    return redirect(url_for("video_home"))


@app.route("/register", methods = ["GET", "POST"])
def driver_register():
    hint = "Please register"
    return render_template("register.html", hint = hint)


@app.route("/get_register", methods = ["POST"])
def driver_register_a():
    print(request.form)
    if "name" not in request.form or "password" not in request.form:
        hint = "Must specify userid and password"
        return render_template("register.html", hint=hint)
    user_name = request.form["name"]
    user_pw = request.form["password"]
    if finduser(user_name) is not None:
        hint =  "User " + user_name + " exists. Please Login or try again."
        return render_template("register.html", hint=hint)
    check = legal_pw(user_pw)
    if check != 1:
        hint = check
        return render_template("register.html", hint=hint)
    else:
        pw = sha512(user_pw.encode('utf-8')).hexdigest()
        users.insert_one({"userid": user_name, "password": pw})
        hint = "Successfully Register, please login"
        return render_template("login.html", hint=hint)


@app.route("/stop", methods=["POST", "GET"])
def stop():
    if "stop" in request.form:

        return render_template("numbers.html", d=info[0], p=info[1], s=info[2], judge=info[3])


@app.route('/detect', methods=['POST'])
def detect_image():
    try:
        # Get the JSON data
        data = request.get_json()

        time3 = datetime.datetime.now()
        img_data = data["images"]
        img_data = b64decode(img_data)  # if the client encode
        img = Image.frombytes("RGB", size=data["size"], data=img_data)
        img1 = np.array(img)
        time4 = datetime.datetime.now()
        print("Server preprocesses the image, consume:", (time4-time3).microseconds)

        ret, buffer = cv2.imencode('.jpg', img1)
        frame = buffer.tobytes()
        img_stream = base64.b64encode(frame).decode()
        time5 = datetime.datetime.now()
        print("Server process image stream for web done, consume:", (time5-time4).microseconds)

        res = detect_get_response(img1)
        check_num(res)
        upload_actuators(res)
        render_template("index2.html", img_stream=img_stream, situation=res)
        # return render_template("index2.html", img_stream=img_stream, situation = res)
        return res

    except Exception as e:

        return "Error processing request." + str(e), 500


@app.route("/count", methods=["POST"])
def judge():
    try:
        data = request.get_json()


    except Exception as e:
        return "Error processing request." + str(e), 500


print("Connect to mongoDB")
mongodbUri = 'mongodb://user2:myadmin@ip:27017/admin'
# ip should be replaced by yours
# user1，myadmin
# user2，myadmin
# user3，myadmin
client = MongoClient(mongodbUri)
mydb = client['images']
pictures = mydb["pictures"]
respo = mydb["responses"]
users = mydb["users"]
print("Successfully connect to mongoDB")


imgsz = 640
weights = r'weights/best.pt'
opt_device = ''
opt_conf_thres = 0.3
opt_iou_thres = 0.45
device = select_device(opt_device)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
if half:
    model.half()  # to FP16
# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]


app.run(debug=True, host="127.0.0.1")
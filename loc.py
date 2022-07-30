from flask import Flask, request, url_for, redirect, render_template,Response
import cv2
from hashlib import sha512
from pymongo import MongoClient
# import requests
# import json
import random
import torch
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, \
    set_logging
from utils.torch_utils import select_device, time_synchronized
import datetime
import base64
import numpy as np


app = Flask(__name__)
# bootstrap = Bootstrap(app)
camera = cv2.VideoCapture(0)
headers = {"Content-type":"application/json"}

def return_img_stream(img_local_path):
    import base64
    img_stream = ''
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()
    return img_stream
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

def request_img(frame):
    size = [frame.shape[1], frame.shape[0]]
    img_bytes = frame.tobytes()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    now = datetime.datetime.now()
    body = {"images": img_b64, "size": size}
    # Server: 101.35.252.209
    # local : 127.0.0.1
    # res = requests.post("http://101.35.252.209:5000/detect", headers=headers, data=json.dumps(body))
    res = predicta(frame)
    fin = datetime.datetime.now()
    print(res, "is the result, consume:", (fin-now).microseconds)
    return res

def gen_frames1():
    # d = 10
    while camera.isOpened():
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            res = request_img(frame)
            # 0,255,255 yellow; 255,255,0 green;255,0,255 red;0,0,255 red;
            if res == "no driver":
                cv2.putText(frame, res, (25, 50), font, 0.7, (255, 255, 0), 2, cv2.LINE_4)
            if res == "face":
                cv2.putText(frame, "detect", (25, 50), font, 0.7, (255, 255, 255), 2, cv2.LINE_4) # ye
            if res == "drink":
                info[0] += 1
                cv2.putText(frame, res, (25, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_4)
            if res == "smoke":
                info[2] += 1
                cv2.putText(frame, res, (25, 50), font, 0.7, (0, 0, 255), 2, cv2.LINE_4)
            if res == "phone":
                info[1] += 1
                cv2.putText(frame, res, (25, 50), font, 0.7, (0,255, 255), 2, cv2.LINE_4)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


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
    return render_template("cool_form.html", img_stream1=login_img1,img_stream3=login_img2,img_stream0=img_logo,img_steam2=login_img2)

@app.route("/login",methods = ["GET", "POST"])
def driver_login():
    hint = "Please Login"
    return render_template("login.html", hint=hint, img_steam2=login_img2,img_stream1=login_img1)
    # return render_template("new_login.html")


@app.route("/get_login", methods = ["POST"])
def driver_login_a():
    print(request.form)
    user_name = request.form["name"]
    user_pw = request.form["password"]
    check_login = check_pw(user_name, user_pw)
    if check_login == USER_NOT_EXIST:
        hint = "User Not Exist! Please check your UserID or Register"
        return render_template("login.html", hint = hint, img_steam2=login_img2,img_stream1=login_img1)
    if check_login == USER_BAD_PASSWORD:
        hint = "User Exists, but WRONG password, please try again"
        return render_template("login.html", hint = hint, img_steam2=login_img2,img_stream1=login_img1)
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
        return render_template("register.html", hint=hint, img_stream1 = img_logo)
    check = legal_pw(user_pw)
    if check != 1:
        hint = check
        return render_template("register.html", hint=hint,img_stream1 = img_logo)
    else:
        pw = sha512(user_pw.encode('utf-8')).hexdigest()
        users.insert_one({"userid": user_name, "password": pw})
        hint = "Successfully Register, please login"
        return render_template("login.html", hint=hint, img_steam2=login_img2,img_stream1=login_img1)




@app.route("/stop", methods=["POST", "GET"])
def stop():
    if "stop" in request.form:
        infoa= [0,0,0,0]
        # cv2.destroyWindow(window_name)
        infoa[1] = info[1]
        infoa[2] = info[2]
        infoa[0] = info[0]
        print(infoa)
        if info[0] > 0 or info[1] > 0 or info[2] > 0:
            if (info[0]+info[1]+info[2]) > 15:
                info[3] = "Dangerous Driving. Please Concentrate on driving."
            else:
                info[3] = "Slightly Dangerous Driving. Please be more serious."
        else:
            if info[2] < 10 and info[1]<=2 and info[0]<=2:
                info[3] = "Driving Safely"
        info[0] = 0
        info[1] = 0
        info[2] = 0
        print(info)
        return render_template("numbers.html", d=infoa[0], p=infoa[1], s=infoa[2], judge=info[3])

@app.route("/video_", methods=["POST", "GET"])
def video_home():
    return render_template("video_d.html")


@app.route("/video_d", methods=["POST", "GET"])
def video_detect():
    return Response(gen_frames1(), mimetype='multipart/x-mixed-replace; boundary=frame')

print("Connect to mongoDB")
mongodbUri = 'mongodb://user2:myadmin@IP:27017/admin'
# IP, user, password all should be replaced by your own
# user1，myadmin
# user2，myadmin
# user3，myadmin
client = MongoClient(mongodbUri)
mydb = client['images']
users = mydb["users"]
font = cv2.FONT_HERSHEY_SIMPLEX
img_p1 = "static/img/robot.jpg"
img_p2 = "static/img/human.jpg"
login_img1 = return_img_stream(img_p1)
login_img2 = return_img_stream(img_p2)
img_logo =return_img_stream("static/t1.jpg")
info = [0,0,0,""]



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



app.run()

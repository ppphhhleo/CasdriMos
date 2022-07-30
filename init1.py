from flask import Flask, request, url_for, redirect, render_template,Response
import cv2
from hashlib import sha512
from pymongo import MongoClient
import requests
import json
import datetime
import base64
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
def request_img(frame):
    size = [frame.shape[1], frame.shape[0]]
    img_bytes = frame.tobytes()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    now = datetime.datetime.now()
    body = {"images": img_b64, "size": size}
    # Server: your server IP
    # local : 127.0.0.1
    # ServerIP and Port should be replaced by yours
    res = requests.post("http://ServerIP:Port/detect", headers=headers, data=json.dumps(body))
    fin = datetime.datetime.now()
    print(res.text, "is the result, consume:", (fin-now).microseconds)
    return res.text

def gen_frames():

    while camera.isOpened():
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            res = "detect"
            cv2.putText(frame, res, (25, 50), font, 0.7, (255,255, 255), 2, cv2.LINE_4)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

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
mongodbUri = 'mongodb://user2:myadmin@ip:27017/admin'
# ip, user, password all should be replaced by yours
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
app.run()
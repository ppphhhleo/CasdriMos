import RPi.GPIO as GPIO
from pymongo import MongoClient
import time
import datetime
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)


class pi():
    def get_response(self, check=0):
        while (True):
            query_res = respo.find({}, {"_id": 0, "timestamp": 1, "response": 1}).sort("timestamp", -1).limit(1)
            try:
                tmp_res = query_res[0]["response"]
            except:
                time.sleep(5)
                print("find no response")
                continue

            if tmp_res == "":
                continue
            tmp_res = tmp_res[0]
            global his_res
            if his_res == tmp_res:
                time.sleep(1)
                continue
            else:
                his_res = tmp_res

            if (check != 0):
                thetime = datetime.datetime.utcnow()
                nowt = datetime.datetime.strptime(tmp_res["timestamp"], '%d:%m:%Y:%H:%M:%S')
                if ((thetime - nowt).seconds > check):
                    continue
                else:
                    self.actuators(tmp_res)
                    time.sleep(3)
            else:
                self.actuators(tmp_res)
                time.sleep(3)


    def get_actuators(self,check=0):
        while(True):
            q_a = actuators.find({}, {"_id":0}).sort("timestamp", -1).limit(1)
            resp = q_a[0]

            if (check != 0):
                thetime = datetime.datetime.utcnow()
                nowt = datetime.datetime.strptime(resp["timestamp"], '%d:%m:%Y:%H:%M:%S')
                if ((thetime - nowt).seconds > check):
                    continue
                else:
                    self.actuators(resp["res"])
            else:
                self.actuators(resp["res"])
            time.sleep(3)

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
        if res == "no driver":
            self.setup_a(0,0,0)
        if res == "face" :
            self.setup_a(0,0,0)
        if res == "smoke":
            self.setup_a(1, 0, 0)
        if res == "drink":
            self.setup_a(1, 1, 0)
        if res == "phone":
            self.setup_a(0, 0, 1)

    def run1(self,c):
        self.get_response(check=c)
    def run2(self,c):
        self.get_actuators(check=c)


if __name__ == "__main__":
    print("Connect to mongoDB")
    mongodbUri = 'mongodb://user3:myadmin@101.35.252.209:27017/admin'
    # user1，myadmin
    # user2，myadmin
    # user3，myadmin
    client = MongoClient(mongodbUri)
    mydb = client['images']
    pictures = mydb["pictures"]
    respo = mydb["responses"]
    actuators = mydb["actuators"]

    print("Successfully connect to mongoDB")
    his_res = ""
    pi_c = pi()
    # pi_c.run1(c=0)   # response db, c to check time delta
    pi_c.run2(c=0)  # actuators db
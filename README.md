# CasdriMos

**CasdriMos is a safe-driving monitor system.**

# Quick Start
* **python loc.py**
  
  
> **loc.py**, detects by local camera and local model.  
> init1.py, detects by local camera and server model.   
> main.py, with UI.
> ./Raspberry PI, run on the PI, to get response and make actions  
> ./Server, run on your Server, to process the images

# Overview
## Workflow
CasdriMos is consisted by an Arduino board and actuators like RGB bulb, LCD screen and buzzer.
WeMos Camera captures pictures, and PI sends the pictures as POST request to the server. 
Server processes the frames, recognizes the driving behaviors with YOLOv5, and sends results back to PI, which
makes corresponding actions.

![Overview2](https://raw.githubusercontent.com/ppphhhleo/CasdriMos/main/media/overview2.jpg)

## Details
The system contains Model and Hardware.
For the model, we use YOLOv5 to recognize 5 driving situations: No Driver, Safe, Smoke, Drink, Phone. 

![Overview1](https://raw.githubusercontent.com/ppphhhleo/CasdriMos/main/media/overview1.jpg)

## Web
System implements the user's login and registration functions aiming to provide a more personalized 
and detailed database for the business platform in the future. Drivers' driving history will be recorded and evaluated, which can
supervises drivers and protect passengers.

![Overview3](https://raw.githubusercontent.com/ppphhhleo/CasdriMos/main/media/overview3.jpg)

# To Do
* Server. Response, stress, threads.

## References
[YOLOv5+Deepsort](https://github.com/JingyibySUTsoftware/Yolov5-deepsort-driverDistracted-driving-behavior-detection)
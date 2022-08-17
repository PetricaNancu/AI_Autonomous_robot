import sys
import os
import cv2
import time
import re
import numpy as np
from openvino.inference_engine import IECore
import RPi.GPIO as GPIO          
from time import sleep

model ="./TFOD_35_adam/FP16/saved_model.xml"
device ="MYRIAD"
input_stream = 0
CAMERA_HEIGHT = 480

in1 = 23
in2 = 24
en1 = 25
p1=0
p2=0

in3 = 22
in4 = 27
en2 = 17

turned = 0
speed = 0


GPIO.setmode(GPIO.BCM)
GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(en1,GPIO.OUT)
GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)
p1=GPIO.PWM(en1,1000)

GPIO.setup(in3,GPIO.OUT)
GPIO.setup(in4,GPIO.OUT)
GPIO.setup(en2,GPIO.OUT)
GPIO.output(in3,GPIO.LOW)
GPIO.output(in4,GPIO.LOW)
p2=GPIO.PWM(en2,1000)

p1.start(25)
p2.start(17)

def set_speed(speed):

    p1.ChangeDutyCycle(speed)
    p2.ChangeDutyCycle(speed)


def turn_left():
    print("turning left")
    set_speed(80)
    
    GPIO.output(in4,GPIO.HIGH)
    GPIO.output(in3,GPIO.LOW)
    GPIO.output(in1,GPIO.HIGH)
    GPIO.output(in2,GPIO.LOW)

    time.sleep(1.1)
    #GPIO.output(in1,GPIO.LOW)
    #GPIO.output(in2,GPIO.HIGH)
    
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.LOW)
    GPIO.output(in3,GPIO.LOW)
    GPIO.output(in4,GPIO.LOW)
    
    global turned
    turned = 1
    
    
def turn_right():
    print("turning right")
    set_speed(80)
    
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.HIGH)
    GPIO.output(in3,GPIO.HIGH)
    GPIO.output(in4,GPIO.LOW)
    
    time.sleep(1.1)
    #GPIO.output(in4,GPIO.HIGH)
    #GPIO.output(in3,GPIO.LOW)
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.LOW)
    GPIO.output(in3,GPIO.LOW)
    GPIO.output(in4,GPIO.LOW)
    
    global turned
    turned = 2
    
def move_forward():
    print("forward")
    set_speed(80)
    global speed
    speed = 80
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.HIGH)
    GPIO.output(in3,GPIO.LOW)
    GPIO.output(in4,GPIO.HIGH)


def move_stop():
    print("stop")
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.LOW)
    GPIO.output(in3,GPIO.LOW)
    GPIO.output(in4,GPIO.LOW)
    
    global speed
    speed = 0
    
def speed_low():
    print("low")
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.HIGH)
    GPIO.output(in3,GPIO.LOW)
    GPIO.output(in4,GPIO.HIGH)
    p1.ChangeDutyCycle(60)
    p2.ChangeDutyCycle(60)
    time.sleep(0.5)
    p1.ChangeDutyCycle(30)
    p2.ChangeDutyCycle(30)
    
    global speed
    speed = 30

def speed_medium():
    print("medium")
    p1.ChangeDutyCycle(60)
    p2.ChangeDutyCycle(60)
    
    global speed
    speed = 75

def speed_high():
    print("high")
    p1.ChangeDutyCycle(80)
    p2.ChangeDutyCycle(80)
    
    global speed
    speed = 80
    
def load_labels(path='labels.txt'):
  """Citirea tagurilor fiecarei clase. Suporta fisiere cu sau fara numar de index """
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels

def main():
    
    global turned
    global speed
    nr_frames = 0
    false_detection = 0
    
    labels = load_labels()
    model_xml = model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)
    img_info_input_blob = None
    feed_dict = {}
    for blob_name in net.inputs:
        if len(net.inputs[blob_name].shape) == 4:
            input_blob = blob_name
        elif len(net.inputs[blob_name].shape) == 2:
            img_info_input_blob = blob_name
        else:
            raise RuntimeError("Layer nesuportat de retea {}D  '{}'. Doar 2D and 4D input layers sunt suportate"
                               .format(len(net.inputs[blob_name].shape), blob_name))
    assert len(net.outputs) == 1, "SUporta o singura topologie"
    out_blob = next(iter(net.outputs))
    exec_net = ie.load_network(network=net, num_requests=2, device_name=device)
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob].shape
    if img_info_input_blob:
        feed_dict[img_info_input_blob] = [h, w, 1]
    cap = cv2.VideoCapture(input_stream)
    assert cap.isOpened(), "Can't open " + input_stream
    cur_request_id = 0
    prev_frame_time = 0
    new_frame_time = 0

    while cap.isOpened():
        
        new_frame_time = time.time()
        
        ret, frame = cap.read()
        nr_frames +=1
        
        if nr_frames >20 :
            turned = 0
            nr_frames = 0
            
        false_detection += 1
        
        if ret:
            frame_h, frame_w = frame.shape[:2]
        
        in_frame = cv2.resize(frame, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))
        
        feed_dict[input_blob] = in_frame
        exec_net.start_async(request_id=cur_request_id, inputs=feed_dict)
        if exec_net.requests[cur_request_id].wait(-1) == 0:

            res = exec_net.requests[cur_request_id].outputs[out_blob]
            for obj in res[0][0]: 
                    
                if obj[2] > 0.75 and labels[int(obj[1])-1] == "stanga" and turned != 1 :
                    xmin = int(obj[3] * frame_w)
                    ymin = int(obj[4] * frame_h)
                    xmax = int(obj[5] * frame_w)
                    ymax = int(obj[6] * frame_h)
                    size = ((xmax-xmin)/160)
                    cv2.rectangle(frame,(xmin,ymin), (xmax,ymax) , (21,234,164),2)  
                    cv2.putText(frame,labels[int(obj[1])-1]+ "- " +str(obj[2]),(xmin, min(ymax+20, CAMERA_HEIGHT-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA)
                    turn_left()
                    
                    if speed > 0:
                    
                        move_forward()
                    
                elif obj[2] > 0.70 and labels[int(obj[1])-1] == "dreapta" and turned != 2 :
                    xmin = int(obj[3] * frame_w)
                    ymin = int(obj[4] * frame_h)
                    xmax = int(obj[5] * frame_w)
                    ymax = int(obj[6] * frame_h)
                    size = ((xmax-xmin)/160)
                    cv2.rectangle(frame,(xmin,ymin), (xmax,ymax) , (21,234,164),2)  
                    cv2.putText(frame,labels[int(obj[1])-1]+ "- " +str(obj[2]),(xmin, min(ymax+20, CAMERA_HEIGHT-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA)
                    turn_right()
                    if speed > 0:
                    
                        move_forward()
                    
                elif obj[2] > 0.98 and labels[int(obj[1])-1] == "inainte":
                    xmin = int(obj[3] * frame_w)
                    ymin = int(obj[4] * frame_h)
                    xmax = int(obj[5] * frame_w)
                    ymax = int(obj[6] * frame_h)
                    size = ((xmax-xmin)/160)
                    cv2.rectangle(frame,(xmin,ymin), (xmax,ymax) , (21,234,164),2)  
                    cv2.putText(frame,labels[int(obj[1])-1]+ "- " +str(obj[2]),(xmin, min(ymax+20, CAMERA_HEIGHT-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA)
                    turned = 0
                    move_forward()
                    
                elif obj[2] > 0.85 and labels[int(obj[1])-1] == "stop":
                    xmin = int(obj[3] * frame_w)
                    ymin = int(obj[4] * frame_h)
                    xmax = int(obj[5] * frame_w)
                    ymax = int(obj[6] * frame_h)
                    size = ((xmax-xmin)/160)
                    cv2.rectangle(frame,(xmin,ymin), (xmax,ymax) , (21,234,164),2)  
                    cv2.putText(frame,labels[int(obj[1])-1]+ "- " +str(obj[2]),(xmin, min(ymax+20, CAMERA_HEIGHT-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA)
                    turned = 0
                    move_stop()
                elif obj[2] > 0.75 and labels[int(obj[1])-1] == "interzis_ambele_sensuri":
                    xmin = int(obj[3] * frame_w)
                    ymin = int(obj[4] * frame_h)
                    xmax = int(obj[5] * frame_w)
                    ymax = int(obj[6] * frame_h)
                    size = ((xmax-xmin)/160)
                    cv2.rectangle(frame,(xmin,ymin), (xmax,ymax) , (21,234,164),2)  
                    cv2.putText(frame,labels[int(obj[1])-1]+ "- " +str(obj[2]),(xmin, min(ymax+20, CAMERA_HEIGHT-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA)
                    turned = 0
                    move_stop()
                    
                elif obj[2] > 0.65 and labels[int(obj[1])-1] == "limitare_50":
                    xmin = int(obj[3] * frame_w)
                    ymin = int(obj[4] * frame_h)
                    xmax = int(obj[5] * frame_w)
                    ymax = int(obj[6] * frame_h)
                    size = ((xmax-xmin)/160)
                    cv2.rectangle(frame,(xmin,ymin), (xmax,ymax) , (21,234,164),2)  
                    cv2.putText(frame,labels[int(obj[1])-1]+ "- " +str(obj[2]),(xmin, min(ymax+20, CAMERA_HEIGHT-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA)
                    turned = 0
                    speed_low()
                    
                elif obj[2] > 0.65 and labels[int(obj[1])-1] == "limitare_100":
                    xmin = int(obj[3] * frame_w)
                    ymin = int(obj[4] * frame_h)
                    xmax = int(obj[5] * frame_w)
                    ymax = int(obj[6] * frame_h)
                    size = ((xmax-xmin)/160)
                    cv2.rectangle(frame,(xmin,ymin), (xmax,ymax) , (21,234,164),2)  
                    cv2.putText(frame,labels[int(obj[1])-1]+ "- " +str(obj[2]),(xmin, min(ymax+20, CAMERA_HEIGHT-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA)
                    turned = 0
                    speed_low()
            
            # Draw performance stats

            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time

            cv2.putText(frame,'FPS: {0:.2f}'.format(fps),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)


        cv2.imshow("Nancu_Petrica", frame)
        key = cv2.waitKey(1)
        if key == 27:
            GPIO.cleanup()
            break
    cv2.destroyAllWindows()
if __name__ == '__main__':
    sys.exit(main() or 0)



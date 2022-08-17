import re
import cv2
import time
from tflite_runtime.interpreter import Interpreter
import numpy as np
import argparse


CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

def load_labels(path='./TFOD_35/labels.txt'):
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

def set_input_tensor(interpreter, image):

  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = np.expand_dims((image-255)/255, axis=0)


def get_output_tensor(interpreter, index):

  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):

  set_input_tensor(interpreter, image)
  interpreter.invoke()
  # Get all output details
  
  scores = get_output_tensor(interpreter, 0)
  #print(boxes)
  boxes = get_output_tensor(interpreter, 1)
  #print(classes)
  count = get_output_tensor(interpreter, 2)
  
  classes = np.asarray(get_output_tensor(interpreter, 3)).astype(int)
  #count = get_output_tensor(interpreter, 3)
  #print(scores)

  
  

  
  results = []
  for i in range(int(count)):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
      #print(result)
  return results

def main():
    labels = load_labels()
    interpreter = Interpreter('./TFOD_35/detect.tflite')
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    
    
    prev_frame_time = 0
    new_frame_time = 0

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        
        ret, frame = cap.read()
        new_frame_time = time.time()
        
        img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320,320))
        res = detect_objects(interpreter, img, 0.7)
        #print(res)

        for result in res:
            ymin, xmin, ymax, xmax = result['bounding_box']
            xmin = int(max(1,xmin * CAMERA_WIDTH))
            xmax = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
            ymin = int(max(1, ymin * CAMERA_HEIGHT))
            ymax = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))
            
            cv2.rectangle(frame,(xmin, ymin),(xmax, ymax),(0,255,0),3)
            cv2.putText(frame,labels[int(result['class_id'])],(xmin, min(ymax+20, CAMERA_HEIGHT-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA)
            
        
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        
        cv2.putText(frame,'FPS: {0:.2f}'.format(fps),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        cv2.imshow('Object_detection_Nancu', frame)      
        

        if cv2.waitKey(10) & 0xFF ==ord('q'):
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

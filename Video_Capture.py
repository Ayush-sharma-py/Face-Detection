import cv2 
from PIL import Image
import numpy
import random
import Main

vid = cv2.VideoCapture(0) 


save_directory = Main.main_path +  "//" + "predictions"

while(True): 
    ret, frame = vid.read() 
    cv2.imshow('Face Recognition', frame) 

    if cv2.waitKey(1) & 0xFF == ord('t'):
        n = Image.fromarray(frame,"RGB")
        temp_name = random.randint(10000,1000000)
        n.save(save_directory + f"\\{temp_name}.png")

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
vid.release() 
cv2.destroyAllWindows() 
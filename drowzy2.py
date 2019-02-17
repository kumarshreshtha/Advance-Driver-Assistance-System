import cv2
import dlib
import time
from scipy.spatial import distance as dist

def compute_ear(coord,disp=0):
    
    p_26=dist.euclidean((coord.part(41+disp).x,coord.part(41+disp).y),(coord.part(37+disp).x,coord.part(37+disp).y))
    p_35=dist.euclidean((coord.part(40+disp).x,coord.part(40+disp).y),(coord.part(38+disp).x,coord.part(38+disp).y))
    p_14=dist.euclidean((coord.part(39+disp).x,coord.part(39+disp).y),(coord.part(36+disp).x,coord.part(36+disp).y))
	
    return (p_26+p_35)/(2.0*p_14)


face_detector = dlib.get_frontal_face_detector()
landmarks_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

ear_threshold=0.23
perclos_threshold=48

frame_count=0
no_of_frames=0
blink_count=0

time_start=time.time()
blink_time_start=time_start
eye_close=False
while True:
	
    no_of_frames+=1
    
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
    faces = face_detector(gray, 0)
    
    for face in faces:

        sp = landmarks_predictor(gray, face)
		
		
        av_ear=(compute_ear(sp,0) + compute_ear(sp,6))/2.0
		
        cv2.rectangle(frame,(face.left(),face.top()),(face.right(),face.bottom()),(255,0,0),2)
        cv2.line(frame,(sp.part(42).x,sp.part(42).y),(sp.part(45).x,sp.part(45).y),(0,0,255),2)
        cv2.line(frame,(sp.part(36).x,sp.part(36).y),(sp.part(39).x,sp.part(39).y),(0,0,255),2)
        cv2.line(frame,(int((sp.part(41).x+sp.part(40).x)/2),int((sp.part(41).y+sp.part(40).y)/2)),(int((sp.part(37).x+sp.part(38).x)/2),int((sp.part(37).y+sp.part(38).y)/2)),(0,0,255),2)
        cv2.line(frame,(int((sp.part(47).x+sp.part(46).x)/2),int((sp.part(47).y+sp.part(46).y)/2)),(int((sp.part(43).x+sp.part(44).x)/2),int((sp.part(43).y+sp.part(44).y)/2)),(0,0,255),2)
		
        if av_ear <= ear_threshold:
            frame_count+=1
            if eye_close==False:
                blink_count+=1
                eye_close=True
            if frame_count>=perclos_threshold:
                cv2.putText(frame, "WARNING!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            frame_count=0
            eye_close=False

        
        cv2.putText(frame, "Eye Aspect Ratio: {:.2f}".format(av_ear), (350, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
        cv2.putText(frame, "PERCLOS: {:d}".format(frame_count), (350, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        cv2.putText(frame, "Blink Frequnecy: {:d}".format(blink_count), (350, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1)
    
    time_end=time.time()
    if time_end-blink_time_start>=30:
        blink_count=0
        blink_time_start=time.time()
    cv2.putText(frame, "FPS: {:.2f}".format(no_of_frames/(time_end-time_start)), (350, 120),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1)
    cv2.imshow("Frame", frame)
	
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

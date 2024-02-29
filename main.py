import cv2
import HandTrackingModule as htm
import numpy as np
detector=htm.handDetector()  #calling the function from other file(handtrackingmodule)
import time
draw_color=(255,255,255)     
img_canvas = np.zeros((720,1280,3),np.uint8)
previous_time =0

video_cap=cv2.VideoCapture(0)
while True:
    success,frame=video_cap.read()
    frame=cv2.resize(frame,(1280,720))
    frame=cv2.flip(frame,1)    #1-->horizontal flip      -1---->vertical flip
    cv2.rectangle(frame,pt1=(20,10),pt2=(230,100),color=(0,0,255),thickness=-3)
    cv2.rectangle(frame,pt1=(240,10),pt2=(460,100),color=(0,255,0),thickness=-3)
    cv2.rectangle(frame,pt1=(470,10),pt2=(690,100),color=(255,0,0),thickness=-3)
    cv2.rectangle(frame,pt1=(700,10),pt2=(920,100),color=(0,255,255),thickness=-3)
    cv2.rectangle(frame,pt1=(930,10),pt2=(1270,100),color=(255,255,255),thickness=-3)
    cv2.putText(frame,'ERASER',(1040,70),fontScale=1,fontFace=cv2.FONT_HERSHEY_COMPLEX,color=(0,0,0),thickness=3)

    frame=detector.findHands(frame,draw=True)  #called findhands function from detector defined at beginning
    lmlist=detector.findPosition(frame)   #called findposition function
    #print(lmlist)
    
    if len(lmlist)!=0:
        x1,y1=lmlist[8][1:]             #8th landmark(index finger) from file handlandmarks
        x2,y2=lmlist[12][1:]              #12th landmark(middle finger) from file handlandmarks
        #print(x1,y1)

        fingers=detector.fingersUp()  #to check finger up or down , calling the fingersup function from detector, up=1 and down=0
        #print(fingers)

        #selection mode(when index and middle finger is up)
        if fingers[1] & fingers[2]:  #means if fing1 and fing2 =1, default is 1 , if its zero we have to give ==0
            #print('selection mode')

            xp,yp=0,0   #origin

            if y1<100:
                if 10<x1<230:
                    draw_color=(0,0,255)
                    #print('red')
                elif 240<x1<460:
                    draw_color=(0,255,0)
                    #print('green')
                elif 470<x1<690:
                    draw_color=(255,0,0)
                    #print('blue')
                elif 700<x1<920:
                    draw_color=(0,255,255)
                    #print('yellow')
                elif 930<x1<1270:
                    draw_color=(0,0,0)
                    #print('eraser')


            cv2.rectangle(frame,(x1,y1),(x2,y2),color=draw_color,thickness=-3)  #to draw rectangle
            cv2.putText(frame,'Selection Mode',(1000,650),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=2,color=(255,255,0),thickness=3)

        if (fingers[1] and not fingers[2]):  #if only index finger is up and middle finger is 0/down
        
            #print('drawing mode')
            cv2.putText(frame,'Drawing Mode',(1000,650),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=2,color=(255,255,0),thickness=3)
            cv2.circle(frame,(x1,y1),15,color=draw_color,thickness=-1)                  #x1,y1-->index finger
            
            if xp==0 and yp==0:  #origin points where the drawing starts/previous points.

                xp=x1  #x1 and y1 are the next points, so we draw line from xp,yp to x1,y1, this process continues for whole drawing
                yp=y1

            if draw_color==(0,0,0):  #for eraser , here we can adjust thickness to erase lines faster, we can increase thickness
                cv2.line(frame,(xp,yp),(x1,y1),color=draw_color,thickness=50)
                cv2.line(img_canvas,(xp,yp),(x1,y1),color=draw_color,thickness=50)

            else:  #drawing with colours
                cv2.line(frame,(xp,yp),(x1,y1),color=draw_color,thickness=10)
                cv2.line(img_canvas,(xp,yp),(x1,y1),color=draw_color,thickness=10)
        
    
            xp,yp =x1,y1    #loop - continues to detect xp,yp,x1,y2 for drawing
        
    #merging 2 canvas

    img_gray = cv2.cvtColor(img_canvas,cv2.COLOR_BGR2GRAY)    #coverting to grey

    #from gray scale converted to inverse image by applying threshold
    thresh,img_inv = cv2.threshold(img_gray,20,255,cv2.THRESH_BINARY_INV)

    img_inv = cv2.cvtColor(img_inv,cv2.COLOR_GRAY2BGR)

    # AND operation
    frame = cv2.bitwise_and(frame,img_inv)
    frame = cv2.bitwise_or(frame,img_canvas)   #here the frame inside bracket is the output of and operation and real img_canvas we defined at beg of prgrm
    frame = cv2.addWeighted(frame,1,img_canvas,0.5,0) #1-weight of frst image,0.5 weight of second img and 0 is a constant, these three are known alpha,beta,gamma

    #calculate fpd
    c_time=time.time()  #to get current time
    fps = 1/(c_time-previous_time )    #frame per second
    previous_time=c_time   #then c_time will become previous time and so on like a loop
    cv2.putText(frame,str(int(fps)),(50,150),cv2.FONT_HERSHEY_COMPLEX,5,(0,255,0),thickness=4)

    cv2.imshow('Virtual Painter',frame)
    #cv2.imshow('Canvas',img_canvas)
    #cv2.imshow('grey',img_inv)
    if cv2.waitKey(1) & 0xFF==27:
        break
video_cap.release()
cv2.destroyAllWindows()
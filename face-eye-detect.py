import cv2

cap=cv2.VideoCapture("face-video.mp4")
insan=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
goz=cv2.CascadeClassifier("haarcascade_eye.xml")
while True:
    ret,frame=cap.read()
    gri=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    insanlar=insan.detectMultiScale(gri,1.2,4)
    gozler = goz.detectMultiScale(gri, 1.2, 4)
    
    for x,y,w,h in insanlar:
     cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
     cv2.namedWindow("resimm",cv2.WINDOW_NORMAL)
     cv2.imshow("resimm",frame)

    #
    for x,y,w,h in gozler:
      cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
      cv2.namedWindow("resimm",cv2.WINDOW_NORMAL)
      cv2.imshow("resimm",frame)
    cv2.imshow("resimm",frame)
    if ret==0:
        break

    if cv2.waitKey(20) & 0xFF ==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
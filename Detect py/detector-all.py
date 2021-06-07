#utf-8
import cv2
import numpy

cascadeClassifierPath = 'haarcascade_fullbody.xml' #Full_Body:haarcascade_fullbody.xml Face:'haarcascade_frontalface_alt.xml'
cascadeClassifier = cv2.CascadeClassifier(cascadeClassifierPath) 
cap = cv2.VideoCapture("video.mp4") #name of the video or number(ex: 0 for your cam)

while(cap.isOpened()):
	_, frame = cap.read()
	grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #video on rey
	detectedFaces = cascadeClassifier.detectMultiScale(grayImage,  scaleFactor = 1.2, minNeighbors = 5, minSize = (30, 30)) #face detection

	for(x,y, width, height) in detectedFaces:
		cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 0, 255), 2) #red rectangle
		
	cv2.imshow("Programme de detection", frame) #close program
	if cv2.waitKey(1) == ord('e'):
		break

cap.release()
cv2.destroyAllWindows()
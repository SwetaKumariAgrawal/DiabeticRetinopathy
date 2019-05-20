import cv2
import math
file= open("len.txt","a")
for i in range(45):
	if i<9:
		file_name=str(i+1)+"_bloodvessel.png"
	else:
		file_name=str(i+1)+"_bloodvessel.png"
	image=cv2.imread("bloodVessels/"+file_name,cv2.CV_8UC1)
	contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	img = cv2.drawContours(image, contours, -1, (255,0,0), 1)
	perimeter=0
	for cnt in contours:
		perimeter = perimeter+cv2.arcLength(cnt,True)
	perimeter=round(perimeter,2)
	file.write(str(perimeter)+",")
file.close()
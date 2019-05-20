import cv2
import math
file= open("area.txt","a")
for i in range(45):
	if i<9:
		file_name="ex_im000"+str(i+1)+".jpg"
	else:
		file_name="ex_im00"+str(i+1)+".jpg"
	image=cv2.imread("exudates/"+file_name,cv2.CV_8UC1)
	contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	area=0
	for cnt in contours:
		area = cv2.contourArea(cnt)
	file.write(str(area)+",")
file.close()

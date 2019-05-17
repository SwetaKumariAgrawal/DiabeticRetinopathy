import cv2
import numpy as np
def adjust_gamma(image, gamma=1.0):

   
   table = np.array([((i / 255.0) ** gamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)
def extract_ma(image):
    r,g,b=cv2.split(image)
    comp=255-g
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    histe=clahe.apply(comp)
    adjustImage = adjust_gamma(histe,gamma=3)
    comp = 255-adjustImage
    J =  adjust_gamma(comp,gamma=4)
    J = 255-J
    J = adjust_gamma(J,gamma=4)
    
    K=np.ones((11,11),np.float32)
    L = cv2.filter2D(J,-1,K)
    
    ret3,thresh2 = cv2.threshold(L,125,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    kernel2=np.ones((9,9),np.uint8)
    tophat = cv2.morphologyEx(thresh2, cv2.MORPH_TOPHAT, kernel2)
    kernel3=np.ones((7,7),np.uint8)
    opening = cv2.morphologyEx(tophat, cv2.MORPH_OPEN, kernel3)
    return opening




if __name__ == "__main__":
    for i in range(45):
        if(i<9):
            img_name="im000"+str(i+1)
        else:
            img_name="im00"+str(i+1)
        fundus = cv2.imread("images/"+img_name+".ppm")
        bloodvessel = extract_ma(fundus)
        kernel=np.ones((3,3),np.uint8)
        #dilation
        dilation = cv2.dilate(bloodvessel,kernel,iterations = 1)
        '''cv2.imshow("img",dilation)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''
        cv2.imwrite("microaneurysms/m_"+img_name+".jpg",dilation)

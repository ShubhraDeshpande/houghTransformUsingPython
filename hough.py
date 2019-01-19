# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 10:09:40 2018

@author: Shubhra Jayant
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
img = cv2.imread('hough.jpg')
image = cv2.imread('hough.jpg',0)

#blur = cv2.blur(img_b,(5,5))

def kernel_y():
	yMatrix = np.zeros((3, 3))
	yMatrix[0,0]= 1
	yMatrix[0,1]= 2
	yMatrix[0,2]= 1
	yMatrix[1,0]= 0
	yMatrix[1,1]= 0
	yMatrix[1,2]= 0
	yMatrix[2,0]= -1
	yMatrix[2,1]= -2
	yMatrix[2,2]= -1

	Gy = conv(yMatrix, image)
	return Gy

def kernel_x():
	xMatrix = np.zeros((3, 3))
	xMatrix[0,0]= 1
	xMatrix[0,1]= 0
	xMatrix[0,2]= -1
	xMatrix[1,0]= 2
	xMatrix[1,1]= 0
	xMatrix[1,2]= -2
	xMatrix[2,0]= 1
	xMatrix[2,1]= 0
	xMatrix[2,2]= -1

	print (xMatrix)

	Gx = conv(xMatrix, image) # Gx = img_conv
	return Gx

def gradeMag(Gx,Gy):
	
	img_edge = np.zeros(Gx.shape)
	img_x_sq = np.zeros(Gx.shape)
	for i in range (0,Gx.shape[0]):
		for j in range(0,Gx.shape[1]):
			img_edge[i][j] = math.sqrt(Gx[i][j]**2 + Gy[i][j]**2)	
			

	return img_edge


def conv(X, img):
	img_conv = np.zeros(img.shape)
	X = X[::-1, ::-1]
	img_h = img.shape[0]
	img_w = img.shape[1]
	print(X)

	X_h = X.shape[0]
	X_w = X.shape[1]

	start_h = 1
	start_w = 1

	r = img_h -1
	s = img_w -1

	for i in range (start_h, r):
		for j in range (start_w, s):
			con = 0
			for m in range(0, X.shape[0]):
				for n in range(0, X.shape[1]):
					con = con +  X[m][n] * img[i-start_h+m][j-start_w+n]
					img_conv[i][j] = con
	return img_conv

Gx = kernel_x()
Gy = kernel_y()
edges= gradeMag(Gx,Gy)
cv2.imwrite('edges.jpg',edges)

e = np.zeros((edges.shape))
for i in range(e.shape[0]):
    for j in range(e.shape[1]):
        if edges[i][j]>120:
            e[i][j] = 255

cv2.imwrite('edges.jpg',e)
edges = e
			





rho=[]
thetaa = []



#rhoSHT = np.array()
def hough_op(img,T):
    theta = np.arange(0,180)
    r,c = img.shape
    rho_l = int(2*(np.ceil(np.sqrt((r-1)**2 + (c-1)**2))))
    rho_len = rho_l #+1
    H = np.zeros((rho_len, len(theta)))
    #print("initial H ",H)
    
    for i in range(r):
        for j in range(c):
            if img[i][j]==255:
                for t in theta:
                    p = j* np.cos((t*(np.pi/180))) + i * np.sin((t*(np.pi/180)))
                    p = round((p))# + (rho_len/2))
                    #print("rho ",p, "for theta ",t)
                    
                    
                    H[int(p)][t] = H[int(p)][t] + 1
                    if H[int(p)][t] >= T:
                        rho.append(int(p))
                        #print("rho ",rho)
                        #a = input("do next")
                        thetaa.append(t)
                        #print("value of H for rho ",int(p),H[int(p)][t])
                        #a = input("do next")
                        
                        #print(rho)
                        
                        
                   
                        
    return (rho,thetaa,H)
T = 170
rho, thetaa, H = hough_op(edges,T)

cv2.imwrite("sigma.jpg",H)
#vot_mat = np.zeros((rho.shape[0],theta.shape[0]))

        
for i in range(len(rho)):
    rh = rho[i]
    theta = np.deg2rad(thetaa[i])
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rh
    y0 = b*rh
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    print("Theta: "+str(theta))
    #if theta> 3.10:
    #    cv2.line(img,(x1,y1),((x2),y2),(0,0,255),1)  # used for inclided vertical lines
    
    if theta < 3.10:
        cv2.line(img,(x1,y1),((x2),y2),(0,0,255),1) #used for detecting vertical lines
        
    
#cv2.imwrite('blue_line.jpg',img)
#cv2.imwrite('red_line.jpg',img)





        
        


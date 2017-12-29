#/usr/bin/python

import cv2
import numpy as np
import math

def Sobel_op(img):
    rows, cols = img.shape
    cp_img = np.empty((rows,cols))
    grad_img = np.empty((rows,cols))

    for i in range(rows-1):
        for j in range(cols-1):
            if(i != 0 and j != 0 ):
                Gx = (img[i-1,j-1]+(2*img[i,j-1])+img[i+1,j-1])-(img[i-1,j+1]+(2*img[i,j+1])+img[i+1,j+1])
                Gy = (img[i-1,j-1]+(2*img[i-1,j])+img[i-1,j+1])-(img[i+1,j-1]+(2*img[i+1,j])+img[i+1,j+1])
            else:
                Gx = 0
                Gy = 0

            cp_img[i,j] = math.sqrt((Gx*Gx)+(Gy*Gy))
            grad_img[i,j] = abs(math.degrees(math.atan2(Gy,Gx)))

    return cp_img, grad_img

def visualise_hog(hog_, img_rows, img_cols, ker_siz ,ker_row, ker_cols):
    rows , cols = hog_.shape
    blank = np.zeros((img_cols, img_rows))
    ghost = np.zeros((img_rows,img_cols))

    for i in range(ker_row):
        for l in range(ker_cols):
            sum = 0
            for k in range(cols):
                sum += hog_[(i*ker_cols)+l,k]

            for j in range(cols):
                prob_ho = (hog_[(i*ker_cols)+l,j]/sum)*100
                if (prob_ho > 10):
                    if(j == 0):
                        cv2.line(blank,((i*ker_siz)+0,(l*ker_siz)+4),((i*ker_siz)+8,(l*ker_siz)+4),prob_ho)
                    elif(j == 1):
                        cv2.line(blank,((i*ker_siz)+1,(l*ker_siz)+3),((i*ker_siz)+7,(l*ker_siz)+3),prob_ho)
                    elif(j == 2):
                        cv2.line(blank,((i*ker_siz)+2,(l*ker_siz)+2),((i*ker_siz)+6,(l*ker_siz)+6),prob_ho)
                    elif(j == 3):
                        cv2.line(blank,((i*ker_siz)+3,(l*ker_siz)+1),((i*ker_siz)+5,(l*ker_siz)+7),prob_ho)
                    elif(j == 4):
                        cv2.line(blank,((i*ker_siz)+4,(l*ker_siz)+0),((i*ker_siz)+4,(l*ker_siz)+8),prob_ho)
                    elif(j == 5):
                        cv2.line(blank,((i*ker_siz)+5,(l*ker_siz)+1),((i*ker_siz)+3,(l*ker_siz)+7),prob_ho)
                    elif(j == 6):
                        cv2.line(blank,((i*ker_siz)+6,(l*ker_siz)+2),((i*ker_siz)+2,(l*ker_siz)+6),prob_ho)
                    elif(j == 7):
                        cv2.line(blank,((i*ker_siz)+7,(l*ker_siz)+3),((i*ker_siz)+1,(l*ker_siz)+5),prob_ho)
                    elif(j == 8):
                        cv2.line(blank,((i*ker_siz)+8,(l*ker_siz)+4),((i*ker_siz)+0,(l*ker_siz)+4),prob_ho)

    for i in range(img_rows):
        for j in range(img_cols):
            ghost[i,j] = blank[j,i]

    return ghost


def HOG(edges,grad_img):
    rows , cols = edges.shape
    ker_siz = 8
    H_rows = int(rows/ker_siz)
    H_cols = int(cols/ker_siz)

    Hog_val = np.zeros(((H_rows*H_cols),9))
    vis_hog = np.zeros((rows,cols))

    for i in range(H_rows):
        for j in range(H_cols):

            value_0 = 0
            value_20 = 0
            value_40 = 0
            value_60 = 0
            value_80 = 0
            value_100 = 0
            value_120 = 0
            value_140 = 0
            value_160 = 0

            for n in range(ker_siz):
                for m in range(ker_siz):
                    grad_value = grad_img[(i*ker_siz)+n,(j*ker_siz)+m]
                    edges_value = edges[(i*ker_siz)+n,(j*ker_siz)+m]
                    if ( 0 <= grad_value < 20 ):
                        value = grad_value - 0
                        prob_value = (value/20)* edges_value
                        value_20 += prob_value
                        value_0 += edges_value-prob_value
                    elif (20 <= grad_value < 40):
                        value = grad_value - 20
                        prob_value = (value/20)* edges_value
                        value_40 += prob_value
                        value_20 += edges_value-prob_value
                    elif (40 <= grad_value < 60):
                        value = grad_value - 40
                        prob_value = (value/20)* edges_value
                        value_60 += prob_value
                        value_40 += edges_value-prob_value
                    elif (60 <= grad_value < 80):
                        value = grad_value - 60
                        prob_value = (value/20)* edges_value
                        value_80 += prob_value
                        value_60 += edges_value-prob_value
                    elif (80 <= grad_value < 100):
                        value = grad_value - 80
                        prob_value = (value/20)* edges_value
                        value_100 += prob_value
                        value_80 += edges_value-prob_value
                    elif (100 <= grad_value < 120):
                        value = grad_value - 100
                        prob_value = (value/20)* edges_value
                        value_120 += prob_value
                        value_100 += edges_value-prob_value
                    elif (120 <= grad_value < 140):
                        value = grad_value - 120
                        prob_value = (value/20)* edges_value
                        value_140 += prob_value
                        value_120 += edges_value-prob_value
                    elif (140 <= grad_value < 160):
                        value = grad_value - 140
                        prob_value = (value/20)* edges_value
                        value_160 += prob_value
                        value_140 += edges_value-prob_value
                    elif (160 <= grad_value < 180):
                        value = grad_value - 160
                        prob_value = (value/20)* edges_value
                        value_0 += prob_value
                        value_160 += edges_value-prob_value

            Hog_val[(i*H_cols)+j,0] = value_0
            Hog_val[(i*H_cols)+j,1] = value_20
            Hog_val[(i*H_cols)+j,2] = value_40
            Hog_val[(i*H_cols)+j,3] = value_60
            Hog_val[(i*H_cols)+j,4] = value_80
            Hog_val[(i*H_cols)+j,5] = value_100
            Hog_val[(i*H_cols)+j,6] = value_120
            Hog_val[(i*H_cols)+j,7] = value_140
            Hog_val[(i*H_cols)+j,8] = value_160


    vis_hog = visualise_hog(Hog_val,rows,cols,ker_siz,H_rows,H_cols)
    return Hog_val, vis_hog


def main():
    img = cv2.imread("Bikesgray.jpg",0)
    edges, img_grad = Sobel_op(img)
    ret, thresh = cv2.threshold(edges,120,255,cv2.THRESH_BINARY)

    H,v_h = HOG(edges,img_grad)

    cv2.imshow("th",thresh)
    cv2.imwrite("H_v.png", v_h)

    cv2.waitKey(0)
    cv2.destoryAllWindows()


if __name__ == "__main__":
    main()

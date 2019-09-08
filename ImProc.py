
import os
from common import color
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy import interp
from math import pi, tan, cos, sin, degrees

import re
class ImProcess(object):
    def __init__(self, folder):
        self.dirName = folder
        self.ext = '.bmp'
        self.fig = plt.figure()
        self.pnt_i = []
        self.pnt_x = []
        self.pnt_y = []
        self.pnt_r = []
        self.ifFolderExist()
        self.trackBarInitialisation()
        self.BGSubtractor()
        self.ball_positions()
        (pnt_in, pnt_xn, pnt_yn, pnt_rn) = self.interpolation()
        pnt_in = [round(i, 1) for i in linearly_interpolate_nans(pnt_in)]
        self.pnt_i = list(map(int, pnt_in))
        self.pnt_x = list(map(int, linearly_interpolate_nans(pnt_xn)))
        self.pnt_y = list(map(int, linearly_interpolate_nans(pnt_yn)))
        self.pnt_r = [round(i, 2) for i in linearly_interpolate_nans(pnt_rn)]
        self.loadAllPoints('reference.txt')
        # size of pixel in mm
        self.pixel_size = 0.0048
        # focuse length in mm
        self.focuse = 8
        # distance to the first ball in mm
        self.distance = 4*1000
        self.C = []
        self.dP = 37.3
        print("Reference")
        print("i", self.i)
        print(self.xy[:,0], self.xy[:,1])
        print(self.r)
        print("Measured")
        print("i", self.pnt_i)
        print(self.pnt_x, self.pnt_y)
        print(self.pnt_r)
        (U0, V0, Px0, Py0, Pz0, Distance0,  _) = self.firstBall(self.i, self.xy[:,0], self.xy[:,1], self.r)
        (U1, V1, Px1, Py1, Pz1, Distance0_, _) = self.firstBall(self.pnt_i, self.pnt_x, self.pnt_y, self.pnt_r)
        self.error_show(U0,V0, U1,V1, Px0, Py0, Pz0, Px1, Py1, Pz1, Distance0,Distance0_)



    def ball_positions(self):
        def ResizeImage(image, x, y, xsize, ysize):
            original = image.copy()
            resized = original[y:y + ysize, x:x + xsize]
            return (original, resized)
        Calc_enable = False
        K =0
        while K < 4:
            print (color.GREEN+"---------------"+color.END)
            f_num = 0
            self.pnt_i = []
            self.pnt_x = []
            self.pnt_y = []
            self.pnt_r = []

            for f in self.file_name:
                f_num = f_num +1
                (im_grey, im_grey_resized) = ResizeImage(image=cv2.imread(f, cv2.COLOR_BGR2GRAY), x=0, y=512, xsize=640, ysize=320)
                self.height = im_grey.shape[0]
                self.width = im_grey.shape[1]
                self.offset = [round(self.width / 2) - 1, round(self.height / 2) - 1]
                im_contrasted = self.image_adjust(im_grey_resized)
                MOG2 = self.BG.apply(im_contrasted)
                (opening) = self.morphology(MOG2)
                mask  =self.Contours(image = im_contrasted, mog = opening, f = f)
                print (color.RED + f + color.END)
                if Calc_enable == True:
                    cv2.imshow(self.title_window, mask)
                    cv2.waitKey(100)
                    key = cv2.waitKey(1)
                    if key == 27:
                        break
            K = K +1
            if K  ==  3:
                K = 3
                Calc_enable = True
            else:
                False
        print (color.RED + "ball' positions calculated" + color.END)

    def Contours (self, image, mog, f):
        f_num = int(f[10:-4])
        _, contours, _ = cv2.findContours(mog, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=False)
        image = self.to_rgb(image)
        for c in sorted_contours:
            hull = cv2.convexHull(c)
            area = cv2.contourArea(hull)
            if area > 300:
                cv2.drawContours(image, [hull], -1, (255, 255, 0), 2)
                (xc, yc), radius = cv2.minEnclosingCircle(hull)
                x, y, w, h = cv2.boundingRect(hull)
                rect_area = w * h
                extent = float(area) / rect_area
                if radius > 10 and radius < 25:
                    if extent > 0.65:
                        center = (int(xc), int(yc))
                        radius = int(radius)
                        self.pnt_i.append(f_num)
                        self.pnt_x.append(int(xc))
                        self.pnt_y.append(int(yc)+self.offset[1])
                        self.pnt_r.append(int(radius*2))
                        cv2.circle(image, center, radius, (0,255,0), 2)
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return image
    def to_rgb(self, im):
        w, h = im.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0], ret[:, :, 1], ret[:, :, 2] = im, im, im
        return ret
    def image_adjust(self, im_grey):
        def check_value(N):
            count = N % 2
            if count == 0:N = N+1
            else: N = N
            return N
        g = np.arange(0.1, 2, 0.1)
        self.param['gamma'] = cv2.getTrackbarPos(list(self.param.keys())[0], self.title_window)
        #print(color.GREEN + 'gamma: ' + str(g[self.param['gamma']]) + color.END)
        invGamma = 1.0 / g[self.param['gamma']]
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        self.param['gauss'] = cv2.getTrackbarPos(list(self.param.keys())[1], self.title_window)
        N = check_value(self.param['gauss'])
        output = cv2.GaussianBlur(cv2.LUT(im_grey, table), (N, N), 0)
        return output

    def BGSubtractor(self):
        self.BG = cv2.createBackgroundSubtractorMOG2()
        self.BG.setHistory(15)
        self.param['bg'] = cv2.getTrackbarPos(list(self.param.keys())[2], self.title_window)
        self.BG.setVarThreshold (100)
        self.BG.setDetectShadows (False)

    def morphology(self, mask):
        #closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        opening = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
        #tophat = cv2.morphologyEx(closing, cv2.MORPH_GRADIENT, np.ones((5,5), np.uint8))
        output = opening
        #dilate = cv2.dilate(tophat, np.ones((3, 3), np.uint8), iterations=2)

        return output

    def trackBarInitialisation(self):
        def donothing(x):
            pass
        self.title_window = "Ball"
        self.param = {'gamma': 12,
                      'gauss': 7,
                      'bgsbs': 100,
                      'min_v': 30,
                      'max_v': 70,
                      'min_r': 13,
                      'max_r': 20}
        cv2.namedWindow(self.title_window, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar(list(self.param.keys())[0], self.title_window, self.param['gamma'], 19, donothing)
        cv2.createTrackbar(list(self.param.keys())[1], self.title_window, self.param['gauss'], 9, donothing)
       # cv2.createTrackbar(list(self.param.keys())[2], self.title_window, self.param['bgsbs'], 100, donothing)

    def ifFolderExist(self):
        def loadAllImages(dirName):
            def loadImages(path):
                return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(self.ext)]
            imageNames = (loadImages(path="./" + dirName))
            print(color.GREEN + color.BOLD + "Found " +str(len(imageNames)) +" *"+self.ext +" images" + color.END)
            return sorted(imageNames, reverse=False, key=lambda s: int(s[10:-4]))
        print ("----------------")
        folder = self.dirName
        if os.path.exists(folder) and os.path.isdir(folder):
            if not os.listdir(self.dirName):
                print(color.RED + color.BOLD + "Directory " +self.dirName +" is empty " + color.RED)
            else:
                print(color.GREEN + color.BOLD + "Directory " +self.dirName +" is not empty " + color.END)
                self.file_name = loadAllImages(self.dirName)
        else:
            print(color.RED + color.BOLD + "Given Directory " +self.dirName +" does not exists" + color.END)

############################################



    def interpolation(self):
        pnt_in = []
        pnt_xn = []
        pnt_yn = []
        pnt_rn = []
        for i in range(1,16):
            if i in self.pnt_i:
                pnt_in.append(self.pnt_i[self.pnt_i.index(i)])
                pnt_xn.append(self.pnt_x[self.pnt_i.index(i)])
                pnt_yn.append(self.pnt_y[self.pnt_i.index(i)])
                pnt_rn.append(self.pnt_r[self.pnt_i.index(i)])
            else:
                pnt_in.append(np.nan)
                pnt_xn.append(np.nan)
                pnt_yn.append(np.nan)
                pnt_rn.append(np.nan)

        return(np.asarray(pnt_in),np.asarray(pnt_xn), np.asarray(pnt_yn), np.asarray(pnt_rn))



    def loadAllPoints(self, filename):
        self.i = np.loadtxt(filename, dtype=np.int, usecols=[0])
        self.xy = np.loadtxt(filename, dtype=np.int, usecols=[1,2])
        self.r = np.loadtxt(filename, dtype=np.float64, usecols=[3])

    def firstBall(self, i, x, y, r):
        def CoordP(x, offset, pixel_size, r, focuse, dP, s):
            U = (x - offset)
            #print(s, "X", color.RED + str(x) + color.END, " offset ", color.RED + str(offset) + color.END, color.BLUE + str(U) + color.END,)
            U = round(U*pixel_size,2)
           # print(U)
            dU = ((r/2)*pixel_size)
            # P = dP + P1
            # U = dU + U1
            # dU/U = dP/P -> P = U*dP/dU
            P = int(U * dP / dU)
            U1 = U - dU
            P1 = P- dP
            # U1/f = P1/D -> D = f*P1/U1
            D = int((focuse*P1/U1))
            return (P, D, U)
        def speedVector(Px, Py, Dist):
            Distance = []
            Px = [i * 0.001 for i in Px]
            Py = [i * 0.001 for i in Px]
            Dist = [i * 0.001 for i in Px]
            for i in range (1,len(Px)):
                #print(color.RED+ str(i) + color.END)
                dx = Px[i] - Px[i-1]
                dy = Py[i] -  Py[i-1]
                dz = Dist[i] - Dist[i-1]
                d = round(np.sqrt(dx**2 + dy**2 + dz**2),3)*1000
                Distance.append(d)
            return Distance
        # convert Distance value from m to mm
        #self.distance = self.distance*1000
        #Vector = np.sqrt((self.focuse + self.distance)**2)
        U,V = [],[]
        Px = []
        Py = []
        Py1 = []
        Pz = []
        for pos, val in enumerate(i):
            print ("----")
            (Px_, D_, U_) = CoordP(x=x[pos], offset=self.offset[0], pixel_size=self.pixel_size, r=r[pos],
                                   focuse=self.focuse, dP=self.dP, s="X")

            (Py_, D_, V_) = CoordP(x=y[pos], offset=self.offset[1], pixel_size=self.pixel_size, r=r[pos],
                                   focuse=self.focuse, dP=self.dP, s="Y")

            #(Px_, D_, U_) = CoordP(x = self.xy[pos, 0], offset = self.offset[0], pixel_size = self.pixel_size, r = self.r[pos], focuse = self.focuse, dP = self.dP, s = "X")
            #(Py_, D_, V_) = CoordP(x = self.xy[pos, 1], offset = self.offset[1], pixel_size = self.pixel_size, r = self.r[pos], focuse = self.focuse, dP = self.dP, s = "Y")
            #U = round((self.xy[pos, 0] - self.offset[0]) * self.pixel_size, 4)
            #dU = round((self.r[pos]/2)*self.pixel_size, 4)
            #U1 = U - dU
            # P = dP + P1
            # U = dU + U1
            # dU/U = dP/P -> P = U*dP/dU
            #P = int(U * self.dP / dU)
            # U/f = P/D -> D = f*P/U
            #Dist = self.focuse*P/U
            #self.U.append(U)
            #self.dU.append(dU)
            Py1_ = Py_ + tan(10*pi/180)*D_
            Px.append(Px_)
            Py.append(Py_)
            Py1.append(Py1_)
            Pz.append(D_)
            U.append(U_)
            V.append(V_)
           # print (pos, ":", Px, "Py", Py, "Py1", Py1, U, V, D)
        Distance0 =speedVector(Px = Px, Py = Py, Dist = Pz)
        Distance1 =speedVector(Px = Px, Py = Py1, Dist = Pz)
        print(" dist", Distance0)
        print("dist", Distance1)
        print ("Dist max dist min", Pz)
        return (U, V, Px, Py, Pz, Distance0, Distance1)

    def error_show(self, U0, V0, U1,V1, Px0, Py0, Pz0, Px1, Py1, Pz1, Distance0,Distance0_):
        self.fig.suptitle('Position of balls')
        ax1 = self.fig.add_subplot(2, 2, 1)
        ax2 = self.fig.add_subplot(2, 2, 2)
        ax3 = self.fig.add_subplot(2, 2, 3)
        ax4 = self.fig.add_subplot(2, 2, 4)
        ####
        ax1.plot(U0, V0, marker='o', label='Reference projections')
        ax1.plot(U1, V1, marker='o', label='Measured and interpolated projections')
        ax1.text(U0[0], V0[0], (str(1)), fontsize=10)
        ax1.text(U0[-1], V0[-1], (str(15)), fontsize=10)
        ax1.text(U1[0], V1[0], (str(1)), fontsize=10)
        ax1.text(U1[-1], V1[-1], (str(15)), fontsize=10)
        ###
        ax2.plot(Px0, Py0,  marker='o', label='Reference path')
        ax2.plot(Px1, Py1,  marker='o', label='measured path')
        ax2.text(Px0[0], Py0[0], (str(1)), fontsize=10)
        ax2.text(Px0[-1], Py0[-1], (str(15)), fontsize=10)
        ax2.text(Px1[0], Py1[0], (str(1)), fontsize=10)
        ax2.text(Px1[-1], Py1[-1], (str(15)), fontsize=10)
        ax2.text(-1000, -500, "ATTENTION. Some balls positions measured with a wrong radius", fontsize=10)
        ###
        ax3.plot(Pz0, Px0, marker='o', label='Reference path')
        ax3.plot(Pz1, Px1, marker='o', label='measured path')
        ax3.text(0, 0, "Camera Posittion")
        ####
        avSpeed0 = np.mean(Distance0[1:-2])
        avSpeed1 = np.mean(Distance0_[1:-2])
        ax4.bar(np.arange(14, 0, -1) -0.4, Distance0, width=0.4 , edgecolor='black', label = 'Reference path')
        ax4.bar(np.arange(14, 0, -1), Distance0_ ,width=0.4, edgecolor='black', label='measured path')

        ax4.plot(np.arange(13, 2, -1), [avSpeed0] * 11, label = "Mean velocity "+str(int(avSpeed0)) + "m/s")
        ax4.plot(np.arange(13, 2, -1), [avSpeed1] * 11, label = "Mean velocity "+str(int(avSpeed1))+ "m/s")
        print (color.GREEN + "Velosity : " + str(round(avSpeed0)) + "m/s" + color.END)
        ax1.set_ylim(2.5, -2.5)
        ax1.set_xlim(-3, 3)
        ax1.set_xlabel('U (mm)')
        ax1.set_ylabel('V (mm)')
        ax4.set_xlabel('frame number')
        ax4.set_ylabel('velocity [m/s]')
        ax2.set_ylim(1500, -1000)
        ax2.set_xlim(-1200, 1000)
        ax2.set_xlabel('Pu (mm)')
        ax2.set_ylabel('Pv (mm)')
        ax3.set_ylim(200, -1200)
        ax3.set_xlabel('Pu (mm)')
        ax3.set_ylabel('Pv (mm)')
        ax1.legend(loc='upper right')
        ax1.set_title('Corresponding plane. (Ball positions in  [mm])')
        ax2.set_title('3D Object plane in mm.')
        ax2.legend(loc='upper right')
        ax3.set_title('Corresponding plane. View from the top [mm]')
        ax3.legend(loc='upper right')
        ax4.set_title('Velocity and mean Velocity, [m/c]')
        ax4.legend(loc='upper right')
        ax1.grid(True)
        plt.show()

# borrowed in a web
def linearly_interpolate_nans(y):
   # Fit a linear regression to the non-nan y values

   # Create X matrix for linreg with an intercept and an index
   X = np.vstack((np.ones(len(y)), np.arange(len(y))))

   # Get the non-NaN values of X and y
   X_fit = X[:, ~np.isnan(y)]
   y_fit = y[~np.isnan(y)].reshape(-1, 1)

   # Estimate the coefficients of the linear regression
   beta = np.linalg.lstsq(X_fit.T, y_fit)[0]

   # Fill in all the nan values using the predicted coefficients
   y.flat[np.isnan(y)] = np.dot(X[:, np.isnan(y)].T, beta)
   return y

def mkDir(dirName="Output"):
    try:
        os.mkdir(dirName)
        print(color.GREEN + color.BOLD + "Directory {}: Created".format(
            dirName) + color.END)
    except FileExistsError:
        print(color.BLUE + "Directory {}: ALREADY EXIST".format(dirName) + color.END)
    return dirName
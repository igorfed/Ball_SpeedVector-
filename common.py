import numpy as np


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


np.warnings.filterwarnings('ignore')

def interpolation():
   pnt_i = [3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]
   pnt_x = [547, 481, 450, 416, 379, 340, 296, 204, 149, 92, 31]
   pnt_y = [294, 280, 274, 267, 259, 250, 233, 221, 210, 197, 183]
   pnt_r = [13, 16, 17, 17, 17, 19, 15, 21, 22, 22, 24]
   pnt_in = []
   pnt_xn = []
   pnt_yn = []
   pnt_rn = []
   for i in range(1,16):
      if i in pnt_i:
         pnt_in.append(pnt_i[pnt_i.index(i)])
         pnt_xn.append(pnt_x[pnt_i.index(i)])
         pnt_yn.append(pnt_y[pnt_i.index(i)])
         pnt_rn.append(pnt_r[pnt_i.index(i)])
      else:
         pnt_in.append(np.nan)
         pnt_xn.append(np.nan)
         pnt_yn.append(np.nan)
         pnt_rn.append(np.nan)
   print("pnt_in", pnt_in)
   print("pnt_xn", pnt_xn)
   print("pnt_yn", pnt_yn)
   print("pnt_rn", pnt_rn)
   return(pnt_in,pnt_xn, pnt_yn, pnt_rn)
(pnt_in,pnt_xn, pnt_yn, pnt_rn) = interpolation()
pnt_in = np.asarray(pnt_in)
pnt_xn = np.asarray(pnt_xn)
pnt_yn = np.asarray(pnt_yn)
pnt_rn = np.asarray(pnt_rn)

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


#pnt_i = list(map(int, linearly_interpolate_nans(pnt_in)))
pnt_in = [round(i, 1) for i in linearly_interpolate_nans(pnt_in) ]
pnt_in = list(map(int, pnt_in))
print (pnt_in)
pnt_x = list(map(int, linearly_interpolate_nans(pnt_xn)))
pnt_y = list(map(int, linearly_interpolate_nans(pnt_yn)))
print (pnt_x)
print (pnt_y)
pnt_r = [round(i, 2) for i in linearly_interpolate_nans(pnt_rn) ]
print (pnt_r)

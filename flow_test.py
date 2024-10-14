import matplotlib.pyplot as plt
from scipy.special import erf
import numpy as np
import csv
import pandas as pd
from scipy.optimize import curve_fit
data_line_1 = pd.read_csv('line_1.csv')
data_line_2 = pd.read_csv('line_2.csv')
data_line_3 = pd.read_csv('line_3.csv')
data_line_4 = pd.read_csv('line_4.csv')
width = 500 #um
depth = 75 #um
length = 2723.977 #um
flowrate = 2.5e10/60 #ul/s
um_s = flowrate/(width*depth) #um/s
#lengths measured in imageJ
l_1 = 789 * 10**-6 
l_2 = 1360 * 10**-6 
l_3 = 1838 * 10**-6 
l_4 = 2468 * 10**-6 
#diffusion time t=l/v
t_1 = l_1/um_s
t_2 = l_2/um_s
t_3 = l_3/um_s
t_4 = l_4/um_s
print(t_1, t_2, t_3, t_4)
print(um_s)
#Transpose data from ImageJ
x_1, I_1 = data_line_1.values.T
x_2, I_2 = data_line_2.values.T
x_3, I_3 = data_line_3.values.T
x_4, I_4 = data_line_4.values.T
#Normalisation
def norm(val):
    max_val = sum(val[1:100])/len(val[1:100])
    min_val = sum(val[-100:])/len(val[-100:])
    return [(i - min_val)/(max_val - min_val) for i in val]
#Model to be used to optimalize the parameters D and x_0. One model for each diffusion time
def ficks_1(x, x_0, D):
    return 0.5*(1-erf((x - x_0)/np.sqrt(4*D*t_1)))
def ficks_2(x, x_0, D):
    return 0.5*(1-erf((x - x_0)/np.sqrt(4*D*t_2)))
def ficks_3(x, x_0, D):
    return 0.5*(1-erf((x - x_0)/np.sqrt(4*D*t_3)))
def ficks_4(x, x_0, D):
    return 0.5*(1-erf((x - x_0)/np.sqrt(4*D*t_4)))
#Absorbance
def A(I_max, I):
    return np.log10(I_max/I)
#Diffusion to radius
v = 1e-3 #viscosity Pa*s
T = 293 #Kelvin, 20 degrees celcius in lab
k_b = 1.3806505e-23 #J/K
def SE(D):
    return (k_b*T)/(6*np.pi*v*D)
I_1_max = sum(I_1[1:100])/len(I_1[1:100])
A_1 = [A(I_1_max, i) for i in I_1]
popt, pcov = curve_fit(ficks_1, x_1, norm(A_1))
#Find D, multiply with 10^-12, all good
x_0_1, D_1 = popt
print(D_1*1e-12)
print('Radius=', SE(D_1*1e-12))
plt.rcParams["figure.figsize"] = [10, 3]
plt.plot(x_1, ficks_1(x_1, *popt), 'r', label='fit: x_0=%500f, D=%500f' %tuple(popt)) # D is in,→ um/s^2
plt.scatter(x_1, norm(A_1), s = 4)
plt.xlabel('Cross-section from bottom [\u03BCm]')
plt.ylabel('Relative concentration')
plt.title('Line 1')
plt.show()
I_2_max = sum(I_2[1:100])/len(I_2[1:100])
A_2 = [A(I_2_max, i) for i in I_2]
popt, pcov = curve_fit(ficks_2, x_2, norm(A_2))
#Find D, multiply with 10^-12, all good
x_0_2, D_2 = popt
print(D_2*1e-12)
print('Radius=', SE(D_2*1e-12))
plt.plot(x_2, ficks_2(x_2, *popt), 'r', label='fit: x_0=%500f, D=%500f' %tuple(popt)) # D is in,→ um/s^2
plt.scatter(x_2, norm(A_2), s = 4)
plt.xlabel('Cross-section from bottom [\u03BCm]')
plt.ylabel('Relative concentration')
plt.title('Line 2')
plt.show()

I_3_max = sum(I_3[1:100])/len(I_3[1:100])
A_3 = [A(I_3_max, i) for i in I_3]
popt, pcov = curve_fit(ficks_3, x_3, norm(A_3))
#Find D, multiply with 10^-12, all good
x_0_3, D_3 = popt
print(D_3*1e-12)
print('Radius=', SE(D_3*1e-12))
plt.plot(x_3, ficks_3(x_3, *popt), 'r', label='fit: x_0=%500f, D=%500f' %tuple(popt)) # D is in,→ um/s^2
plt.scatter(x_3, norm(A_3), s = 4)
plt.xlabel('Cross-section from bottom [\u03BCm]')
plt.ylabel('Relative concentration')
plt.title('Line 3')
plt.show()
I_4_max = sum(I_4[1:100])/len(I_4[1:100])
A_4 = [A(I_4_max, i) for i in I_4]
popt, pcov = curve_fit(ficks_4, x_4, norm(A_4))
#Find D, multiply with 10^-12, all good
x_0_4, D_4 = popt
print(D_4*1e-12)
print(D_4)
print('Radius=', SE(D_4*1e-12))
plt.plot(x_4, ficks_4(x_4, *popt), 'r', label='fit: x_0=%500f, D=%500f' %tuple(popt)) # D is in,→ um/s^2
plt.scatter(x_4, norm(A_4), s = 4)
plt.xlabel('Cross-section from bottom [\u03BCm]')
plt.ylabel('Relative concentration')
plt.title('Line 4')
plt.show()
plt.plot(x_1, ficks_1(x_1, *popt), label='Time 1')
plt.plot(x_2, ficks_2(x_2, *popt), label='Time 2')
plt.plot(x_3, ficks_3(x_3, *popt), label='Time 3')
plt.plot(x_4, ficks_4(x_4, *popt), label='Time 4')
plt.xlabel('Cross-section from bottom [\u03BCm]')
plt.ylabel('Relative concentration')
plt.title('Plot of the curve-fitted functions')
plt.legend()
plt.show()
D_list = (1e-12)*np.array([D_1, D_2, D_3, D_4])
D_mean = np.mean(D_list)
std = np.std(D_list)
r_values = []
for i in D_list:
    r_values.append(SE(i))
print("Mean diffusion coefficient=", D_mean, "| Standard deviation=", std)
16
print("Radius from average diffusion coefficient = ", SE(D_mean) , " | Radius for D - std = ", SE(D_mean
- std), " | Radius for D + std = ", SE(D_mean + std), 'std for radius=', np.std(np.array(r_values))
)

import numpy as np
from functions.chambolle_prox_TV import DivergenceIm 
x0 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9,10,11,12],[13,14,15,16]])
print(x0)
print(x0[:,-1])
x1 = np.array([[1, 2, 3, 4], [1, 4, 8, 12], [9, 10, 11, 12], [13, 14, 15, 16]])
print(x1)
gradim = DivergenceIm(x0, x1)
print(np.size(gradim))
print(gradim)
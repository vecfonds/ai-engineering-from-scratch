import numpy as np

A = np.array([1,2,3])
B = np.array([4,5,6])

#Với hai vector 1D, np.dot(A, B) và A @ B đều thực hiện Dot Product.
print(np.dot(A,B))
print(A @ B)
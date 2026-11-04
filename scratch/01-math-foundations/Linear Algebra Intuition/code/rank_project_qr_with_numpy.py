import numpy as np

A = np.array([[1,2],[3,4]])
print(f"Rank: {np.linalg.matrix_rank(A)}")

a = np.array([3,4])
b = np.array([1,0])
proj = (np.dot(a,b)/np.dot(b,b)) * b
print(f"Projection of {a} onto {b}: {proj}")


Q, R = np.linalg.qr(np.random.randn(3,3))
print(f"Q is orthogonal: {np.allclose(Q.T @ Q, np.eye(3))}")
print(f"R is upper triangular: {np.allclose(R, np.triu(R))}")


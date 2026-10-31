import math

a = [3,2]

magnitude = math.sqrt(sum(x**2 for x in a))

print(f"The magnitude of vector {a} is: {magnitude:.4f}")
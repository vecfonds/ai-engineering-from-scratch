def dot_product(a,b):
    if(len(a) != len(b)):
        raise ValueError("Vectors must have same dimensions")

    return sum(x*y for x,y in zip(a,b))

A = [1,2,3]
B = [4,5,6]

print(dot_product(A,B))
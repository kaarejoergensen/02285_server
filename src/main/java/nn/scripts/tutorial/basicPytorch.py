from __future__ import print_function

import torch

#Tom matrix
x = torch.empty(5, 3)
print("Empty matrix:\n",x)

#Random matrix
x = torch.rand(5, 3)
print("Random matrix:\n",x)

#Matrix med bare 0ere
x = torch.zeros(5, 3)
print("Zero-matrix:\n",x)

#Dtype endrer dataen til long
x = torch.zeros(5, 3, dtype=torch.long)
print("Zero-matrix with dtype:\n",x)

#Hard codet matrix
x = torch.tensor([5.5, 3])
print("Hardkodet:\n",x)

#Lage tensor basert på en tidligere
print("Tensor basert på tidligere. Andre matrix skriver over den gamle med annen datatype")
x = x.new_ones(5, 3, dtype=torch.double)
print(x)
x = torch.randn_like(x, dtype=torch.float)
print(x)

#Hente størrelse
print("Tensor size: ", x.size())

#Litt mattestuff
print("Math Stuff:")
y = torch.rand(5,3)
print(x + y)
result = torch.empty(5,3)
torch.add(x,y,out=result)
print(result)


#Nympy slicing
print("Litt slicing som nympy")
print(x)
print(x[:,1])








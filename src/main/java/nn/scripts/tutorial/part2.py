import torch

x = torch.ones(2,2, requires_grad=True)
print(x)

y = x + 2
print(y)

print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(z, out)

print("#####")

a = torch.randn(2,2)
print(a)
a = ((a * 3)/(a-1))

if(a.requires_grad == False):
    a.requires_grad_(True)

b = (a*a).sum()
print(b, b.grad_fn)


print("####")
out.backward()
print(x)
print(x.grad)


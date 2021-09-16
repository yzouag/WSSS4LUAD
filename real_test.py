import torch
import torch.nn as nn

m = nn.Sigmoid()
# weights=torch.randn(3)

loss = nn.BCELoss()
input = torch.Tensor([[0.3, 0.5, 0.7]])
target = torch.Tensor([[1, 0, 0]])
print(torch.argmax(input))
# print(torch.argmax(input))
print("tensor:", input.data)
# lossinput = m(input)
# output = loss(lossinput, target)

# print("输入值:")
# print(lossinput)
# print("输出的目标值:")
# print(target)
# # print("权重值")
# # print(weights)
# print("计算loss的结果:")
# print(output)
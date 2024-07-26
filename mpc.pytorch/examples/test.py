import torch

# Create batch of matrices A of shape (2, 3, 4)
A = torch.randn(2, 4, 4)

# Create batch of vectors B of shape (2, 4), then reshape to (2, 4, 1)
B = torch.randn(2, 4)
print(B.shape)
B = B.unsqueeze(2)
print(B.shape)

# Perform batch matrix multiplication
C = torch.bmm(A, B)

# Print the shapes of the tensors
print("Shape of A:", A.shape)  # Output: Shape of A: torch.Size([2, 3, 4])
print("Shape of B:", B.shape)  # Output: Shape of B: torch.Size([2, 4, 1])
print("Shape of C:", C.shape)  # Output: Shape of C: torch.Size([2, 3, 1])

# Print the result for verification
print("A[0]:", A[0])
print("B[0]:", B[0])
print("C[0]:", C[0])  # C[0] should be equal to torch.mm(A[0], B[0])

print(torch.dot(torch.tensor([1, 2]), torch.tensor([3, 4])))  # Output: tensor(11)
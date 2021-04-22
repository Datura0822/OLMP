import numpy as np

seed=np.random.randint(1000000)
seed=981118#seed 93306,124x,0.04;78011, 127x, 0.05,430000,150x, 515769,185x
print(np.random.seed(seed))

print(np.random.rand(4))
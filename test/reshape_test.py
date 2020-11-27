import numpy as np
a = np.arange(1,25)
print(a)

a = a.reshape(2,3,4)
print(a)

a = a.transpose(1,0,2)
print(a)

a = a.reshape(3,2,4)
print(a)
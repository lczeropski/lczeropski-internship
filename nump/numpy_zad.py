#89
#%%
from typing import Any
import numpy as np
#%%
def n_largest(li,n):   
    pointer = np.argpartition(li, -n)[-n:]
    return li[pointer]


# %%
a = np.array([1,2,3,4,5,6,7,8,9,10])
b = np.array([4,6,2,8,5,0,8,2,67])

# %%
n_largest(b,3)
# %%
#81
#Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate 
#an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]? 
# %%
Z = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
# %%
R=[]
for i in range(len(Z)-3):
    R.append(Z[i:i+4])
# %%
R
# %%
#87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)
S=np.arange(0,256).reshape(16,16)
# %%
S

# %%
result = np.add.reduceat(np.add.reduceat(S, np.arange(0, S.shape[0], 4), axis=0),
        np.arange(0, S.shape[1], 4), axis=1)
# %%
result
# %%
S=np.ones((16,16))
# %%
#83
freq = np.array([0,1,2,2,2,3,3,3,3,3,3,4,4,6,1,6,2,3,5,6,7])
# %%
counts = np.bincount(freq)
# %%
counts
# %%
max_val = np.where(counts == np.max(counts))
max_val2 = np.argmax(counts)
# %%
max_val # czyli 3 jest najcześciej, bo bincount() zawsze zlicza od 0 do max w liscie
max_val2
# %%
#%%
#82
matrix = np.arange(0,25).reshape(5,5).astype(np.int64)
# %%
matrix
# %%
np.linalg.matrix_rank(matrix)
# %%

# %%
a = np.random.rand(1000,200)
b = np.random.rand(200,200)

# %%
print(np.diag(np.dot(a, b)))
# %%
a = np.array([[4,3, 1],[5 ,7, 0],[9, 9, 3],[8, 2, 4]]) 
# %%
a[[0, 2]] = a[[2, 0]]
print(a)
# %%
not_unq =np.array([[1, 1, 1, 0, 0, 0],
       [0, 1, 1, 1, 0, 0],
       [0, 1, 1, 1, 0, 0],
       [1, 1, 1, 0, 0, 0],
       [1, 1, 1, 1, 1, 0]])
# %%
unq = np.unique(not_unq, axis = 0)
# %%
unq
# %%
A = np.arange(0,128,8)
# %%
A
# %%
Bin = ((A.reshape(-1,1) & (2**np.arange(8))) !=0).astype(int)
# %%
Bin
# %%

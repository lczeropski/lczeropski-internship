#%% 
#1
import numpy as np
# %%
#2
np.__version__
# %%
#3
n=np.nan
v = np.zeros(10)
w = np.empty(100)
# %%

print(n,v,w)
# %%
#4
print(v.itemsize,w.itemsize)

# %%
#5
help(np.add)
# %%
#6
n = np.zeros(10)
n[4] = 1
print(n)
# %%
#7
r = np.linspace(10,49)
# %%
#8
np.flip(r)
# %%
#9
mac = np.array([[0,1,2],[3,4,5],[6,7,8]])
# %%
mac
# %%
#10
np.nonzero([1,2,0,0,4,0] )
# %%
#11
id_mat = np.identity(3)
# %%
id_mat
# %%
#12
rng = np.random.default_rng(0)
ran = rng.random((3,3,3))
# %%
# %%
#13
ten = rng.random((10,10))
# %%
print(ten.min(),ten.max())
# %%
#14
vec = rng.random(30)
# %%
vec.mean()
# %%
#15
one_zero = np.ones((10,10))
# %%
one_zero[1:-1,1:-1] = 0
# %%
one_zero
# %%
#16
one_zero = np.pad(one_zero, pad_width = 1, mode = 'constant', constant_values=0)
# %%
one_zero
# %%
#17
0 * np.nan #nan
np.nan == np.nan #false
np.inf > np.nan #false
np.nan - np.nan #nan
np.nan in set([np.nan]) #True
0.3 == 3 * 0.1 #False
# %%
#18
under = np.diag([1,2,3,4],-1)
# %%
under
# %%
#19
che = np.zeros((8,8))
che[1::2,::2] = 1
che[::2,1::2] = 1
# %%
che
# %%
#20
# %%
#x=3,y=1,z=7
# %%
#21
list_zer_jed = np.array([[0,1],[1,0]])
c = np.tile(list_zer_jed, (4,4))
# %%
c
# %%
#22

# %%
matrix = np.arange(0,25).reshape(5,5).astype(np.int64)
# %%
matrix
# %%
normed_matrix = matrix / np.linalg.norm(matrix, axis =1)[:, np.newaxis]
# %%
normed_matrix
# %%
np.sum( normed_matrix**2, axis=-1 )
# %%
#23
colordt = np.dtype([('R', np.int32),('G', np.int32),('B', np.int32),('A', np.int32) ])
# %%
#24
matrix1 = np.arange(0,15).reshape(5,3).astype(np.int64)
matrix2 = np.arange(0,6).reshape(3,2).astype(np.int64)
# %%
matrix1 @ matrix2
# %%
#25
negate = np.arange(0,10)
# %%
negate[3:9] = np.multiply(negate[3:9],-1)
# %%
negate
# %%
print(sum(range(5),-1))
from numpy_zad import *
print(sum(range(5),-1))
# %%
a=list(range(5))
sum(a,-1)
a[-1]
a
# %%
#26
from numpy_zad import *
b=list(range(5))
sum(b,-1)
# %%
b[-1]
b
# %%
#27
Z=np.ones(2)
# %%
Z**Z #L
#2 << Z >> #N #
Z <- Z #L
1j*Z #L
Z/1/1 #L
Z<Z>Z #N
# %%
#28
np.array(0) / np.array(0) #nan
np.array(0) // np.array(0) #0
np.array([np.nan]).astype(int).astype(float) #array([-9.22337204e+18])
# %%
#29
test = np.array([32.11, 51.5, 0.112])
np.round(test,1)
# %%
a = np.array([1, 2, 3])
b = np.array([0, 2, 4])
# %%
a[a == b]
# %%
#30
np.seterr(all='ignore')
# %%
#32
np.sqrt(-1) == np.emath.sqrt(-1) #F
# %%
#33
yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today     = np.datetime64('today', 'D')
tomorrow  = np.datetime64('today', 'D') + np.timedelta64(1, 'D')

# %%
#34
print(np.arange('2016-07', '2016-08', dtype='datetime64[D]'))
# %%
#39
x = np.zeros((5,5))
x += np.arange(5)
print(x)
# %%
np.arange(0.1,1,0.09).astype(float64)
# %%
#40
x = np.random.random(10)
# %%
x.sort()
# %%
x
# %%
#41
a=np.array(range(5))
# %%
sum(a)
# %%
np.sum(a)
# %%
#42
x = np.random.random(10)
y = np.random.random(10)
# %%
x,y
# %%
x in y
# %%
#43
y = np.random.random(10)
y=(y,)
# %%
#45
y = np.random.random(10)
# %%
y[y.argmax()] = 0
# %%
y
# %%
#53
a=np.array(5.3213)
a.astype(float64)

# %%
a.astype(int32)
# %%
#58
matrix1 = np.arange(0,15).reshape(5,3).astype(np.int64)
matrix1 = matrix1 - matrix1.mean(axis=1, keepdims=True)
# %%
matrix1
# %%
matrix1 = np.arange(15,0,-1).reshape(5,3).astype(np.int64)
# %%
matrix1
# %%
matrix1[matrix1[:,1].argsort()] 
# %%

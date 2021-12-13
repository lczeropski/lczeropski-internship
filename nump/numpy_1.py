#%% 
#1
from math import nan
from pandas.core.construction import array
from pandas.core.dtypes.dtypes import CategoricalDtype
import numpy as np
import numpy.lib as npl
# %%
#2
np.__version__
# %%
#3
n=np.nan
v = np.zeros(10)
w = np.empty(10)
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
r = np.arange(10,50,1)
# %%
#8
np.flip(r)
# %%
#9
mac = np.arange(0,9)
mac.reshape(3,3)
# %%
mac
# %%
#10
x=[1,2,0,0,4,0]
np.nonzero(x )
# %%
#11
id_mat = np.identity(3)
np.eye(3)
# %%
id_mat
# %%
#12
np.random.random((3,3,3))
np.random.randint(27, size=(3,3,3))
# %%
# %%
#13
x = np.random.random((10,10))
# %%
x.min(),x.max()
# %%
#14
x = np.random.random(30)
x.mean()
# %%
#15
one_zero = np.ones((10,10))
one_zero[1:-1,1:-1] = 0
one_zero
#16
#%%
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
under

# %%
#19
che = np.zeros((8,8))
che[1::2,::2] = 1
che[::2,1::2] = 1
che
# %%
#20
# %%
np.unravel_index(100, (6, 7, 8))
# %%
#21
list_zer_jed = np.array([[0,1],[1,0]])
np.tile(list_zer_jed, (4,4))

# %%
#22
matrix = np.random.randint(25,size=(5,5))
######
normed_matrix = matrix / np.linalg.norm(matrix, axis =1)
print(matrix,"\n",normed_matrix)
# %%
#23
colordt = np.dtype([('R', np.ubyte),('G', np.ubyte),('B', np.ubyte),('A', np.ubyte) ])
colordt
# %%
#24
matrix1 = np.arange(15).reshape(5,3)
matrix2 = np.arange(6).reshape(3,2)
# %%
matrix1 @ matrix2 
#lub 
#%%
np.dot(matrix1,matrix2)
# %%
#25
x = np.arange(0,10)
# %%
x[(x >= 3) & (x <=8 )] = np.multiply(x[(x >= 3) & (x <=8 )],-1)
x
# %%
#26
print(sum(range(5),-1)) #9
from numpy import *
print(sum(range(5),-1)) #10
# %%
#27
Z=np.arange(10)
# %%
Z**Z #L
2 << Z >> 2#L 
Z <- Z #L
1j*Z #L
Z/1/1 #L
Z<Z>Z #N valueError
# %%
#28
np.array(0) / np.array(0) #nan
np.array(0) // np.array(0) #0
np.array([np.nan]).astype(int).astype(float) #array([-9.22337204e+18])
# %%
#29
x = np.random.uniform(-10, +10, 10)
x, np.copysign(np.ceil(np.abs(x)), x)
# %%
#30
a = np.arange(0,20)
b = np.arange(10,15)
# %%
np.intersect1d(a,b)
# %%
#31
defaults = np.seterr(all="ignore")
Z = np.ones(10) / 0
Z
#%%
#back to warnings
_ = np.seterr(**defaults)
Z = np.zeros(10) / 0
Z
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
#%%
#35
A = np.ones(3) * 1
B = np.ones(3) * 2
np.add(A, B, out=B)
np.divide(A, 2, out=A)
np.negative(A, out=A)
np.multiply(A, B, out=A)

#%%
#36
x = np.random.uniform(-10, +10, 10)
x
#%%
x.astype(np.int32)
np.trunc(x)
np.floor((x))
np.ceil(x)-1
x-x%1
# %%
#37
x = np.zeros((5,5))
x += np.arange(5)
print(x)
#%%
#38
def generate():
    for i in range(10):
        yield i
np.fromiter(generate(), dtype=np.float32, count=-1)
# %%
np.linspace(start=0.1, stop=1, num=10, endpoint=False)
# %%
#40
x = np.random.random(10)
x.sort()
x
# %%
#41
a=np.array(range(100000))
# %%
import time
start_time = time.time()
sum(a)
end_time = time.time()
print(1000*(end_time - start_time),'sum(a)')
#%%
start_time = time.time()
np.add.reduce(a)
end_time = time.time()
print(1000*(end_time - start_time),'np.add.reduce(a)')
# %%
start_time = time.time()
np.sum(a)
end_time = time.time()
print(1000*(end_time - start_time),'np.sum(a)')
# %%
#42
x = np.random.random(10)
y = np.random.random(10)
x in y, np.array_equal(x, y)
# %%
x = np.array([1,2,3])
y = np.array([1,2,3])
x in y, np.array_equal(x, y)
# %%
#43
y = np.random.random(10)
y.flags.writeable = False
#%%
#44
Z = np.random.random((10, 2))
def to_polar(X):
    Y, Z = X[:, 0], X[:, 1]
    rho = np.sqrt(Y**2 + Z**2)
    phi = np.arctan2(Z,Y)
    return (rho, phi)
#%%

Z, to_polar(Z)
# %%
#45
y = np.random.random(10)
y[y.argmax()] = 0
y
#%%
#46
z = np.zeros((5, 5), [('x', float), ('y', float)])
z['x'], z['y'] = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
z
#%%
#47
X = np.random.random((5,5))
Y = np.random.random((5,5))
C = 1/(X-Y)
C = 1/np.subtract.outer(X, Y)
np.linalg.det(C)
#%%
#48
for dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
    print("min ",np.iinfo(dtype).min,"max ",np.iinfo(dtype).max, "type ",dtype)
#%%
for dtype in [np.float16,np.float32, np.float64]:
    print("min ",np.finfo(dtype).min,"max ",np.finfo(dtype).max,"Eps ",np.finfo(dtype).eps, "type ",dtype)
#%%
#49
np.set_printoptions(threshold=float("inf"))
Z = np.zeros((10,10))
print(Z)
#%%
#50
Z = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(Z-v)).argmin()
print(Z ,v ,Z [index])
#%%
#51
Z = np.zeros(10, [ ('position', [('x', float),
                                 ('y', float)]),
                  ('color',     [('R', float),
                                 ('G', float),
                                 ('B', float)])])
print(Z['color'])
#%%
#52
Z = np.random.random((100,2))
X, Y = np.atleast_2d(Z[:,0], Z[:,1])
dist = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
print(Z,"\n",dist)
# %%
#53
Z = (np.random.rand(10)*10).astype(np.float32)
Y = Z.view(np.int32)
Y[:] = Z
print(Y)
# %%
#54 #np.genfromtxt
from io import StringIO
a = '''1, 2, 3, 4, 5
    6,  ,  , 7, 8
    ,  , 9,10,11'''
s= StringIO(a)
Z = np.genfromtxt(s, delimiter = ",", dtype = np.int32)
Z
#%%
#55 #np.ndenumerate or np.ndindex
Z = np.arange(16).reshape(4,4)
for index, value in np.ndenumerate(Z):
    print(index,value)
#or
#%%
for index in np.ndindex(Z.shape):
    print(index, Z[index])
#%%
#56
X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
D = np.sqrt(X*X+Y*Y)
me, sigma = 0.0, 1.0
Gauss = np.exp(-( (D-me)**2 / ( 2.0 * sigma**2 ) ) )
print(Gauss)
#%%
#57 np.put, np.random.choice
#n=10
p = 4
Z = np.zeros((10,10))
np.put(Z, np.random.choice(range(10*10), p, replace=False),1)
Z
# %%
#58
matrix = np.random.rand(3,5)
print(matrix)
matrix_minus = matrix - matrix.mean(axis=1, keepdims=True)
matrix_minus

#%%
#59 .argsort
nth = 2 # col number to sort
Z = (np.random.rand(5,5))*10
print(Z)
print('###############')
print(Z[Z[:, nth].argsort()])
# %%
#60 ~ , any
Z = np.random.uniform(0,3,(3,10))
print((~Z.any(axis=0)).any())
Z[:,1] = 0
print((~Z.any(axis=0)).any())
# %%
#61
# flat
Z = np.random.uniform(0,1,10)
value = 0.2
result = Z.flat[np.abs(Z - value).argmin()]
print(Z,result)
# %%
#62
# np.nditer, 
# [...] An Ellipsis,
# When indexing an array, shorthand that the missing axes, if they exist, are full slices.
A = np.arange(3).reshape(3,1)
B = np.arange(3).reshape(1,3)
iterator = np.nditer([A,B,None])
for x,y,z in iterator:
    z[...] = x + y
print(iterator.operands[0],"\n",iterator.operands[1],"\n",iterator.operands[2])
# %%
A = np.arange(1,11).reshape(10,1)
B = np.arange(1,11).reshape(1,10)
iterator = np.nditer([A,B,None])
for x,y,z in iterator:
    z[...] = x + y
print(iterator.operands[0],"\n",iterator.operands[1],"\n",iterator.operands[2])
# %%
#63 class method
class NamedArray(np.ndarray):
    def __new__(cls, array, name = None):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.name = getattr(obj, 'name', None)
    def __repr__(self):
        return f" {self.name}: {super().__repr__()} "
    def __str__(self) :
        return  self.name + ": " + super().__str__() 
# %%
#64 np.bincount or np.add.at
Z = np.ones(10)
print(Z)
I = np.random.randint(0,len(Z),20)
print(I,(np.bincount(I, minlength=len(Z))))
Z += np.bincount(I, minlength=len(Z))
print(Z)
#%%
#np.add.at
Z = np.ones(10)
print(Z)
I = np.random.randint(0,len(Z),20)
print(I)
np.add.at(Z, I, 1)
print(Z)

# %%
#65 np.bincount
X = [1,2,3,4,5,6]
I = [0,1,1,1,1,1]
F = np.bincount(I,X)
print(X,I,F)
# %%
#66 np.unique
w, h = 256, 256
I = np.random.randint(0, 4, (h, w, 3)).astype(np.ubyte)
colors = np.unique(I.reshape(-1, 3), axis=0)
print(len(colors))
# %%
#67
Z = np.random.randint(0,10,(2,5,1,4))
sum = Z.sum(axis=(-2,-1)) # by passing a tuple of axes
print(sum)
# %%
sum = Z.reshape(Z.shape[:-2] + (-1,)).sum(axis=-1) #by flattening the last 2 dimensions into one
print(sum)
# %%
#68 bincount
D = np.random.uniform(0,1,100)
S = np.random.randint(0,10,100)
D_sums = np.bincount(S, weights = D)
D_counts = np.bincount(S)
D_means = D_sums/D_counts
print(D_means)
#%%
#pandas
import pandas as pd
print(pd.Series(D).groupby(S).mean())
# %%
#69
A = np.random.uniform(0,1,(5,5))
B = np.random.uniform(0,1,(5,5))
#slow 
#%%
np.diag(np.dot(A,B))
#fast
#%%
np.sum(A*B.T, axis=1)
#faster
#%%
np.einsum("ij,ji->i", A, B) # np.einsum('ii->i', a) is like np.diag(a)
#numpy.einsum(subscripts, *operands, out=None, dtype=None, order='K', casting='safe', optimize=False)[source]
#Evaluates the Einstein summation convention on the operands.
#Using the Einstein summation convention, many common multi-dimensional, linear algebraic array operations can be represented in a simple fashion. In implicit mode einsum computes these values.
#In explicit mode, einsum provides further flexibility to compute other array operations that might not be considered classical Einstein summation operations, by disabling, or forcing summation over specified subscript labels.
# %%
#70 [1,2,3,4,5]
Z = np.array([1,2,3,4,5])
num_zero = 3
W = np.zeros(len(Z) + (len(Z)-1)*(num_zero)) # creates a vector of 0 of length 5+13=17
W[::num_zero+1] = Z # replaces every 4th zero with an element of Z 
W
# %%
#71
A = np.ones((5,5,3))
B = 2*np.ones((5,5))
C = A * B[:,:,None]
print(C)
#%%
#72
Z = np.arange(25).reshape(5,5)
print(Z)
Z[[0,1]] = Z[[1,0]]
print(Z)
# %%
#73
triangles = np.random.randint(0,10,(10,3))
print(triangles)
F = np.roll(triangles.repeat(2,axis=1),-1,axis=1)
F = F.reshape(len(F)*3,2)
F = np.sort(F,axis=1)
G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
G = np.unique(G)
print(G,len(G))
# %%
#74
Z = [1,2,2,3,5,7,1,0,9,5,2]
C = np.bincount(Z)
A = np.repeat(np.arange(len(C)),C)
A,Z
# %%
#75 
def moving_average(a , n=3): #n = window size
    cumulative = np.cumsum(a, dtype = float) #Return the cumulative sum of the elements along a given axis.
    cumulative[n:] = cumulative[n:] - cumulative[:-n] # value = sum of n-1 previous + itself
    return cumulative[n-1:]/n

#%%
#using strides
import numpy.lib as npl
def moving_average_strides(a, n = 3):
    v = npl.stride_tricks.sliding_window_view(a,n)
    result = v.mean(axis=-1)
    return result
#%%
Z = np.arange(10,35)
#%%
moving_average_strides(Z,2), moving_average(Z,2)
# %%
#76 #stride_tricks
import numpy.lib as npl #includes stride_tricks
# Create a view into the array with the given shape and strides.
# use writeable=False to avoid accidental write operations
def rolling(a, window): # a-1d array, window-size of row in new 2d array
    shape = (a.size - window + 1, window) # shape of new array 
    strides = (a.strides[0], a.strides[0]) # num of strides better to not touch for 1d (8,8)
    return npl.stride_tricks.as_strided(a, shape=shape, strides=strides, writeable=False)
Z = rolling(np.arange(10), 3)
print(Z)
# %%
#77 negate a boolean np.logical_not
Z = np.random.randint(0,2,10)
np.logical_not(Z, out = Z)
#%%
#negate a sign np.negative
Z = np.random.uniform(-1.0,1.0,100)
np.negative(Z, out=Z)
# %%
#78
def distance(P0, P1, p):
    T = P1 - P0
    L = (T**2).sum(axis=1)
    U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L
    U = U.reshape(len(U),1)
    D = P0 + U*T - p
    return np.sqrt((D**2).sum(axis=1))

# %%
P0 = np.random.uniform(-10,10,(10,2))
P1 = np.random.uniform(-10,10,(10,2))
p  = np.random.uniform(-10,10,( 1,2))
#%%
print(distance(P0, P1, p))
# %%
#79
p = np.random.uniform(-10, 10, (10,2))
print(np.array([distance(P0,P1,p_i) for p_i in p]))
# %%
#80 Consider an arbitrary array, write a function that extract a subpart with a 
# fixed shape and centered on a given element (pad with a fill value when necessary) (★★★)
def sub(A, s=(5,5), fill = 0, pos = (1,1) ):
    R = np.ones(s, dtype=A.dtype)*fill
    P  = np.array(list(pos)).astype(int)
    Rs = np.array(list(R.shape)).astype(int)
    Zs = np.array(list(A.shape)).astype(int)

    R_start = np.zeros((len(s),)).astype(int)
    R_stop  = np.array(list(s)).astype(int)
    A_start = (P-Rs//2)
    A_stop  = (P+Rs//2)+Rs%2

    R_start = (R_start - np.minimum(A_start,0)).tolist()
    A_start = (np.maximum(A_start,0)).tolist()
    R_stop = np.maximum(R_start, (R_stop - np.maximum(A_stop-Zs,0))).tolist()
    A_stop = (np.minimum(A_stop,Zs)).tolist()

    r = [slice(start,stop) for start,stop in zip(R_start,R_stop)]
    a = [slice(start,stop) for start,stop in zip(A_start,A_stop)]
    R[tuple(r)] = A[tuple(a)]
    return R
# %%
Z = np.arange(0,64).reshape(8,8)
#%%
sub(Z,(3,3)) 
#%%
#81 @75,76
Z = np.arange(1,15,dtype=np.uint32)
R = npl.stride_tricks.as_strided(Z,(11,4),(4,4))
print(R)
# %%
#82 the number of linearly independent rows or columns in the matrix
# np.linalg.svd
# Singular Value Decomposition
Z = np.random.uniform(0,1,(10,10))
S = np.linalg.svd(Z, compute_uv = False) # S - Vector(s) with the singular values, compute_uv = True, only S is needed
rank = np.sum(S > 1e-10)
print(rank)
# %%
#83
Z = np.random.randint(0,10,50)
print(np.bincount(Z).argmax())
# %%
#84
# strides
Z = np.random.randint(0,5,(10,10))
n = 3
i = 1 + (Z.shape[0]-3)
j = 1 + (Z.shape[1]-3)
C = npl.stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
print(C)
# %%
#85 
class Symetric(np.ndarray):
    def __setitem__(self, index, value):
        i,j = index
        super(Symetric, self).__setitem__((i,j), value)
        super(Symetric, self).__setitem__((j,i), value)

def symetric(Z):
    return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)

S = symetric(np.random.randint(0,10,(5,5)))
#%%
S[2,3] = 67
print(S)
#%%
#86
# np.tensordot Compute tensor dot product along specified axes
#axes = 0 : tensor product A x B
#axes = 1 : tensor dot product A * B
#axes = 2 : (default) tensor double contraction A : B
p, n = 10, 20
M = np.ones((p,n,n))
V = np.ones((p,n,1))
S = np.tensordot(M, V, axes =[[0,2],[0,1]])
print(S)
# %%
#87
#np.add.reduceat
Z = np.ones((16,16))
k = 4
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                       np.arange(0, Z.shape[1], k), axis=1)
print(S)
# %%
# strides
windows = npl.stride_tricks.sliding_window_view(Z, (k, k))
S = windows[::k, ::k, ...].sum(axis=(-2, -1))
print(S)
windows
# %%
#88 game of life
def iterate(Z):
    # Count neighbours
    N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
         Z[1:-1,0:-2]                + Z[1:-1,2:] +
         Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])

    # Apply rules
    birth = (N==3) & (Z[1:-1,1:-1]==0)
    survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)
    Z[...] = 0
    Z[1:-1,1:-1][birth | survive] = 1
    return Z

Z = np.random.randint(0,2,(20,20))
#%%
for i in range(10): 
    Z = iterate(Z)
    print(Z)
    print("\n\n\n\n")
# %%
#89
Z = np.arange(10000)
np.random.shuffle(Z)
n = 5
# np.argsort  
#%%
print (Z[np.argsort(Z)[-n:]])
# %%
#np.argpartition
print (Z[np.argpartition(-Z,n)[:n]])
#%%
#90
#np.indices
def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)

    ind = np.indices(shape, dtype=int)
    ind = ind.reshape(len(arrays), -1).T #creates every combination as
    print(ind)

    for n, arr in enumerate(arrays):
        ind[:, n] = arrays[n][ind[:, n]] # assign values from arrays coresponding to generated indexes

    return ind
#%%
print (cartesian(([1, 2, 3], [4, 5], [6, 7])))
# %%
#91
#access fields of structured arrays by attribute rather than by index
# np.core.records.fromarrays
Z = np.array([("Hello", 2.5, 3,'641243'),
              ("World", 3.6, 2,'74214')])
R = np.core.records.fromarrays(Z.T, names= 'col1, col2, col3, col4',
                              formats = 'S8, f8, i8, S8')
# %%
#92
Z = np.random.rand(10000000)
# %%
%timeit np.power(Z,3)
%timeit Z*Z*Z
%timeit np.einsum('i,i,i->i',Z,Z,Z)
# %%
#93
#np.where
A = np.random.randint(0,10,(8,3))
B = np.random.randint(0,10,(2,2))

C = (A[..., np.newaxis, np.newaxis] == B)
rows = np.where(C.any((3,1)).all(1))[0]
print('A',A,'\n B',B)
print('result', rows)

# %%
#94
Z = np.random.randint(0,3,(10,3))
print(Z)
# %%
E = np.all(Z[:,1:] == Z[:,:-1], axis=1)
U = Z[~E] #~not
#%%
print(E,U)
# %%
#95 np.unpackbits
I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
print(np.unpackbits(I[:, np.newaxis], axis=1))
# %%
#96 np.unique
Z = np.random.randint(0,2,(10,3))
print(Z)
print("#############")
print(np.unique(Z, axis=0))
# %%
#97 np.einsum
A = np.random.uniform(0,1,10)
B = np.random.uniform(0,1,10)

np.einsum('i->', A)       # np.sum(A)
np.einsum('i,i->i', A, B) # A * B
np.einsum('i,i', A, B)    # np.inner(A, B)
np.einsum('i,j->ij', A, B)    # np.outer(A, B)
# %%
#98
phi = np.arange(0, 10*np.pi, 0.1)
a = 1
x = a*phi*np.cos(phi)
y = a*phi*np.sin(phi)

dr = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths
r = np.zeros_like(x)
r[1:] = np.cumsum(dr)                # integrate path
r_int = np.linspace(0, r.max(), 200) # regular spaced path
x_int = np.interp(r_int, r, x)       # integrate path
y_int = np.interp(r_int, r, y)
#%%
#99
X = np.asarray([[1.0, 0.0, 3.0, 8.0],
                [2.0, 0.0, 1.0, 1.0],
                [1.5, 2.5, 1.0, 0.0]])
n = 4
M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1)
M &= (X.sum(axis=-1) == n)
print(X[M])
#%%
#100
X = np.random.randn(100) # random 1D array
N = 1000 # number of bootstrap samples
idx = np.random.randint(0, X.size, (N, X.size))
means = X[idx].mean(axis=1)
confint = np.percentile(means, [2.5, 97.5])
print(confint)

# %%

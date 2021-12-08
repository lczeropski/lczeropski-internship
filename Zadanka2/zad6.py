#%%
from typing import List
import numpy as np
import itertools as it
import operator

#%%
class Solution:
    def __init__(self,nums) -> None:
        self.n = nums
    def productExceptSelf(self) -> List[int]:
        return [np.prod(self.n[:i] + self.n[i+1:]) for i in range(0,len(self.n)) ]
 
# %%
a=Solution([1,2,3,4])
b=Solution([-1,-1,1,-1,2,1,-1])
# %%
a.productExceptSelf()
#%%
b.productExceptSelf()
# %%
class Solution:
    def __init__(self,nums) -> None:
        self.n = nums
    def productExceptSelf(self):
        t1 = [1] + list(it.accumulate(self.n, operator.mul))[:-1]
        t2 = list(it.accumulate(self.n[::-1], operator.mul))[::-1][1:] + [1]
        return [x*y for x, y in zip(t1, t2)]
    
    


# %%

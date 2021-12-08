#%%
from typing import List
import time
#%%
class Solution:
    def __init__(self,nums) -> None:
        self.n = nums
    def findMin(self) -> int:
        if len(self.n)<=1 : return self.n[0]
        if self.n[-1] > self.n [0]:
            return self.n[0]
        for i in range(len(self.n)):
            if self.n[i] > self.n[i+1]:
                return self.n[i+1]
# %%
a=Solution([2,3,4,5,1])
# %%
start_time = time.time()
a.findMin()
print((time.time() - start_time)*10000)
# %%
class Solution:
    def findMin(self, nums: List[int]) -> int:
        return min(nums)
# %%

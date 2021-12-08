#%%
from typing import List
#%%
class Solution:
    def __init__(self , height) -> None:
        self.h = height
    def maxArea(self) -> int:
        max_area=0
        i=0
        j=len(self.h)-1
        while i<j:
            area = min(self.h[i],self.h[j]) * (j-i)
            max_area = max(max_area,area)
            if self.h[i] < self.h[j]:
                i+=1
            else :
                j-=1
        return max_area
# %%
s=Solution([1,8,6,2,5,4,8,3,7])
s.maxArea()
# %%
s=Solution([1,2,4,3])
s.maxArea()
# %%
s=Solution([8,20,1,2,3,4,5,6])
s.maxArea()
# %%

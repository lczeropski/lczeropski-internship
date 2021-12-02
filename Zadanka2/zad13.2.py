#%%
from typing import List
#%%
class Solution:
    def maxArea(self, height: List[int]) -> int:
        max_area=0
        i=0
        j=len(height)-1
        while i<j:
            area = min(height[i],height[j]) * (j-i)
            max_area = max(max_area,area)
            if height[i] < height[j]:
                i+=1
            else :
                j-=1
        return max_area
# %%
s=Solution
# %%
s.maxArea(s,[1,8,6,2,5,4,8,3,7])
# %%
s.maxArea(s,[1,2,4,3])
# %%
#%%
s.maxArea(s,[8,20,1,2,3,4,5,6])

#%%
from typing import List
#%%
class Solution:
    def maxArea(self, height: List[int]) -> int:
        max_area = 0
        m=max(height)
        ind = height.index(m)
        temp=height[:height.index(m)]+height[height.index(m)+1:]
        if height[0] <= height[-1]:
            max_area = height[0] * (len(height)-1)
        if len(height)==2:
            return min(height) * 1
        s = 0
        while height:
            m=max(height)
            ind = height.index(m)
            temp=height[:height.index(m)]+height[height.index(m)+1:]
            for i in range(len(temp)):
                if i < ind :
                    dist = ind - i +s
                elif i > ind :
                    dist =  i - ind +1 +s
                elif i == ind:
                    dist = 1 
                r = m - temp[i]
                if (m-r) * dist > max_area:
                    max_area = (m-r) * dist
            s+=1
            height.pop(height.index(m))
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

# %%
a[-1]
# %%
s.maxArea(s,[2,1])
# %%

#%%

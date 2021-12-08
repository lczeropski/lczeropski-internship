#%%
from typing import List
#%%
class Solution:
    def __init__(self , height) -> None:
        self.h = height
    def maxArea(self) -> int:
        max_area = 0
        m=max(self.h)
        ind = self.h.index(m)
        temp=self.h[:self.h.index(m)]+self.h[self.h.index(m)+1:]
        if self.h[0] <= self.h[-1]:
            max_area = self.h[0] * (len(self.h)-1)
        if len(self.h)==2:
            return min(self.h) * 1
        s = 0
        while self.h:
            m=max(self.h)
            ind = self.h.index(m)
            temp=self.h[:self.h.index(m)]+self.h[self.h.index(m)+1:]
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
            self.h.pop(self.h.index(m))
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

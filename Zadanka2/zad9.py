#%%
from typing import List


class Solution:
    def __init__(self, intervals) -> None:
        self.i = intervals
    def merge(self) -> List[List[int]]:
        self.i.sort(key=lambda x: (x[0], x[1]))
        beg = self.i[0][0]
        end = self.i[0][1]
        m_point = 0
        for i in range(1, len(self.i)):
            if self.i[i][0] <= end:
                end = max(end,self.i[i][1])
            else:
                self.i[m_point] = [beg, end]
                m_point += 1
                beg = self.i[i][0]
                end = self.i[i][1]
        self.i[m_point] = [beg, end]
        m_point +=1
        return self.i[:m_point]
                    

# %%
s=Solution([[1,3],[2,6],[8,10],[15,18]])
ss=Solution([[1,4],[0,4]])
# %%
s.merge()
# %%
ss.merge()
# %%

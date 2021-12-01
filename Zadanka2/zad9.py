#%%
class Solution:
    def merge(self, intervals: list[list[int]]) -> list[list[int]]:
        intervals.sort(key=lambda x: (x[0], x[1]))
        print(intervals)
        beg = intervals[0][0]
        end = intervals[0][1]
        m_point = 0
        for i in range(1, len(intervals)):
            if intervals[i][0] <= end:
                end = max(end,intervals[i][1])
            else:
                intervals[m_point] = [beg, end]
                m_point += 1
                beg = intervals[i][0]
                end = intervals[i][1]
        intervals[m_point] = [beg, end]
        m_point +=1
        return intervals[:m_point]
                    

# %%
s=Solution
# %%
s.merge(s,[[1,3],[2,6],[8,10],[15,18]])
# %%
s.merge(s,[[1,4],[0,4]])
# %%

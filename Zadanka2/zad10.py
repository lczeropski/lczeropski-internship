
#%%
from typing import List


class Solution:
    def __init__(self, strs) -> None:
        self.s = strs
    def groupAnagrams(self) -> List[List[str]]:
        d = {}
        for i in self.s:
            h = "".join(sorted(i))
            if h in d.keys():
                d[h].append(i)
            else:
                d[h] = [i]
        return list(d.values())
                
            
#%%
s=Solution(["eat","tea","tan","ate","nat","bat"])
# %%
s.groupAnagrams()
# %%

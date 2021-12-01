
#%%
class Solution:
    def groupAnagrams(self, strs: list[str]) -> list[list[str]]:
        d = {}
        for i in strs:
            h = "".join(sorted(i))
            if h in d.keys():
                d[h].append(i)
            else:
                d[h] = [i]
        return list(d.values())
                
            
#%%
s=Solution
# %%
s.groupAnagrams(s,["eat","tea","tan","ate","nat","bat"])
# %%

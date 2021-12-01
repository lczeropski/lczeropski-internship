#%%
class Solution:
    def isValid(self, s: str) -> bool:
        para = ['()', '{}', '[]']
        while any(x in s for x in para):
            for pa in para:
                s = s.replace(pa, '')
        return not s
            
                    
# %%
a=Solution
# %%
a.isValid(a,s = "()[]{}")
# %%
a.isValid(a,s = "([)]")
# %%
a.isValid(a,s = "(]")
# %%

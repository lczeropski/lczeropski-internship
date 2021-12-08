#%%
class Solution:
    def __init__(self, string) -> None:
        self.s = string
    def isValid(self) -> bool:
        para = ['()', '{}', '[]']
        while any(x in self.s for x in para):
            for pa in para:
                self.s = self.s.replace(pa, '')
        return not self.s
            
                    
# %%
a=Solution("()[]{}")
b=Solution("sda(){}:")
c=Solution('{{{})}}')
# %%
a.isValid()
#%%
b.isValid()
#%%
c.isValid()

# %%

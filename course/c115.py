#%%
def includes(li, num, st=0):
    if type(li) == list:
        return num in li[st:]
    if type(li) == dict:
        return num in li.values()
    return num in li
# %%
includes([1,2,3],1)
# %%
includes([1,2,3],1,1)
# %%
includes({ 'a': 1, 'b': 2 }, 1)
# %%
includes({ 'a': 1, 'b': 2 }, 'a')
# %%
includes('abcd', 'b') 
# %%
includes('abcd', 'e') 
# %%

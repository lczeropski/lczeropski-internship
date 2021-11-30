#%%
def range_in_list(li, start=0, end=None):
    end = end or len(li)
    return sum(li[start:end+1])
# %%
range_in_list([1,2,3,4],0,2) #  6
# %%

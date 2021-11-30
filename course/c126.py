#%%
def two_oldest_ages(li):
    o = max(li)
    li.pop(li.index(max(li)))
    s = max(li)
    li.append(o)
    return [s,o]

# %%
two_oldest_ages( [4,25,3,20,19,5] ) # [20,25]

#%%
def mode(li):
    a={i : li.count(i) for i in li}
    v=list(a.values())
    k=list(a.keys())
    return k[v.index(max(v))]
# %%
mode([2,4,1,2,3,3,4,4,5,4,4,6,4,6,7,4]) # 4
# %%

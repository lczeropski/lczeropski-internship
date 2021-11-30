#%%
a=['1',[1],1]

b=[[1],['a'],[]]
# %%
def list_check(l):
    for elem in l:
        if type(elem) != (list) :
            return False
    return True
# %%
list_check(a)

# %%
list_check(b)

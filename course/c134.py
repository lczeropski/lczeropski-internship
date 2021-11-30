#%%
def once(f):
    f.is_called = 1
    def inner(*args):
        if f.is_called:
            f.is_called = 0
            return f(*args)
        return None
    return inner
        
        
# %%
def add(a,b):
    return a+b

oneAddition = once(add)
# %%
oneAddition(2,2) # 4
# %%

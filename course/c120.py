#%%
def same_frequency(first, sec):
    return [str(first).count(num) for num in '1234567890'] == [str(sec).count(num) for num in '1234567890']

# %%
same_frequency(551122,221515) # True
# %%

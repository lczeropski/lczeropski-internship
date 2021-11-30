#%%
def is_odd_string(word):
    return sum([ord(c)-96 for c in word.lower()])%2!=0
# %%
is_odd_string('aaabbbccc') # True
# %%
is_odd_string('veryfun') # True
# %%
is_odd_string('veryfunny') # False
# %%

#%%
def truncate(word, num=3):
    if num < 3:
        return "Truncation must be at least 3 characters."
    if num > len(word):
        return word
    return word[:num-3] + '...'
# %%
truncate("Hello World",3)
# %%
truncate("Woah",3)
# %%
truncate("Problem solving is the best!", 10) 
# %%
truncate("Yo",100) # "Yo"
# %%

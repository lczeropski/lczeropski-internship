#%%
def letter_counter(word):
    word = word.lower()
    def inner(chr):
        return word.count(chr.lower())
    return inner
# %%
counter = letter_counter('Amazing')
counter('a') # 2
counter('m') # 1
# %%
counter2 = letter_counter('This Is Really Fun!')
counter2('i') # 2
#%%
counter2('t') # 1
# %%
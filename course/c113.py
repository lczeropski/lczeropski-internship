#%%
def titleize(words):
    return ' '.join([word[0].upper() + word[1:] + '' for word in words.split()])


#%%

# %%
titleize('oNLy cAPITALIZe fIRSt')
# %%

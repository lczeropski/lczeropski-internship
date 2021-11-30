#%%
def vowel_count(word):
    vowels=['a','e','i','o','u']
    return { v : word.lower().count(v) for v in word if v in vowels}
# %%
vowel_count('aabcdeifghoooooo')

# %%
vowel_count('aABcdeifEEghOOoooooo')
# %%

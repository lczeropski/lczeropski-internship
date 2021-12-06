#%%
import matplotlib as plt
import numpy as np
import seaborn as sns
import pandas as pd

# %%
# %%
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
chipo = pd.read_csv(url, sep = '\t')
# %%
chipo.head(10)
# %%
chipo.shape[0] #ilosc obserwacji
# %%
chipo.info() # pełne info ilosc kolumn, typ danych, ilosc obesrwacji
# %%
#ilosc kolumn mozna tez z 
chipo.shape[1]
# %%
chipo.columns #nazwy kolumn
# %%
chipo.index # jak jest dataset indeksowany
# %%
c = chipo.groupby('item_name')
c = c.sum()
c = c.sort_values(['quantity'], ascending = False)
# %%
c.head()
# %%
c.head(1)
# %%
total_quan = chipo.quantity.sum()
# %%
total_quan
# %%
chipo.item_price.dtype
# %%
chipo.item_price = chipo.item_price.apply(lambda x: float(x[1:-1]))
# %%
chipo.item_price.dtype
# %%
rev = (chipo['quantity'] * chipo['item_price']).sum() # zysk = cena * sprzedana ilosc
# %%
rev
# %%
orders = chipo.order_id.value_counts().count() # liczba zamówień
# %%
orders
# %%
chipo['rev_ord'] = chipo['item_price'] * chipo['quantity']
# %%
chipo['rev_ord']
chipo.groupby(by=['order_id']).sum().mean()['rev_ord']

# %%
chipo.item_name.value_counts().count()
# %%

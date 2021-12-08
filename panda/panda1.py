#%%
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# set this so the 
%matplotlib inline
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
################
chipo.item_price
# %%
#chipo.item_price = chipo.item_price.apply(lambda x: float(x[1:-1]))

# %%
prices = chipo.item_price
# %%
chipo_set = chipo.drop_duplicates(['item_name','quantity','choice_description'])
# %%
chipo_set
# %%
chipo_one_q = chipo_set[chipo_set.quantity == 1]
# %%
chipo_one_q
#%%
chipo_one_q[chipo_one_q['item_price']>10].item_name.nunique()
#%%
chipo_one_q[chipo_one_q['item_price']>10]
# %%
chipo_ndup = chipo.drop_duplicates(['item_name','quantity'])
# %%
chipo_one_prod = chipo_ndup[chipo_ndup.quantity == 1]
# %%
ppi= chipo_one_prod[['item_name', 'item_price']]
# %%
ppi.sort_values(by = "item_price", ascending = False).head(20)
# %%
chipo.item_name.sort_values()
####
#%%
chipo.sort_values(by = "item_name")
# %%
chipo.sort_values(by = "item_price",ascending = False).head(1)
# %%
salads = chipo[chipo.item_name == 'Veggie Salad Bowl']
# %%
len(salads)
# %%
soda = chipo[(chipo.item_name == "Canned Soda") & (chipo.quantity > 1)]
# %%
len(soda)
# %%
#visualizing
x = chipo.item_name
# %%
counts = Counter(x)
# %%
df = pd.DataFrame.from_dict(counts, orient='index')
# %%
df = df[0].sort_values(ascending = True)
# %%
df
# %%
p = df.tail(5)
# %%
p.plot(kind = 'bar')
plt.xlabel('Items')
plt.ylabel('Number of Times Ordered')
plt.title('Most ordered Chipotle\'s Items')
plt.show()
# %%
orders = chipo.groupby('order_id').sum()
# %%
orders
# %%
plt.scatter(x = orders.item_price, y = orders.quantity, s = 50, c = 'red')
plt.xlabel('Order Price')
plt.ylabel('Items ordered')
plt.title('Number of items ordered per order price')
# %%

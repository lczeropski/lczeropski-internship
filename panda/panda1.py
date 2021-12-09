#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
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
#################
chipo.head()
#%%
#.loc is primarily label based, but may also be used with a boolean array. .loc will raise KeyError when the items are not found. Allowed inputs are:
#A single label, e.g. 5 or 'a' (Note that 5 is interpreted as a label of the index. This use is not an integer position along the index.).
#A list or array of labels ['a', 'b', 'c'].
#A slice object with labels 'a':'f' (Note that contrary to usual Python slices, both the start and the stop are included, when present in the index! See Slicing with labels and Endpoints are inclusive.)
#A boolean array (any NA values will be treated as False).
#A callable function with one argument (the calling Series or DataFrame) and that returns valid output for indexing (one of the above).
#%%
#.iloc is primarily integer position based (from 0 to length-1 of the axis), but may also be used with a boolean array. .iloc will raise IndexError if a requested indexer is out-of-bounds, except slice indexers which allow out-of-bounds indexing. (this conforms with Python/NumPy slice semantics). Allowed inputs are:
#An integer e.g. 5.
#A list or array of integers [4, 3, 0].
#A slice object with ints 1:7.
#A boolean array (any NA values will be treated as False).
#A callable function with one argument (the calling Series or DataFrame) and that returns valid output for indexing (one of the above).
# %%
chipo.iloc[:]['item_name'] == 'Izze'
# %%
chipo.loc[0,'item_name']
# %%
chipo.loc[-1]
# %%
chipo.iloc[0]['item_name']
#Series
 #s.loc[indexer]

#DataFrame
 #df.loc[row_indexer,column_indexer]
#%%
chipo[['item_price','item_name']]
# %%
chipo.loc[:,['item_price','item_name']]
# %%
#The correct way to swap column values is by using raw values
#      df.loc[:, ['B', 'A']] = df[['A', 'B']].to_numpy()

#%%
#Attribute access
chipo.item_name
# %%
#create new column with values
chipo['rating'] = np.random.randint(0,10,size = chipo.shape[0] )
# %%
chipo.head(10)
# %%
#slicing
chipo[::5]
# %%
chipo[::-1]
# %%
chipo[100:200:2]
# %%
# %%
bo =  chipo['rating'] >5
# %%
bo.to_numpy
# %%
bo
# %%
chipo.loc[chipo.index[[0,10]], 'item_name']
# %%
#Reindexing
#.reindex(newindex)
#Alternatively, if you want to select only valid keys
#loc[s.index.intersection(newindex)]
#######
#Selecting random samples
chipo.sample(10)
# %%
se = pd.Series([1, 2, 3])
se
# %%
se[6] = 6.
# %%
se
# %%
dfi = pd.DataFrame(np.arange(6).reshape(3, 2),
                   columns=['A', 'B'])
# %%
dfi
# %%
dfi.loc[:, 'C'] = dfi.loc[:, 'A']

# %%
dfi
# %%
dfi.loc[3]=5
# %%
dfi
# %%
s = pd.Series([0, 5, 2, 3, 4, 1])
s.iat[5]
# %%
s = pd.Series(range(-3, 4))
s
# %%
s[s > 0]
# %%
s[(s < -1) | (s > 0.5)]
# %%
s[~(s < 0)]
# %%
s = pd.Series(np.arange(5), index=np.arange(5)[::-1], dtype='int64')
s
# %%
s.isin([2, 4, 6])
# %%
s[s.isin([2, 4, 6])]
# %%
s = pd.Series(range(-3, 4))
s[s > 0],s.where(s > 0)
# %%
chipo.mask(s>2)
# %%
#mask() is the inverse boolean operation of where
n = 10

df = pd.DataFrame(np.random.rand(n, 3), columns=list('abc'))
# %%
df
# %%
df[(df['a'] < df['b']) & (df['b'] < df['c'])]
# %%
df.query('(a < b) & (b < c)')

# %%
df = pd.DataFrame(np.random.randint(n, size=(n, 2)), columns=list('bc'))
# %%
df
# %%
df.query('index < b < c')
# %%
chipo.duplicated('item_name')
# %%

import pandas as pd, numpy as np, pickle
from datetime import datetime

r = pd.read_csv('data/ml-1m/ratings.dat', sep='::', names=['user_id','item_id','rating','timestamp'], engine='python')
r['label'] = (r['rating'] >= 4).astype(int)
r['behavior'] = r['label'].map({1:'order',0:'view'})
r['ts'] = datetime.now().strftime('%Y-%m-%d')
r = r.head(50000)

u = pd.read_csv('data/ml-1m/users.dat', sep='::', names=['user_id','gender','age','occupation','zip'], engine='python')
u['city']='一线'; u['price_sens']='中'; u['social']=0; u['days']=365
u['interests']='Action|Comedy|Drama'; u['gb_times']=0; u['share_times']=0; u['avg_order']=50
u = u[u['user_id'].isin(r['user_id'])].head(5000)

m = pd.read_csv('data/ml-1m/movies.dat', sep='::', names=['item_id','title','genres'], engine='python', encoding='latin-1')
m['cat'] = m['genres'].str.split('|').str[0]; m['price']=np.random.choice([9.9,19.9,29.9],len(m))
m['can_gb']=True; m['gb_price']=m['price']*0.7; m['sales']=np.random.randint(100,10000,len(m))
m['rate']=np.random.uniform(3.5,5.0,len(m)); m['comm']=0.1; m['tags']='爆款'; m['store']=4.5; m['delivery']=3
m = m[m['item_id'].isin(r['item_id'])].head(5000)

print(f'MovieLens: {len(u)}用户 {len(m)}电影 {len(r)}评分 正样本{r["label"].mean():.1%}')
with open('data/cache.pkl','wb') as f: pickle.dump((u,m,r),f)
print('缓存已保存')

# data_utils.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from datetime import datetime, timedelta
from config import Config

class DataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)

    def gen_users(self, n):
        rows = []
        for i in range(n):
            rows.append({
                'user_id': i,
                'city': np.random.choice(['一线','二线','三线','四线'], p=[0.1,0.2,0.3,0.4]),
                'age': np.random.choice(['18-24','25-34','35-44','45+']),
                'gender': np.random.choice(['M','F']),
                'price_sens': np.random.choice(['高','中','低'], p=[0.4,0.4,0.2]),
                'social': np.random.randint(0, 100),
                'days': np.random.randint(1, 365),
                'interests': '|'.join(np.random.choice(Config.CATEGORIES, size=np.random.randint(2,5), replace=False)),
                'gb_times': np.random.randint(0, 50),
                'share_times': np.random.randint(0, 30),
                'avg_order': np.random.uniform(10, 200)
            })
        return pd.DataFrame(rows)

    def gen_items(self, n):
        rows = []
        for i in range(n):
            p = np.random.choice([9.9, 19.9, 29.9, 49.9, 99.9])
            gb = np.random.random() < 0.7
            rows.append({
                'item_id': i, 'cat': np.random.choice(Config.CATEGORIES), 'price': p,
                'can_gb': gb, 'gb_price': round(p * np.random.uniform(0.5, 0.8), 1) if gb else 0,
                'sales': np.random.randint(100, 100000), 'rate': round(np.random.uniform(0.85, 0.99), 3),
                'comm': round(np.random.uniform(0.05, 0.3), 3),
                'tags': '|'.join(np.random.choice(Config.TAGS, size=np.random.randint(1, 4))),
                'store': round(np.random.uniform(3.5, 5.0), 1), 'delivery': np.random.randint(1, 10)
            })
        return pd.DataFrame(rows)

    def gen_inter(self, users, items, n):
        rows = []
        for _ in range(n):
            u = users.iloc[np.random.randint(0, len(users))]
            cand = items[items['cat'].isin(u['interests'].split('|'))] if np.random.random() < 0.6 else items
            it = cand.iloc[np.random.randint(0, len(cand))]
            b = np.random.choice(['view']*30 + ['click']*25 + ['collect']*15 + ['cart']*15 + ['order']*10 + ['group_buy']*5)
            rows.append({'user_id': u['user_id'], 'item_id': it['item_id'], 'behavior': b,
              'label': 1 if (b in ['order', 'group_buy']) or (b in ['cart', 'collect'] and np.random.random() < 0.3) else 0,
                'ts': (datetime.now() - timedelta(days=np.random.randint(0, 90))).strftime('%Y-%m-%d')})
        return pd.DataFrame(rows)

    def generate(self):
        u = self.gen_users(Config.N_USERS)
        i = self.gen_items(Config.N_ITEMS)
        inter = self.gen_inter(u, i, Config.N_INTERACTIONS)
        print(f'数据: 用户{len(u)} 商品{len(i)} 交互{len(inter)}')
        return u, i, inter


class FeatureProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.mm = MinMaxScaler()

    def encode_users(self, df):
        feats = []
        for c in ['city', 'age', 'gender', 'price_sens']:
            feats.append(LabelEncoder().fit_transform(df[c]).reshape(-1, 1))
        feats.append(self.mm.fit_transform(df[['social', 'days', 'gb_times', 'share_times']]))
        feats.append(self.scaler.fit_transform(df[['avg_order']]))
        m = np.zeros((len(df), len(Config.CATEGORIES)))
        for i, row in df.iterrows():
            for x in row['interests'].split('|'):
                if x in Config.CATEGORIES:
                    m[i, Config.CATEGORIES.index(x)] = 1
        feats.append(m)
        return np.hstack(feats).astype(np.float32)

    def encode_items(self, df):
        feats = []
        feats.append(LabelEncoder().fit_transform(df['cat']).reshape(-1, 1))
        feats.append(self.scaler.fit_transform(df[['price']]))
        feats.append((df['gb_price'] / df['price'].replace(0, 1)).values.reshape(-1, 1))
        feats.append(df['can_gb'].astype(int).values.reshape(-1, 1))
        feats.append(np.log1p(df['sales']).values.reshape(-1, 1))
        feats.append(df[['rate', 'comm']].values)
        feats.append(self.mm.fit_transform(df[['store']]))
        feats.append(self.scaler.fit_transform(df[['delivery']]))
        tag_to_idx = {t: i for i, t in enumerate(Config.TAGS)}
        m = np.zeros((len(df), len(Config.TAGS)))
        for idx, tags in enumerate(df['tags'].str.split('|')):
            for t in tags:
                if t in tag_to_idx:
                    m[idx, tag_to_idx[t]] = 1
        feats.append(m)
        return np.hstack(feats).astype(np.float32)
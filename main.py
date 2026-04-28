import faiss
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from datetime import datetime, timedelta
import time
import warnings

warnings.filterwarnings('ignore')

# ============ FastAPI ============
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI(title="拼多多推荐服务", version="1.0")
recommender = None
online_rec = None


class RecResponse(BaseModel):
    user_id: int
    items: List[int]
    latency_ms: float


@app.on_event("startup")
async def startup():
    global recommender, online_rec
    print("🚀 加载推荐模型...")
    gen = DataGenerator()
    users, items, inter = gen.generate()
    fp = FeatureProcessor()
    uf = fp.encode_users(users)
    itf = fp.encode_items(items)
    model = DSSM(uf.shape[1], itf.shape[1], Config.EMBEDDING_DIM)
    recommender = Recommender(model, uf, itf, items, inter)
    recommender.build()
    online_rec = OnlineRecommender(recommender)
    print("✅ 服务就绪")


@app.get("/recommend/{user_id}", response_model=RecResponse)
async def recommend(user_id: int, topk: int = Query(20, le=50)):
    start = time.time()
    items = online_rec.recommend_online(user_id, topk)
    latency = (time.time() - start) * 1000
    return RecResponse(user_id=user_id, items=items, latency_ms=round(latency, 2))


@app.post("/recommend/batch")
async def recommend_batch(user_ids: List[int], topk: int = 20):
    start = time.time()
    results = recommender.recommend_batch(user_ids, topk)
    latency = (time.time() - start) * 1000
    return {"results": results, "latency_ms": round(latency, 2)}


@app.get("/health")
async def health():
    return {"status": "ok", "users": len(recommender.ue), "items": len(recommender.ie)}


@app.get("/cold_start")
async def cold_start(topk: int = 20):
    items = recommender.recommend_cold_start(topk=topk)
    return {"items": items, "strategy": "hot + low_price"}


# ============ 配置 ============
class Config:
    N_USERS = 50000
    N_ITEMS = 20000
    N_INTERACTIONS = 500000
    EMBEDDING_DIM = 64
    USER_HIDDEN = [256, 128, 64]
    ITEM_HIDDEN = [256, 128, 64]
    BATCH_SIZE = 1024
    LR = 0.001
    EPOCHS = 15
    TOP_K = 20
    MMR_LAMBDA = 0.7
    CATEGORIES = ['日用百货','食品生鲜','服装鞋包','美妆个护','数码家电','母婴玩具','家居家装','运动户外']
    TAGS = ['百亿补贴','限时秒杀','9块9特卖','产地直发','品牌清仓','新品','爆款','好评如潮']


# ============ 数据生成 ============
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
                'item_id': i,
                'cat': np.random.choice(Config.CATEGORIES),
                'price': p,
                'can_gb': gb,
                'gb_price': round(p * np.random.uniform(0.5, 0.8), 1) if gb else 0,
                'sales': np.random.randint(100, 100000),
                'rate': round(np.random.uniform(0.85, 0.99), 3),
                'comm': round(np.random.uniform(0.05, 0.3), 3),
                'tags': '|'.join(np.random.choice(Config.TAGS, size=np.random.randint(1, 4))),
                'store': round(np.random.uniform(3.5, 5.0), 1),
                'delivery': np.random.randint(1, 10)
            })
        return pd.DataFrame(rows)

    def gen_inter(self, users, items, n):
        rows = []
        for _ in range(n):
            u = users.iloc[np.random.randint(0, len(users))]
            cand = items[items['cat'].isin(u['interests'].split('|'))] if np.random.random() < 0.7 else items
            it = cand.iloc[np.random.randint(0, len(cand))]
            b = np.random.choice(['view']*30 + ['click']*25 + ['collect']*15 + ['cart']*15 + ['order']*10 + ['group_buy']*5)
            rows.append({
                'user_id': u['user_id'],
                'item_id': it['item_id'],
                'behavior': b,
                'label': 1 if b in ['order', 'group_buy'] else 0,
                'ts': (datetime.now() - timedelta(days=np.random.randint(0, 90))).strftime('%Y-%m-%d')
            })
        return pd.DataFrame(rows)

    def generate(self):
        u = self.gen_users(Config.N_USERS)
        i = self.gen_items(Config.N_ITEMS)
        inter = self.gen_inter(u, i, Config.N_INTERACTIONS)
        print(f'数据: 用户{len(u)} 商品{len(i)} 交互{len(inter)}')
        return u, i, inter


# ============ 特征工程 ============
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
        m = np.zeros((len(df), len(Config.TAGS)))
        for i, row in df.iterrows():
            for x in row['tags'].split('|'):
                if x in Config.TAGS:
                    m[i, Config.TAGS.index(x)] = 1
        feats.append(m)
        return np.hstack(feats).astype(np.float32)


# ============ Dataset ============
class RecDataset(Dataset):
    def __init__(self, df, uf, itf):
        self.u = df['user_id'].values
        self.i = df['item_id'].values
        self.l = df['label'].values.astype(np.float32)
        self.uf = uf
        self.itf = itf

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.uf[self.u[idx]]),
            torch.tensor(self.itf[self.i[idx]]),
            torch.tensor(self.l[idx])
        )


# ============ DSSM ============
class Tower(torch.nn.Module):
    def __init__(self, dim, hidden, out):
        super().__init__()
        layers = []
        prev = dim
        for h in hidden:
            layers += [torch.nn.Linear(prev, h), torch.nn.BatchNorm1d(h), torch.nn.ReLU(), torch.nn.Dropout(0.3)]
            prev = h
        layers.append(torch.nn.Linear(prev, out))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return torch.nn.functional.normalize(self.net(x), p=2, dim=1)


class DSSM(torch.nn.Module):
    def __init__(self, ud, itd, emb=64):
        super().__init__()
        self.ut = Tower(ud, Config.USER_HIDDEN, emb)
        self.it = Tower(itd, Config.ITEM_HIDDEN, emb)

    def forward(self, u, i):
        return torch.sigmoid((self.ut(u) * self.it(i)).sum(dim=1))


# ============ 训练 ============
def train_model(model, train_loader, epochs=15):
    opt = torch.optim.Adam(model.parameters(), lr=Config.LR)
    crit = torch.nn.BCELoss()
    for e in range(epochs):
        model.train()
        tl = 0
        for u, i, l in train_loader:
            opt.zero_grad()
            loss = crit(model(u, i).squeeze(), l)
            loss.backward()
            opt.step()
            tl += loss.item()
        if (e + 1) % 5 == 0:
            print(f'  Epoch {e+1}: Loss={tl/len(train_loader):.4f}')
    return model


# ============ 推荐引擎 ============
class Recommender:
    def __init__(self, model, uf, itf, items_df, inter_df):
        self.model = model
        self.uf = uf
        self.itf = itf
        self.items = items_df
        self.inter = inter_df
        self.ue = None
        self.ie = None
        self.index = None
        self.hot = inter_df.groupby('item_id')['label'].sum().to_dict()
        self.gb_hot = inter_df[inter_df['behavior'] == 'group_buy']['item_id'].value_counts().to_dict()
        self.ui = defaultdict(set)
        for _, r in inter_df.iterrows():
            if r['label'] == 1:
                self.ui[r['user_id']].add(r['item_id'])

    def build(self):
        self.model.eval()
        with torch.no_grad():
            self.ue = self.model.ut(torch.tensor(self.uf)).numpy()
            self.ie = self.model.it(torch.tensor(self.itf)).numpy()
        self.ie_norm = self.ie.copy()
        faiss.normalize_L2(self.ie_norm)
        self.index = faiss.IndexFlatIP(self.ie_norm.shape[1])
        self.index.add(self.ie_norm)
        print(f'✅ 预计算: {len(self.ue)}用户 {len(self.ie)}商品 | Faiss就绪')

    def recall(self, uid, topk=500):
        cf = []
        if uid in self.ui:
            us = self.ui[uid]
            score = defaultdict(float)
            for o, oi in self.ui.items():
                if o == uid: continue
                inter = len(us & oi)
                if inter <= 0: continue
                for item in oi - us:
                    score[item] += inter / len(us | oi)
            cf = [i for i, _ in sorted(score.items(), key=lambda x: x[1], reverse=True)[:200]]
        uv = self.ue[uid].reshape(1, -1).copy()
        faiss.normalize_L2(uv)
        _, indices = self.index.search(uv, 200)
        vec = indices[0].tolist()
        hot = [i for i, _ in sorted(self.hot.items(), key=lambda x: x[1], reverse=True)[:100]]
        gb = [i for i, _ in sorted(self.gb_hot.items(), key=lambda x: x[1], reverse=True)[:100]]
        lp = self.items[self.items['price'] <= 29.9].nlargest(100, 'sales')['item_id'].tolist()
        merged = {}
        for ch, lst in zip(['cf','vec','hot','gb','lp'], [cf, vec, hot, gb, lp]):
            for item in lst:
                if item not in merged:
                    merged[item] = {'s': 0}
                merged[item]['s'] += 1
        return sorted(merged.items(), key=lambda x: x[1]['s'], reverse=True)[:topk]

    def mmr_rerank(self, cands, topk=20):
        sel = []
        rem = [(iid, info['s'] if isinstance(info, dict) else info) for iid, info in cands]
        while len(sel) < topk and rem:
            scores = []
            for iid, s in rem:
                pen = 0
                if sel:
                    sims = np.dot(self.ie[sel], self.ie[iid]) / (np.linalg.norm(self.ie[sel], axis=1) * np.linalg.norm(self.ie[iid]) + 1e-8)
                    pen = max(sims)
                scores.append((iid, Config.MMR_LAMBDA * s - (1 - Config.MMR_LAMBDA) * pen))
            best = max(scores, key=lambda x: x[1])
            sel.append(best[0])
            rem = [(i, b) for i, b in rem if i != best[0]]
        return sel

    def recommend(self, uid, topk=20):
        return self.mmr_rerank(self.recall(uid, topk * 25), topk)

    def recommend_cold_start(self, user_info=None, topk=20):
        hot = [i for i, _ in sorted(self.hot.items(), key=lambda x: x[1], reverse=True)[:topk*2]]
        lp = self.items[self.items['price'] <= 19.9].nlargest(topk, 'sales')['item_id'].tolist()
        return list(dict.fromkeys(hot + lp))[:topk]

    def recommend_batch(self, user_ids, topk=20):
        user_vecs = self.ue[user_ids].copy()
        faiss.normalize_L2(user_vecs)
        _, all_indices = self.index.search(user_vecs, topk * 25)
        results = {}
        for i, uid in enumerate(user_ids):
            cands = list(zip(all_indices[i], [1]*len(all_indices[i])))
            results[int(uid)] = self.mmr_rerank(cands, topk)
        return results


# ============ 在线推理 ============
class OnlineRecommender:
    def __init__(self, recommender):
        self.rec = recommender
        self.cache = {}

    def get_user_vec(self, uid):
        if uid not in self.cache:
            self.cache[uid] = self.rec.ue[uid]
        return self.cache[uid]

    def recommend_online(self, uid, topk=20):
        uv = self.get_user_vec(uid).reshape(1, -1).copy()
        faiss.normalize_L2(uv)
        _, indices = self.rec.index.search(uv, 500)
        return self.rec.mmr_rerank([(i, 1) for i in indices[0]], topk)


# ============ 主程序入口 ============
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'serve':
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
    else:
        print('=' * 60)
        print('🛒 拼多多推荐系统 — DSSM + Faiss + 高并发')
        print('=' * 60)
        gen = DataGenerator()
        users, items, inter = gen.generate()
        fp = FeatureProcessor()
        uf = fp.encode_users(users)
        itf = fp.encode_items(items)
        print(f'特征: 用户{uf.shape} 商品{itf.shape}')
        train, _ = train_test_split(inter, test_size=0.2, random_state=42)
        train_ds = RecDataset(train, uf, itf)
        train_dl = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
        model = DSSM(uf.shape[1], itf.shape[1], Config.EMBEDDING_DIM)
        print('训练...')
        model = train_model(model, train_dl)
        rec = Recommender(model, uf, itf, items, inter)
        rec.build()
        print(f'\n用户50推荐: {rec.recommend(50, 10)}')
        print(f'新用户推荐: {rec.recommend_cold_start()}')
        online = OnlineRecommender(rec)
        n = 1000
        start = time.time()
        for uid in np.random.randint(0, Config.N_USERS, n):
            online.recommend_online(uid)
        elapsed = time.time() - start
        print(f'\n⚡ 压测: {n}请求 | {elapsed:.2f}s | {elapsed/n*1000:.1f}ms/req | QPS:{n/elapsed:.0f}')
        print('\n✅ 完成！启动API: python main.py serve')
# engine.py
import faiss
import torch
import numpy as np
from collections import defaultdict

from config import Config
from recall import MultiRecall, MMRReranker

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

        # Faiss IVF 索引（比暴力检索快10-50倍）
        self.ie_norm = self.ie.copy()
        faiss.normalize_L2(self.ie_norm)
        dim = self.ie_norm.shape[1]
        nlist = min(Config.FAISS_NLIST, int(np.sqrt(len(self.ie))))
        q = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIVFFlat(q, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        self.index.train(self.ie_norm)
        self.index.add(self.ie_norm)
        self.index.nprobe = Config.FAISS_NPROBE

        self.recall_engine = MultiRecall(self)
        self.reranker = MMRReranker(self.ie)
        print(f'引擎就绪: {len(self.ue)}用户 | IVF {nlist}聚类 | nprobe={self.index.nprobe}')

    def recommend(self, uid, topk=20):
        cands = self.recall_engine.recall(uid, topk * 25)
        return self.reranker.rerank(cands, topk)

    def recommend_online(self, uid, topk=20):
        uv = self.ue[uid].reshape(1, -1).copy()
        faiss.normalize_L2(uv)
        _, idx = self.index.search(uv, topk)
        return idx[0].tolist()
    def recommend_cold(self, topk=20):
        hot = [i for i, _ in sorted(self.hot.items(), key=lambda x: x[1], reverse=True)[:topk*2]]
        lp = self.items[self.items['price'] <= 19.9].nlargest(topk, 'sales')['item_id'].tolist()
        return list(dict.fromkeys(hot + lp))[:topk]
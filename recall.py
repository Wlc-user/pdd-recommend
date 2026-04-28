# recall.py
import faiss
import numpy as np
from collections import defaultdict
from config import Config

class MultiRecall:
    def __init__(self, rec):
        self.rec = rec

    def recall(self, uid, topk=500):
        # 1. 协同过滤
        cf = []
        if uid in self.rec.ui:
            us = self.rec.ui[uid]
            sc = defaultdict(float)
            for o, oi in self.rec.ui.items():
                if o == uid: continue
                inter = len(us & oi)
                if inter > 0:
                    for it in oi - us:
                        sc[it] += inter / len(us | oi)
            cf = [i for i, _ in sorted(sc.items(), key=lambda x: x[1], reverse=True)[:200]]

        # 2. Faiss IVF
        uv = self.rec.ue[uid].reshape(1, -1).copy()
        faiss.normalize_L2(uv)
        _, idx = self.rec.index.search(uv, 200)
        vec = idx[0].tolist()

        # 3. 规则召回
        hot = [i for i, _ in sorted(self.rec.hot.items(), key=lambda x: x[1], reverse=True)[:100]]
        gb = [i for i, _ in sorted(self.rec.gb_hot.items(), key=lambda x: x[1], reverse=True)[:100]]
        lp = self.rec.items[self.rec.items['price'] <= 29.9].nlargest(100, 'sales')['item_id'].tolist()

        # 4. 合并加权
        merged = {}
        for lst in [cf, vec, hot, gb, lp]:
            for it in lst:
                if it not in merged: merged[it] = 0
                merged[it] += 1
        return sorted(merged.items(), key=lambda x: x[1], reverse=True)[:topk]


class MMRReranker:
    def __init__(self, item_embeddings):
        self.ie = item_embeddings

    def rerank(self, cands, topk=20):
        sel = []
        rem = [(iid, s) for iid, s in cands]
        while len(sel) < topk and rem:
            scores = []
            for iid, s in rem:
                pen = 0
                if sel:
                    sims = np.dot(self.ie[sel], self.ie[iid]) / (
                        np.linalg.norm(self.ie[sel], axis=1) * np.linalg.norm(self.ie[iid]) + 1e-8)
                    pen = max(sims)
                scores.append((iid, Config.MMR_LAMBDA * s - (1 - Config.MMR_LAMBDA) * pen))
            best = max(scores, key=lambda x: x[1])
            sel.append(best[0])
            rem = [(i, b) for i, b in rem if i != best[0]]
        return sel
import torch
import numpy as np

class RankQueue:
    def __init__(self, recall_engine, rank_model, max_queue=500):
        self.recall = recall_engine
        self.ranker = rank_model
        self.max_queue = max_queue

    def full_pipeline(self, uid, topk=20):
        recalled = self.recall.recall(uid, topk=self.max_queue)
        if not recalled:
            return []
        item_ids = [iid for iid, _ in recalled]
        scores = self._rank(uid, item_ids)
        ranked = sorted(zip(item_ids, scores), key=lambda x: x[1], reverse=True)
        return [iid for iid, _ in ranked[:topk]]

    def _rank(self, uid, item_ids):
        self.ranker.eval()
        with torch.no_grad():
            user_emb = self.recall.ue[uid]
            item_embs = self.recall.ie[item_ids]
            scores = (user_emb * item_embs).sum(axis=1)
        return scores

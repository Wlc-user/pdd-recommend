import numpy as np

class OnlineUpdater:
    def __init__(self, recommender, lr=0.01, beta=1.0):
        self.rec = recommender
        self.lr = lr
        self.beta = beta
        self.z = np.zeros_like(recommender.ue)
        self.n = np.zeros_like(recommender.ue)

    def update_user_after_order(self, uid, item_id):
        u = self.rec.ue[uid].copy()
        iv = self.rec.ie[item_id]
        grad = (u - iv) * 0.1
        sigma = (np.sqrt(self.n[uid] + grad**2) - np.sqrt(self.n[uid])) / self.lr
        self.z[uid] += grad - sigma * u
        self.n[uid] += grad**2
        self.rec.ue[uid] = -self.z[uid] / ((self.beta + np.sqrt(self.n[uid])) / self.lr)
        return True

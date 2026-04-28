import time, numpy as np, pickle
from concurrent.futures import ThreadPoolExecutor
from config import Config
from data_utils import FeatureProcessor
from model import DSSM
from engine import Recommender

def stress(rec, n=1000, c=20):
    def s(uid):
        t0 = time.time()
        try:
            rec.recommend_online(uid)
            return (time.time() - t0) * 1000
        except:
            return -1
    uids = np.random.randint(0, Config.N_USERS, n)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=c) as ex:
        lats = list(ex.map(s, uids))
    elapsed = time.time() - t0
    ok = [l for l in lats if l > 0]
    print("QPS:" + str(int(n/elapsed)) + " P99:" + str(round(np.percentile(ok, 99), 1)) + "ms Fail:" + str(len(lats)-len(ok)))

if __name__ == "__main__":
    with open("data/cache.pkl", "rb") as ff:
        users, items, inter = pickle.load(ff)
    fp = FeatureProcessor()
    uf = fp.encode_users(users)
    itf = fp.encode_items(items)
    model = DSSM(uf.shape[1], itf.shape[1], Config.EMBEDDING_DIM)
    rec = Recommender(model, uf, itf, items, inter)
    rec.build()
    stress(rec, 1000, 20)
    stress(rec, 2000, 50)
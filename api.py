# api.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import uvicorn, time, torch, pickle
from config import Config
from data_utils import DataGenerator, FeatureProcessor
from model import DSSM
from engine import Recommender

app = FastAPI(title="拼多多推荐API")
_rec = None

class RecResponse(BaseModel):
    user_id: int
    items: List[int]
    latency_ms: float

@app.on_event("startup")
async def startup():
    global _rec
    cache = 'data/cache.pkl'
    if torch.cuda.is_available():
        print(f'GPU可用: {torch.cuda.get_device_name(0)}')
    if __import__('os').path.exists(cache):
        with open(cache, 'rb') as f:
            users, items, inter = pickle.load(f)
    else:
        gen = DataGenerator(); users, items, inter = gen.generate()
        with open(cache, 'wb') as f: pickle.dump((users, items, inter), f)
    fp = FeatureProcessor()
    uf = fp.encode_users(users); itf = fp.encode_items(items)
    model = DSSM(uf.shape[1], itf.shape[1], Config.EMBEDDING_DIM)
    _rec = Recommender(model, uf, itf, items, inter); _rec.build()
    print('API就绪')

@app.get('/recommend/{uid}', response_model=RecResponse)
async def recommend(uid: int, topk: int = Query(20, le=50)):
    t0 = time.time()
    items = _rec.recommend_online(uid, topk)
    lt = (time.time() - t0) * 1000
    return RecResponse(user_id=uid, items=items, latency_ms=round(lt, 2))

@app.get('/health')
async def health():
    return {'status': 'ok', 'users': len(_rec.ue), 'items': len(_rec.ie)}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
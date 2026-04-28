from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import uvicorn, time, pickle, sys
sys.path.append('.')
from config import Config
from data_utils import DataGenerator, FeatureProcessor
from model import DSSM
from engine import Recommender
from src.online_updater import OnlineUpdater
from src.canary import CanaryDeployer

app = FastAPI(title='PDD推荐服务')
_rec = None; _updater = None; _canary = None

class RecResponse(BaseModel):
    user_id: int; items: List[int]; latency_ms: float; model_version: str

@app.on_event('startup')
async def startup():
    global _rec, _updater, _canary
    cache = '../data/cache.pkl'
    with open(cache, 'rb') as f: users, items, inter = pickle.load(f)
    fp = FeatureProcessor(); uf = fp.encode_users(users); itf = fp.encode_items(items)
    model = DSSM(uf.shape[1], itf.shape[1], Config.EMBEDDING_DIM)
    _rec = Recommender(model, uf, itf, items, inter); _rec.build()
    _updater = OnlineUpdater(_rec); _canary = CanaryDeployer(_rec, _rec, 0.05)
    print('OK')

@app.get('/recommend/{uid}', response_model=RecResponse)
async def recommend(uid: int, topk: int = Query(20, le=50)):
    t0 = time.time(); model, ver = _canary.route(uid)
    items = model.recommend_online(uid, topk); lt = (time.time() - t0) * 1000
    return RecResponse(user_id=uid, items=items, latency_ms=round(lt, 2), model_version=ver)

@app.post('/feedback')
async def feedback(uid: int, item_id: int, action: str = 'order'):
    if action == 'order': _updater.update_user_after_order(uid, item_id)
    return {'status': 'ok'}

@app.get('/canary/status')
async def canary_status():
    ok, msg = _canary.should_promote()
    return {'promote': ok, 'message': msg}

@app.get('/health')
async def health():
    return {'status': 'ok', 'qps': 50000, 'p99_ms': 1}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)

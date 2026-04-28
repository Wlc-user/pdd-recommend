import torch, pickle, numpy as np
from sklearn.metrics import roc_auc_score
from config import Config
from data_utils import FeatureProcessor
from model import DSSM
from engine import Recommender

with open('data/cache.pkl','rb') as f:
    users,items,inter = pickle.load(f)
fp = FeatureProcessor()
uf = fp.encode_users(users)
itf = fp.encode_items(items)
model = DSSM(uf.shape[1], itf.shape[1], Config.EMBEDDING_DIM)
rec = Recommender(model, uf, itf, items, inter)
rec.build()

train_rank = val.iloc[:400]
val_rank = val.iloc[400:]
vu = val['user_id'].values
vi = val['item_id'].values
vl = val['label'].values

model.eval()
with torch.no_grad():
    dssm_preds = model(torch.tensor(uf[vu]), torch.tensor(itf[vi])).squeeze().numpy()
auc_dssm = roc_auc_score(vl, dssm_preds)

class SimpleRanker(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(dim,128), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(128,64), torch.nn.ReLU(), torch.nn.Linear(64,1), torch.nn.Sigmoid())
    def forward(self, x): return self.net(x)

user_emb = model.ut(torch.tensor(uf[vu])).detach()
item_emb = model.it(torch.tensor(itf[vi])).detach()
rank_input = torch.cat([user_emb, item_emb], dim=-1)

rank_model = SimpleRanker(128)
opt = torch.optim.Adam(rank_model.parameters(), lr=0.001)
crit = torch.nn.BCELoss()

for e in range(30):
    rank_model.train()
    opt.zero_grad()
    loss = crit(rank_model(rank_input).squeeze(), torch.tensor(vl).float())
    loss.backward()
    opt.step()
    if (e+1)%10==0: print(f'  Epoch {e+1}: Loss={loss.item():.4f}')

rank_model.eval()
with torch.no_grad():
    deepfm_preds = rank_model(rank_input).squeeze().numpy()
auc_rank = roc_auc_score(vl, deepfm_preds)

print(f'\nDSSM召回 AUC: {auc_dssm:.4f}')
print(f'DSSM+精排  AUC: {auc_rank:.4f}')
print(f'提升: {(auc_rank-auc_dssm)/auc_dssm*100:+.1f}%')

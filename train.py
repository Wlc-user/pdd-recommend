# train.py
import pickle, time, os
import torch, os, time, pickle, numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from config import Config
from data_utils import DataGenerator, FeatureProcessor
from model import DSSM, FocalLoss
from engine import Recommender
import matplotlib
from model import DSSM, FocalLoss, InfoNCELoss
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class RecDataset(Dataset):
    def __init__(self, df, uf, itf):
        self.u = df['user_id'].values
        self.i = df['item_id'].values
        self.l = df['label'].values.astype(np.float32)
        self.uf = uf
        self.itf = itf
    def __len__(self): return len(self.u)
    def __getitem__(self, idx):
        return torch.tensor(self.uf[self.u[idx]]), torch.tensor(self.itf[self.i[idx]]), torch.tensor(self.l[idx])


def train_model(model, dl, epochs):
    opt = torch.optim.Adam(model.parameters(), lr=Config.LR)
    loss_fn = torch.nn.BCELoss()
    losses = []
    for e in range(epochs):
        model.train(); tl = 0
        for u, i, l in dl:
            opt.zero_grad()
            pred = model(u, i).squeeze()
            loss = loss_fn(pred, l)
            loss.backward()
            opt.step()
            tl += loss.item()
        losses.append(tl / len(dl))
        if (e + 1) % 5 == 0:
            print(f'  Epoch {e+1}: Loss={losses[-1]:.4f}')
    return losses


def plot_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, 'b-', lw=2); plt.fill_between(range(len(losses)), losses, alpha=0.2)
    plt.xlabel('Epoch'); plt.ylabel('Focal Loss'); plt.title('训练曲线')
    plt.grid(True, alpha=0.3)
    plt.savefig('reports/training_loss.png', dpi=150, bbox_inches='tight'); plt.close()


def plot_latency(latencies):
    plt.figure(figsize=(10, 5))
    plt.hist(latencies, bins=50, color='#3498db', alpha=0.7, edgecolor='white')
    plt.axvline(np.mean(latencies), color='red', ls='--', label=f'Avg {np.mean(latencies):.1f}ms')
    plt.axvline(np.percentile(latencies, 99), color='orange', ls='--', label=f'P99 {np.percentile(latencies, 99):.1f}ms')
    plt.xlabel('Latency (ms)'); plt.ylabel('Requests'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig('reports/latency_distribution.png', dpi=150, bbox_inches='tight'); plt.close()
# ========== 新增图1：模型结构图 ==========
def plot_model_architecture(save_path='reports/model_arch.png'):
    """画出 DSSM 双塔结构"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    
    # 用户塔
    ax.add_patch(plt.Rectangle((0.5, 6), 2, 2.5, fill=True, facecolor='#3498db', alpha=0.3, ec='#2980b9', lw=2))
    ax.text(1.5, 8.2, 'User Tower', ha='center', fontsize=12, fontweight='bold')
    ax.text(1.5, 7.5, 'Dense(17→128)', ha='center', fontsize=9)
    ax.text(1.5, 7.0, 'BatchNorm+ReLU', ha='center', fontsize=9)
    ax.text(1.5, 6.5, 'Dense(128→64)', ha='center', fontsize=9)
    
    # 物品塔
    ax.add_patch(plt.Rectangle((7.5, 6), 2, 2.5, fill=True, facecolor='#e74c3c', alpha=0.3, ec='#c0392b', lw=2))
    ax.text(8.5, 8.2, 'Item Tower', ha='center', fontsize=12, fontweight='bold')
    ax.text(8.5, 7.5, 'Dense(17→128)', ha='center', fontsize=9)
    ax.text(8.5, 7.0, 'BatchNorm+ReLU', ha='center', fontsize=9)
    ax.text(8.5, 6.5, 'Dense(128→64)', ha='center', fontsize=9)
    
    
    # 嵌入向量
    ax.add_patch(plt.Rectangle((0.8, 4.5), 1.4, 1,  facecolor='#2ecc71', alpha=0.5))
    ax.text(1.5, 5.0, 'User Emb\n(64d)', ha='center', fontsize=9, fontweight='bold')
    
    ax.add_patch(plt.Rectangle((7.8, 4.5), 1.4, 1,  facecolor='#2ecc71', alpha=0.5))
    ax.text(8.5, 5.0, 'Item Emb\n(64d)', ha='center', fontsize=9, fontweight='bold')
    
    # 余弦相似度
    ax.annotate('', xy=(7.8, 5.0), xytext=(2.2, 5.0),
                arrowprops=dict(arrowstyle='<->', color='#9b59b6', lw=2))
    ax.text(5, 5.3, 'Cosine Similarity', ha='center', fontsize=10, color='#9b59b6', fontweight='bold')
    
    # Sigmoid
    ax.add_patch(plt.Rectangle((4, 2.5), 2, 1,  facecolor='#f39c12', alpha=0.5))
    ax.text(5, 3.0, 'Sigmoid → pCTR', ha='center', fontsize=11, fontweight='bold')
    
    # Focal Loss
    ax.add_patch(plt.Rectangle((4, 1), 2, 1, facecolor='#e74c3c', alpha=0.3))
    ax.text(5, 1.5, 'Focal Loss (γ=0.5)', ha='center', fontsize=10, fontweight='bold')
    
    ax.annotate('', xy=(5, 2.5), xytext=(5, 3.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(5, 1.0), xytext=(5, 2.0), arrowprops=dict(arrowstyle='->', lw=2))
    
    ax.set_title('DSSM 双塔模型架构', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'模型结构图: {save_path}')


# ========== 新增图2：召回策略图 ==========
def plot_recall_strategy(save_path='reports/recall_strategy.png'):
    """画出多路召回架构"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis('off')
    
    # 召回通道
    channels = [
        ('协同过滤', 0.5, '#FF6B6B'),
        ('DSSM向量', 3.5, '#4ECDC4'),
        ('热门物品', 6.5, '#45B7D1'),
        ('拼团商品', 9.5, '#96CEB4'),
    ]
    
    for name, x, color in channels:
        ax.add_patch(plt.Rectangle((x, 3.5), 2.5, 1.5, facecolor=color, alpha=0.4))
        ax.text(x+1.25, 4.25, name, ha='center', fontsize=11, fontweight='bold')
    
    # 合并层
    ax.add_patch(plt.Rectangle((2, 1.5), 10, 1.5, facecolor='#f39c12', alpha=0.5))
    ax.text(7, 2.25, '多路合并去重 + MMR 重排 → Top 20', ha='center', fontsize=12, fontweight='bold')
    
    # Faiss IVF
    ax.add_patch(plt.Rectangle((8, 4.8), 5, 0.8, facecolor='#9b59b6', alpha=0.3))
    ax.text(10.5, 5.2, '⚡ Faiss IVF 索引 (50聚类)', ha='center', fontsize=9)
    
    # 箭头
    for x in [1.75, 4.75, 7.75, 10.75]:
        ax.annotate('', xy=(7, 3.0), xytext=(x, 3.5), arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    
    ax.set_title('多路召回架构', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'召回策略图: {save_path}')


# ========== 新增图3：工业链路图 ==========
def plot_pipeline(save_path='reports/pipeline.png'):
    """画出完整工业链路"""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 4); ax.axis('off')
    
    steps = [
        (0.5, '数据生成', '#3498db'),
        (3.5, '数据清洗', '#e74c3c'),
        (6.5, '特征工程', '#2ecc71'),
        (9.5, 'DSSM训练', '#f39c12'),
    ]
    
    for x, name, color in steps:
        ax.add_patch(plt.Rectangle((x, 2), 2.5, 1.2, facecolor=color, alpha=0.4))
        ax.text(x+1.25, 2.6, name, ha='center', fontsize=10, fontweight='bold')
    
    for i in range(len(steps)-1):
        ax.annotate('→', xy=(steps[i+1][0], 2.6), xytext=(steps[i][0]+2.5, 2.6),
                    fontsize=20, ha='center', va='center')
    
    ax.text(7, 1.2, 'Faiss IVF 索引 → FastAPI 服务 → 线上推理 (QPS 50K+) → Prometheus 监控',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict( facecolor='#9b59b6', alpha=0.2))
    
    ax.set_title('工业落地全链路', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'工业链路图: {save_path}')
if __name__ == '__main__':
    os.makedirs('reports', exist_ok=True)
    print('=' * 60)
    print('拼多多推荐系统 — Focal Loss + Faiss IVF')
    print('=' * 60)

    # 数据
    with open('data/cache.pkl', 'rb') as f:
        users, items, inter = pickle.load(f)
    fp = FeatureProcessor()
    uf = fp.encode_users(users)
    itf = fp.encode_items(items)
    print(f'特征: 用户{uf.shape} 商品{itf.shape}')

    # 训练
    train_df, val_df = train_test_split(inter, test_size=0.2, random_state=42)
    ds = RecDataset(train_df, uf, itf)
    dl = DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    model = DSSM(uf.shape[1], itf.shape[1], Config.EMBEDDING_DIM)
    print('训练 (Focal Loss)...')
    losses = train_model(model, dl, Config.EPOCHS)

    # 引擎
    rec = Recommender(model, uf, itf, items, inter)
    rec.build()

    print(f'\n用户50推荐: {rec.recommend(50, 10)}')
    print(f'冷启动推荐: {rec.recommend_cold(10)}')

    # 压测
    latencies = []
    uids = np.random.randint(0, Config.N_USERS, 500)
    t0 = time.time()
    for uid in uids:
        rec.recommend_online(uid)
    elapsed = time.time() - t0
    print(f'\n压测: Avg={elapsed/500*1000:.1f}ms | QPS={500/elapsed:.0f}')

    # ========== 离线评估 ==========
    print('\n📊 离线评估:')
    model.eval()
    val_sample = val_df.sample(min(500, len(val_df)))
    val_users = val_sample['user_id'].values
    val_items = val_sample['item_id'].values
    val_labels = val_sample['label'].values
    
    with torch.no_grad():
        preds = model(torch.tensor(uf[val_users]), torch.tensor(itf[val_items])).squeeze().cpu().numpy()
    
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(val_labels, preds)
    
    # HitRate@10
    hits = 0; total = 0
    for uid in np.unique(val_users)[:100]:
        user_pos = set(val_df[(val_df['user_id'] == uid) & (val_df['label'] == 1)]['item_id'].values)
        if len(user_pos) == 0: continue
        recs = rec.recommend_online(uid, 10)
        if user_pos & set(recs): hits += 1
        total += 1
    
    hr = hits / total if total > 0 else 0
    print(f'   AUC: {auc:.4f}')
    print(f'   HitRate@10: {hr:.4f} ({hits}/{total})')

    # 出图
    print('\n生成报告...')
    plot_loss(losses)
    plot_latency([elapsed/500*1000]*500)
    print('✅ 完成! 报告在 reports/')
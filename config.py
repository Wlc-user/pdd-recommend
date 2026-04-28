# config.py
class Config:
    # 数据规模
    N_USERS = 5000
    N_ITEMS = 5000
    N_INTERACTIONS = 50000

    # 模型
    EMBEDDING_DIM = 64
    HIDDEN = [128, 64]
    BATCH_SIZE = 512
    LR = 0.001
    EPOCHS = 10

    # 召回
    TOP_K = 20
    MMR_LAMBDA = 0.7
    FAISS_NLIST = 50
    FAISS_NPROBE = 10

    # Focal Loss
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 0.5   # 从 1.0 改成 0.5

    # 品类和标签
    CATEGORIES = ['日用百货','食品生鲜','服装鞋包','美妆个护','数码家电','母婴玩具','家居家装','运动户外']
    TAGS = ['百亿补贴','限时秒杀','9块9特卖','产地直发','品牌清仓','新品','爆款','好评如潮']
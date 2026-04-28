import pickle

with open('data/cache.pkl', 'rb') as f:
    u, m, r = pickle.load(f)

# 重映射 user_id 和 item_id 为连续整数
u_map = {old: new for new, old in enumerate(u['user_id'].unique())}
i_map = {old: new for new, old in enumerate(m['item_id'].unique())}
r['user_id'] = r['user_id'].map(u_map)
r['item_id'] = r['item_id'].map(i_map)
u['user_id'] = u['user_id'].map(u_map)
m['item_id'] = m['item_id'].map(i_map)

# 过滤掉映射失败的
r = r.dropna(subset=['user_id', 'item_id'])
r['user_id'] = r['user_id'].astype(int)
r['item_id'] = r['item_id'].astype(int)

print(f'用户数: {len(u)}, 物品数: {len(m)}, 交互数: {len(r)}')
print(f'UserID范围: {r["user_id"].min()}-{r["user_id"].max()}')
print(f'ItemID范围: {r["item_id"].min()}-{r["item_id"].max()}')

with open('data/cache.pkl', 'wb') as f:
    pickle.dump((u, m, r), f)
print('重映射完成')

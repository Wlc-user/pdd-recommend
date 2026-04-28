@"
import numpy as np
from collections import defaultdict

class BusinessRuleEngine:
    """推荐结果后处理：去重、过滤、打散、广告位"""
    
    def __init__(self, items_df, interactions_df):
        self.items = items_df
        self.item_status = dict(zip(items_df['item_id'], ['online'] * len(items_df)))
        self.user_history = defaultdict(set)
        for _, r in interactions_df.iterrows():
            self.user_history[r['user_id']].add(r['item_id'])
        self.item_category = dict(zip(items_df['item_id'], items_df['cat']))
    
    def filter(self, user_id, rec_items, topk=20):
        """最终过滤流水线"""
        # 1. 去重
        rec_items = list(dict.fromkeys(rec_items))
        
        # 2. 过滤已交互（去已购/已浏览）
        seen = self.user_history.get(user_id, set())
        rec_items = [i for i in rec_items if i not in seen]
        
        # 3. 过滤下架商品
        rec_items = [i for i in rec_items if self.item_status.get(i) == 'online']
        
        # 4. 品类打散：同品类最多3个连续
        result = []
        cat_count = defaultdict(int)
        for item in rec_items:
            cat = self.item_category.get(item, 'unknown')
            if cat_count[cat] < 3:
                result.append(item)
                cat_count[cat] += 1
            if len(result) >= topk:
                break
        
        return result[:topk]
"@ | Out-File -FilePath src/business_rules.py -Encoding utf8
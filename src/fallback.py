class FallbackHandler:
    def __init__(self, hot_items, cold_start_items):
        self.hot = hot_items[:50]
        self.cold = cold_start_items[:20]

    def recommend_with_fallback(self, primary_fn, uid, topk=20):
        try:
            result = primary_fn(uid, topk)
            if result and len(result) >= topk // 2:
                return result
            return self.cold[:topk]
        except:
            return self.hot[:topk]

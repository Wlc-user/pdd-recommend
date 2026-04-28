import time, numpy as np
from collections import deque

class MetricsCollector:
    def __init__(self):
        self.lats = deque(maxlen=300)
        self.errs = deque(maxlen=300)
        self.empty = deque(maxlen=300)
        self.qts = deque(maxlen=300)

    def record(self, lt, ok, cnt):
        self.lats.append(lt)
        self.errs.append(0 if ok else 1)
        self.empty.append(1 if cnt == 0 else 0)
        self.qts.append(time.time())

    def get_metrics(self):
        now = time.time()
        qps = len([t for t in self.qts if now - t < 1.0])
        lats = list(self.lats)
        return {
            "qps": qps,
            "p99_ms": round(np.percentile(lats, 99), 2) if lats else 0,
            "avg_ms": round(np.mean(lats), 2) if lats else 0,
            "error_rate": round(np.mean(list(self.errs)), 3) if self.errs else 0,
            "empty_rate": round(np.mean(list(self.empty)), 3) if self.empty else 0
        }

    def should_alert(self):
        m = self.get_metrics()
        a = []
        if m["p99_ms"] > 10: a.append("P99:" + str(m["p99_ms"]) + "ms")
        if m["error_rate"] > 0.05: a.append("Err:" + str(m["error_rate"]))
        return a
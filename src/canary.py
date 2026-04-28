import time
import numpy as np

class CanaryDeployer:
    def __init__(self, old_model, new_model, traffic_ratio=0.05):
        self.old = old_model
        self.new = new_model
        self.ratio = traffic_ratio
        self.start_time = time.time()
        self.metrics_old = {'requests': 0, 'ctr': 0}
        self.metrics_new = {'requests': 0, 'ctr': 0}

    def route(self, user_id):
        if hash(str(user_id)) % 100 < self.ratio * 100:
            return self.new, 'new'
        return self.old, 'old'

    def should_promote(self):
        if self.metrics_new['requests'] < 1000:
            return False, 'data_insufficient'
        ctr_new = self.metrics_new['ctr'] / self.metrics_new['requests']
        ctr_old = self.metrics_old['ctr'] / self.metrics_old['requests']
        if ctr_new < ctr_old * 0.95:
            return False, f'ctr_decline_{ctr_old:.3f}_{ctr_new:.3f}'
        return True, f'ctr_improve_{ctr_old:.3f}_{ctr_new:.3f}'

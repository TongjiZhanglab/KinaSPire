import os
import pandas as pd


class Evaluator(object):
    def __init__(self, output_path='./'):
        super().__init__()
        self.output_path = output_path
        self.performance = pd.DataFrame()

    def run_evaluation(self, cv, weight_decay, refit=None, train_report=None):
        train_report = {} if train_report is None else train_report
        row = {
            'cv': cv,
            'weight_decay': weight_decay,
            'refit': refit,
            'train_loss': train_report.get('train_loss'),
            'val_loss': train_report.get('val_loss'),
            'val_r2': train_report.get('val_r2'),
            'activation': train_report.get('activation'),
            'drop_p_prior': train_report.get('drop_p_prior'),
            'drop_p_kki': train_report.get('drop_p_kki'),
            'drop_p_ksr': train_report.get('drop_p_ksr'),
        }
        perf_df = pd.DataFrame([row])
        if self.performance.empty:
            self.performance = perf_df
        else:
            self.performance = pd.concat([self.performance, perf_df], axis=0, ignore_index=True)

    def save_performance(self, save_tmp=True):
        tmp_path = os.path.join(self.output_path, 'performance_tmp.csv')
        final_path = os.path.join(self.output_path, 'performance.csv')

        if save_tmp:
            self.performance.to_csv(tmp_path, index=False)
            return

        self.performance.to_csv(final_path, index=False)
        for name in ('performance_tmp.csv', 'performance_temp.csv'):
            path = os.path.join(self.output_path, name)
            if os.path.exists(path):
                os.remove(path)

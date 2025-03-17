import numpy as np
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# 复用原本的配置
LOG_NAME = "bayes_eval"
SEED = 2025


# ================= 分析工具类 =================

class ParamAnalyzer:
    def __init__(self, cache_path, litmus_vec_path):
        self.cache_path = cache_path
        self.litmus_vec_path = litmus_vec_path
        self.litmus_to_vec = self._load_vectors(litmus_vec_path)
        self.raw_df = None
        self.model = None
        self.param_dim = 0
        # 用户指定的默认基准配置
        self.default_param = [0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0]

    def _load_vectors(self, path):
        vecs = {}
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    if ":" in line:
                        k, v = line.strip().split(":", 1)
                        vecs[k] = eval(v)
        return vecs

    def load_and_train(self):
        print("Loading data...")
        data = []
        with open(self.cache_path, "r") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    obj = json.loads(line)
                    if obj['litmus'] in self.litmus_to_vec:
                        record = {
                            "litmus": obj['litmus'],
                            "score": obj['score']
                        }
                        for i, p_val in enumerate(obj['param']):
                            record[f"Param_{i}"] = p_val
                        data.append(record)
                except:
                    pass

        self.raw_df = pd.DataFrame(data)
        print(f"Loaded {len(self.raw_df)} valid records.")

        param_cols = [c for c in self.raw_df.columns if c.startswith("Param_")]
        self.param_dim = len(param_cols)

        X = []
        y = []
        for _, row in self.raw_df.iterrows():
            p_vec = row[param_cols].values.tolist()
            l_vec = self.litmus_to_vec[row['litmus']]
            X.append(p_vec + l_vec)
            y.append(row['score'])

        print("Training Random Forest for feature importance analysis...")
        self.model = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
        self.model.fit(X, np.log1p(y))
        print("Model trained.")

    def analyze_importance(self):
        importances = self.model.feature_importances_
        param_importance = importances[:self.param_dim]

        imp_df = pd.DataFrame({
            'Feature': [f"Param_{i}" for i in range(self.param_dim)],
            'Importance': param_importance
        }).sort_values(by='Importance', ascending=False)

        return imp_df

    def generate_refined_configs(self, output_file="refined_configs.json", top_n_params=3):
        """
        核心功能：剔除差值，生成基于默认配置的优化向量列表
        """
        # 1. 获取最重要的 N 个参数
        imp_df = self.analyze_importance()
        top_features = imp_df.head(top_n_params)['Feature'].tolist()

        global_avg = self.raw_df['score'].mean()
        print(f"\n=== Generating Refined Configs (Global Avg: {global_avg:.4f}) ===")
        print(f"Base Configuration: {self.default_param}")

        # 用于存储最终生成的配置向量（列表的列表）
        final_vectors = []
        # 用于去重，防止生成重复的向量
        seen_vectors = set()

        # 这里的逻辑是：对于每个 Top 参数，生成一组配置
        # 这些配置是：Base Config + 该参数的一个“好”值

        for param_name in top_features:
            print(f"\nProcessing {param_name}...")
            param_idx = int(param_name.split('_')[1])  # 获取参数索引，如 "Param_1" -> 1

            # 计算该参数每个取值的平均分
            stats = self.raw_df.groupby(param_name)['score'].mean().reset_index()
            all_values = stats[param_name].tolist()

            # 找出低于全局平均分的值
            bad_candidates = stats[stats['score'] < global_avg].sort_values(by='score')

            # 策略：剔除均值低于 global 的最差 3 个
            # 如果低于 global 的不足 3 个，就剔除到2个
            drop_count = 0
            values_to_drop = bad_candidates.head(drop_count)[param_name].tolist()

            print(f"  -> Found {len(bad_candidates)} values < global avg.")
            if values_to_drop:
                print(f"  -> Dropping worst {len(values_to_drop)} values: {values_to_drop}")
                # 打印详细分数为证
                for v in values_to_drop:
                    s = stats[stats[param_name] == v]['score'].values[0]
                    print(f"     Val {v}: score {s:.4f}")
            else:
                print("  -> No values dropped (all above global avg).")

            # 保留剩下的“好”值
            good_values = [v for v in all_values if v not in values_to_drop]
            print(f"  -> Keeping values: {good_values}")

            # 生成新配置
            for val in good_values:
                # 复制一份默认配置
                new_vec = list(self.default_param)
                # 修改当前的重要参数
                new_vec[param_idx] = int(val)  # 确保是整数

                # 转 tuple 为了放入 set 去重
                vec_tuple = tuple(new_vec)
                if vec_tuple not in seen_vectors:
                    seen_vectors.add(vec_tuple)
                    final_vectors.append(new_vec)

        # 结果统计
        print(f"\nTotal unique configurations generated: {len(final_vectors)}")

        # 保存到 JSON
        with open(output_file, "w") as f:
            json.dump(final_vectors, f, indent=None)  # indent=None 使每个向量在一行，文件更紧凑

        print(f"Saved configurations to {output_file}")

        # 预览
        print("Preview:")
        for v in final_vectors[:]:
            print(v)


# ================= 运行脚本 =================

if __name__ == "__main__":
    CACHE_FILE = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log.cache4_norm.jsonl"
    LITMUS_VEC = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log"
    OUTPUT_JSON = "./refined_configs.json"

    analyzer = ParamAnalyzer(CACHE_FILE, LITMUS_VEC)
    analyzer.load_and_train()

    # 按照你的要求：针对 Top 3 参数，剔除最差的 3 个，生成基于默认配置的新向量
    analyzer.generate_refined_configs(output_file=OUTPUT_JSON, top_n_params=3)
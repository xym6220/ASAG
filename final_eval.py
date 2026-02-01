import numpy as np
import torch
import math
from transformers import AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score, accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr
import pandas as pd

# 引入你的自定义模块
from data_loader import ASAGDataLoader
from model_utils import DebertaForASAG

# --- 配置区 ---
# 替换为你最好的那个 checkpoint 路径
MODEL_PATH = "./results_ra_ocr/checkpoint-2075"
MAX_LENGTH = 512


def main():
    print(f">>> [Final Eval] Loading Model from: {MODEL_PATH}")

    # 1. 加载模型与分词器
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = DebertaForASAG.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. 加载 Mohler 测试集 (严格保持 8:2 切分)
    loader = ASAGDataLoader(tokenizer, max_length=MAX_LENGTH)
    ds_mohler = loader.load_mohler()
    mohler_split = ds_mohler.train_test_split(test_size=0.2, seed=42)
    test_dataset = mohler_split['test']

    print(f">>> Test Set Size: {len(test_dataset)} samples")

    # 3. 预测
    tokenized_test = test_dataset.map(
        loader.preprocess_function,
        batched=True,
        remove_columns=ds_mohler.column_names
    )

    # 预测配置 (FP16 加速)
    args = TrainingArguments(output_dir="./tmp_final_eval", per_device_eval_batch_size=16, fp16=True)
    trainer = Trainer(model=model, args=args)
    output = trainer.predict(tokenized_test)

    # 4. 数据后处理 (反归一化)
    # 模型输出是 0-1 (Sigmoid后的结果)
    preds_norm = output.predictions.squeeze()
    labels_norm = output.label_ids

    # 还原到 0-5 分真实尺度
    preds_real = preds_norm * 5.0
    labels_real = labels_norm * 5.0

    # 截断修正 (防止越界)
    preds_real = np.clip(preds_real, 0, 5)

    # 转为整数 (用于分类指标: QWK, Acc, F1)
    preds_int = np.round(preds_real).astype(int)
    labels_int = np.round(labels_real).astype(int)

    # ================= 计算核心指标 =================

    # 1. 回归指标 (Regression Metrics)
    mse = mean_squared_error(labels_real, preds_real)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(labels_real, preds_real)
    pearson_corr, _ = pearsonr(labels_real, preds_real)
    spearman_corr, _ = spearmanr(labels_real, preds_real)

    # 2. 分类/序数指标 (Classification/Ordinal Metrics)
    qwk = cohen_kappa_score(labels_int, preds_int, weights='quadratic')
    accuracy = accuracy_score(labels_int, preds_int)
    # Macro-F1: 宏平均，对不平衡类别更公平
    f1_macro = f1_score(labels_int, preds_int, average='macro')
    # Weighted-F1: 加权平均，通常分数会更高
    f1_weighted = f1_score(labels_int, preds_int, average='weighted')

    # ================= 输出最终报表 =================
    print("\n" + "=" * 50)
    print("   IJCNN ASAG FINAL EVALUATION REPORT (Mohler)   ")
    print("=" * 50)
    print(f"Dataset: Mohler (0-5 Scale)")
    print(f"Model:   DeBERTa-v3-Large + Transfer Learning")
    print("-" * 50)

    print(">>> [Regression Metrics] (衡量预测值准确度)")
    print(f"RMSE (Root Mean Sq Error) : {rmse:.4f}  [越低越好 | SOTA ~0.70]")
    print(f"MAE  (Mean Absolute Error): {mae:.4f}   [越低越好 | SOTA ~0.50]")
    print(f"Pearson Correlation (r)   : {pearson_corr:.4f}  [越高越好 | SOTA >0.80]")
    print(f"Spearman Correlation (ρ)  : {spearman_corr:.4f} [越高越好]")

    print("\n>>> [Classification Metrics] (衡量评级一致性)")
    print(f"QWK (Quadratic Kappa)     : {qwk:.4f}   [核心指标 | SOTA >0.75]")
    print(f"Accuracy (Exact Match)    : {accuracy:.4f}")
    print(f"Macro F1-Score            : {f1_macro:.4f}   [衡量类别平衡性]")
    print(f"Weighted F1-Score         : {f1_weighted:.4f}")
    print("=" * 50)

    # 保存结果到文件，方便写论文复制
    with open("final_metrics.txt", "w") as f:
        f.write(
            f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nPearson: {pearson_corr:.4f}\nQWK: {qwk:.4f}\nAccuracy: {accuracy:.4f}\nMacro-F1: {f1_macro:.4f}\n")
    print("Results saved to 'final_metrics.txt'.")


if __name__ == "__main__":
    main()
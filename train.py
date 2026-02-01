import os
import torch
import numpy as np
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import mean_squared_error, cohen_kappa_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from datasets import concatenate_datasets
# 导入自定义模块
from data_loader import ASAGDataLoader
from model_utils import DebertaForASAG

# --- 0. 配置 ---
MODEL_PATH = "/mnt/workspace/models/DeBerta-large-v3"  # 确保此路径下有模型文件
# OUTPUT_DIR = "./results_ra_ocr"
OUTPUT_DIR = "./results_seed_256"
MAX_LENGTH = 512
BATCH_SIZE = 4  # 严格限制：单卡 24GB
GRAD_ACCUM = 8  # 累积到 32
EPOCHS = 5
LR = 1e-5  # DeBERTa 敏感，需低学习率


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Sigmoid 输出已经在 0-1，直接使用
    preds = predictions.squeeze()

    # 指标 1: RMSE
    rmse = np.sqrt(mean_squared_error(labels, preds))

    # 指标 2: Pearson Correlation
    pearson = pearsonr(labels, preds)[0]

    # 指标 3: QWK (需要将回归值映射回整数类别 0-5 用于计算 Kappa)
    # 假设归一化是 /5.0，这里乘回去
    preds_int = np.round(preds * 5.0).astype(int)
    labels_int = np.round(labels * 5.0).astype(int)
    qwk = cohen_kappa_score(labels_int, preds_int, weights='quadratic')

    return {"rmse": rmse, "pearson": pearson, "qwk": qwk}


def main():
    print(">>> 初始化 ASAG 专家系统...")

    # 1. 准备 Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    except:
        print(f"本地未找到 {MODEL_PATH}，尝试从 HF 下载...")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

    # 2. 准备数据
    loader = ASAGDataLoader(tokenizer, max_length=MAX_LENGTH)

    # 加载两个数据集
    print(">>> 正在加载数据集...")
    ds_scients = loader.load_scientsbank()  # 约 5000 条
    ds_mohler_full = loader.load_mohler()  # 约 2200 条

    # 【关键步骤】对 Mohler 进行切分 (Train 80% / Test 20%)
    # seed=42 保证每次切分结果一致，这是论文复现的关键
    mohler_split = ds_mohler_full.train_test_split(test_size=0.2, seed=256)

    ds_mohler_train = mohler_split['train']
    ds_mohler_test = mohler_split['test']

    print(f"Mohler 划分: 训练集 {len(ds_mohler_train)} 条, 测试集 {len(ds_mohler_test)} 条")

    # 【新增步骤】加载刚才生成的硬负样本
    ds_augmented = loader.load_augmented_data()

    # 【SOTA 混合策略】
    # 1. SciEntsBank (通用基础)
    # 2. Mohler Train x 3 (目标域加权，确保学会原本的评分标准)
    # 3. Augmented Data x 2 (负样本加权，让模型重点关注这些陷阱)

    # 先放入基础数据
    datasets_to_concat = [ds_scients, ds_mohler_train, ds_mohler_train, ds_mohler_train]

    # 如果找到了增强数据，就加进去
    if ds_augmented is not None:
        print(f">>> 成功混入 {len(ds_augmented)} 条硬负样本！正在进行加权混合...")
        # 混入 2 份增强数据
        datasets_to_concat.append(ds_augmented)
        datasets_to_concat.append(ds_augmented)
    else:
        print(">>> 警告：未找到增强数据，将进行普通训练。")

    # 执行合并
    train_dataset = concatenate_datasets(datasets_to_concat)

    # 验证集 = 仅使用 Mohler 剩下的 20% (从未见过的数据)
    eval_dataset = ds_mohler_test

    # 防御性检查
    loader.verify_data(train_dataset, "Mixed Train Set")
    loader.verify_data(eval_dataset, "Hold-out Test Set")

    print(">>> 正在进行 Tokenization 并清理数据列...")
    # 注意：这里需要分别处理，并且移除列
    tokenized_train = train_dataset.map(
        loader.preprocess_function,
        batched=True,
        remove_columns=ds_scients.column_names  # 移除原始文本列
    )
    tokenized_eval = eval_dataset.map(
        loader.preprocess_function,
        batched=True,
        remove_columns=ds_mohler_full.column_names  # 移除原始文本列
    )

    # 再次检查：确保只剩下 Tensor 友好的列
    print(f"训练集保留列: {tokenized_train.column_names}")
    # 预期输出: ['labels', 'input_ids', 'token_type_ids', 'attention_mask']

    # 3. 初始化模型
    model = DebertaForASAG.from_pretrained(MODEL_PATH)

    # 显存优化关键：开启 Gradient Checkpointing
    model.gradient_checkpointing_enable()

    # 4. 训练参数配置
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        fp16=True,  # 必须开启混合精度
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="qwk",
        save_total_limit=2,
        report_to="none",  # 暂时关闭 wandb，保证纯净
        dataloader_num_workers=0,  # 设置为 0 以避免多进程 DataLoader 带来的额外开销或报错
        remove_unused_columns=False  # 自定义模型必须设为 False，否则 labels 可能被自动移除
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # 5. 开始训练
    print(">>> 开始训练 (RA-OCR)...")
    trainer.train()

    # 6. 最终评估与可视化
    print(">>> 正在生成论文图表...")
    # 预测时同样需要 Collator，使用 Trainer 的 predict 方法最稳妥
    results = trainer.predict(tokenized_eval)

    # --- 自动生成 t-SNE 图 ---
    # 获取 Contrastive Head 的投影特征
    model.eval()

    # 随机采样 500 个样本做可视化，防止内存溢出
    sample_size = min(500, len(tokenized_eval))
    subset_indices = np.random.choice(len(tokenized_eval), sample_size, replace=False)
    subset = tokenized_eval.select(subset_indices)

    features_list = []
    labels_list = []

    print(">>> 正在提取特征用于 t-SNE 可视化...")
    with torch.no_grad():
        for i in range(0, len(subset), BATCH_SIZE * 2):
            batch = subset[i: i + BATCH_SIZE * 2]
            # 手动 Collate
            batch_features = data_collator(batch)
            # 移至 GPU
            inputs = {k: v.to(model.device) for k, v in batch_features.items() if
                      k in ['input_ids', 'attention_mask', 'token_type_ids']}

            # 调用底层 DeBERTa 获取 hidden state
            outputs = model.deberta(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu()
            features_list.append(cls_emb)

            # 获取对应的 labels
            if 'labels' in batch_features:
                labels_list.extend(batch_features['labels'].cpu().numpy())
            else:
                # Fallback provided inputs
                labels_list.extend([x['labels'] for x in batch])

    features = torch.cat(features_list).numpy()
    labels = np.array(labels_list)

    # t-SNE
    print(">>> 计算 t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, sample_size - 1))
    vis_dims = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(vis_dims[:, 0], vis_dims[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Grade (0-1)')
    plt.title('t-SNE visualization of Answer Embeddings (RA-OCR)')
    tsne_path = os.path.join(OUTPUT_DIR, 'tsne_plot.png')
    plt.savefig(tsne_path)
    print(f">>> t-SNE 图已保存至 {tsne_path}")


if __name__ == "__main__":
    main()
import torch

class Config:
    # === 基础配置 ===
    model_name = "/home/ai/QA/ASAG_Project/models/DeBert-large-V3"
    data_dir = "./data/"
    output_dir = "./outputs/"
    seed = 2026
    max_length = 512

    # === 训练参数 (针对双卡 5090 优化) ===
    epochs = 4             # 减少 Epoch，防止过拟合
    train_batch_size = 4   # 开启 Gradient Checkpointing 后可翻倍 (实际单卡处理 4*4=16 样本)
    eval_batch_size = 32   # 验证集不产生梯度，可拉大
    gradient_accumulation_steps = 4 # 等效 Batch Size = 4 * 4 * 2(卡) = 32

    # === 优化器与学习率 (关键！) ===
    learning_rate = 1e-5   # 顶层学习率稍大，底层通过 LLRD 衰减
    weight_decay = 0.05    # 强正则化
    warmup_ratio = 0.1
    layer_decay = 0.85     # 【新增】LLRD 衰减系数，越底层 LR 越小

    # === 究极 Loss 权重 (调整平衡) ===
    # 调高 Rank 权重，强迫模型学习“逻辑排序”而非“死记硬背”
    lambda_rank = 1.0
    lambda_cons = 0.5
    margin = 0.5

    # === 硬件加速 ===
    use_fp16 = False
    use_bf16 = True        # 5090 必须用 BF16
    num_workers = 8
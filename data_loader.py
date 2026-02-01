import os
import json
import logging
from datasets import load_from_disk, Dataset  # 引入 Dataset 用于构建新数据
import torch

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 硬编码路径配置 ---
MOHLER_PATH = "/mnt/workspace/offline_datasets/mohler"
SCIENTS_PATH = "/mnt/workspace/offline_datasets/scientsbank"
# 新增：增强数据路径 (请确保文件名一致)
AUGMENTED_PATH = "offline_datasets/mohler_hard_negatives_test.jsonl"


class ASAGDataLoader:
    def __init__(self, model_tokenizer, max_length=512):
        self.tokenizer = model_tokenizer
        self.max_length = max_length
        # SciEntsBank 标签映射
        self.label_map = {
            'correct': 1.0,
            'partially_correct_incomplete': 0.5,
            'contradictory': 0.0,
            'incorrect': 0.0,
            'irrelevant': 0.0,
            'non_domain': 0.0
        }

    def load_mohler(self):
        """从本地加载 Mohler 数据集"""
        logger.info(f"Loading Mohler dataset from disk: {MOHLER_PATH}")
        try:
            ds = load_from_disk(MOHLER_PATH)

            def normalize_mohler(examples):
                return {
                    "question": examples["question"],
                    "student_answer": examples["student_answer"],
                    "reference_answer": examples["instructor_answer"],
                    "label": float(examples["score_avg"]) / 5.0  # 归一化
                }

            full_ds = ds['open_ended']
            processed_ds = full_ds.map(normalize_mohler, remove_columns=full_ds.column_names)
            return processed_ds
        except Exception as e:
            logger.error(f"Error loading Mohler: {str(e)}")
            raise

    def load_scientsbank(self):
        """从本地加载 SciEntsBank 数据集"""
        logger.info(f"Loading SciEntsBank dataset from disk: {SCIENTS_PATH}")
        try:
            ds = load_from_disk(SCIENTS_PATH)

            def process_scients(examples):
                raw_lbl = examples["label"]
                label_names = ds['train'].features['label'].names
                str_lbl = label_names[raw_lbl]
                score = self.label_map.get(str_lbl, 0.0)

                return {
                    "question": examples["question"],
                    "student_answer": examples["student_answer"],
                    "reference_answer": examples["reference_answer"],
                    "label": score
                }

            processed_ds = ds['train'].map(process_scients, remove_columns=ds['train'].column_names)
            return processed_ds
        except Exception as e:
            logger.error(f"Error loading SciEntsBank: {str(e)}")
            raise

    def load_augmented_data(self):
        """【新增】加载 Qwen 生成的硬负样本"""
        logger.info(f"Loading Augmented Hard Negatives from: {AUGMENTED_PATH}")
        data_list = []
        try:
            if not os.path.exists(AUGMENTED_PATH):
                logger.warning(f"Warning: Augmented file {AUGMENTED_PATH} not found! Returning empty dataset.")
                return None

            with open(AUGMENTED_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        data_list.append({
                            "question": item['question'],
                            "student_answer": item['student_answer'],
                            "reference_answer": item['reference_answer'],
                            "label": float(item['label'])  # 确保是 float 0.0
                        })

            if not data_list:
                return None

            # 从 list 构建 HuggingFace Dataset
            aug_ds = Dataset.from_list(data_list)
            logger.info(f"Successfully loaded {len(aug_ds)} hard negative samples.")
            return aug_ds

        except Exception as e:
            logger.error(f"Error loading Augmented Data: {str(e)}")
            return None

    def verify_data(self, dataset, name="Dataset"):
        """防御性检查"""
        logger.info(f"--- Verifying {name} ---")
        if dataset is None or len(dataset) == 0:
            raise ValueError(f"{name} is empty or None!")
        sample = dataset[0]
        logger.info(f"Sample keys: {sample.keys()}")
        logger.info(f"Label: {sample['label']}")
        logger.info("--- Verification Passed ---")

    def preprocess_function(self, examples):
        input_texts_a = [f"{q} {r}" for q, r in zip(examples["question"], examples["reference_answer"])]
        input_texts_b = examples["student_answer"]

        model_inputs = self.tokenizer(
            input_texts_a,
            input_texts_b,
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )
        model_inputs["labels"] = examples["label"]
        return model_inputs
import os
import time
import json
import random
from datasets import load_from_disk
import dashscope
from dashscope import Generation

# ================= 配置区 =================
# 【请务必替换为你的阿里云 API Key】
dashscope.api_key = "sk-35102c745925493bbcfcc69d6d3a5640"

# 数据路径 (Mohler)
INPUT_PATH = "./offline_datasets/mohler"
OUTPUT_FILE = "mohler_hard_negatives_test.jsonl"

# 【测试开关】设置为 True 只跑 5 条；设置为 False 跑全量
TEST_MODE = False
TEST_COUNT = 5


# =========================================

def generate_hard_negative_qwen(question, ref_answer, student_answer):
    """
    核心逻辑：利用大模型生成"似是而非"的 0 分答案。
    它看起来像那个 5 分的学生写的，但是逻辑是反的。
    """
    prompt = f"""
        你是一位故意要设计“陷阱题”的计算机科学教授。

        【背景信息】
        题目: {question}
        正确标准答案: {ref_answer}
        学生原始高分答案(5分): {student_answer}

        【任务】
        请篡改上述“学生原始高分答案”，生成一个看起来非常专业、使用了正确的术语，但在核心逻辑上是**完全错误**（必须判为0分）的答案。

        【关键修改规则】
        1. **长难句处理**：如果是长句子，保留专业术语，但颠倒因果关系（如把“因为A所以B”改成“因为B所以A”），或修改核心定义。
        2. **短语/列表处理（高危预警）**：如果原答案只是几个词的罗列（例如“Coding and Testing”），**绝对不能**保留其中的正确项！必须将其替换为错误的阶段。
           - 错误示范：原答案是 "Coding"，你生成 "Coding and design" (不行，因为包含了正确答案)。
           - 正确示范：原答案是 "Coding"，你生成 "Deployment and Maintenance" (完全避开正确阶段，但依然是软件工程术语)。
        3. **高迷惑性**：不要说胡话，要用看似合理的术语去“一本正经地胡说”。

        【输出要求】
        仅输出修改后的**答案文本**，不要包含“修改后：”等前缀，不要包含解释。
        """

    try:
        response = Generation.call(
            model='qwen-plus',  # 建议用 Plus 或 Max，逻辑控制能力比 Turbo 强
            prompt=prompt,
            result_format='message',
            temperature=0.8  # 稍微高一点，增加多样性
        )
        if response.status_code == 200:
            return response.output.choices[0].message.content.strip()
        else:
            print(f"API Error Code: {response.code}")
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None


def main():
    print(f">>> 启动增强脚本 (测试模式: {TEST_MODE})...")

    # 1. 加载 Mohler
    try:
        ds = load_from_disk(INPUT_PATH)
        dataset = ds['open_ended']  # Mohler 的核心数据
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return

    # 2. 筛选高分样本 (Score > 4.5 的才值得做负样本)
    # 注意：Mohler raw 数据里 score_avg 是 float 类型
    high_score_data = [
        ex for ex in dataset
        if float(ex.get('score_avg', 0)) >= 4.5
    ]

    print(f"原始数据集大小: {len(dataset)}")
    print(f"筛选出高分样本(>=4.5): {len(high_score_data)} 条")

    # 3. 确定要跑多少条
    target_data = high_score_data[:TEST_COUNT] if TEST_MODE else high_score_data
    print(f"本次计划生成: {len(target_data)} 条")

    # 4. 循环生成
    success_count = 0
    # 如果是测试模式，写入新文件；如果是全量，建议追加(mode='a')
    file_mode = 'w' if TEST_MODE else 'a'

    with open(OUTPUT_FILE, file_mode, encoding='utf-8') as f:
        for i, item in enumerate(target_data):
            print(f"[{i + 1}/{len(target_data)}] 正在生成 ID: {item['id']} ...")

            fake_answer = generate_hard_negative_qwen(
                item['question'],
                item['instructor_answer'],
                item['student_answer']
            )

            if fake_answer:
                # 构造新数据
                # 关键：保留原始 question 和 ref，但 student_answer 换成假的，label 设为 0.0
                new_record = {
                    "id": f"{item['id']}_hard_neg",
                    "question": item['question'],
                    "reference_answer": item['instructor_answer'],  # 统一字段名
                    "student_answer": fake_answer,  # 篡改后的答案
                    "label": 0.0,  # 强制 0 分
                    "is_augmented": True  # 标记位
                }

                f.write(json.dumps(new_record) + "\n")
                success_count += 1

                # 打印对比方便你 Check (只在测试模式打印)
                if TEST_MODE:
                    print(f"\n--- 对比 (ID: {item['id']}) ---")
                    print(f"[Q]: {item['question']}")
                    print(f"[原高分]: {item['student_answer']}")
                    print(f"[新零分]: {fake_answer}")
                    print("-" * 50 + "\n")

            # 限流保护
            time.sleep(0.5)

    print(f"\n>>> 完成！成功生成 {success_count} 条。")
    print(f"结果已保存至: {OUTPUT_FILE}")
    print("请查看生成的 JSONL 文件，确认逻辑是否符合预期（词汇相似但逻辑错误）。")


if __name__ == "__main__":
    main()
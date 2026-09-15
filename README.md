# Agentic RL 最小实验台（可验证奖励 + 拒绝采样 + DPO）

用**强化学习的思路**提升一个 Agent 的任务成功率：不靠人工标注，而是**让模型自己采样、用规则判对错、再拿成功的经验去学**。

**离线/在线都不需要外部服务**：奖励来自**可验证信号**（工具名对不对、参数对不对、答案有没有命中真实实体），不用模型当裁判——从源头避开 reward hacking。

## 一、为什么奖励要"可验证"

Agentic RL 最怕模型学会**讨好裁判**而不是解决问题。所以本实验的奖励全部由代码判定：

| 维度 | 权重 | 怎么判 |
|---|---|---|
| 工具名正确 | 0.4 | 与标准工具名比对 |
| 参数正确 | 0.3 | 关键参数（如 `order_id`）逐一比对 |
| 最终答案命中实体 | 0.3 | 答案里是否包含订单号 / 状态等真实实体 |

满分 = 1.0，视为任务成功。

## 二、任务形态（2 步，真的"agentic"）

```
第1步：模型看到「用户问题 + 工具目录」→ 必须输出 <tool_call>{"name":..., "args":{...}}</tool_call>
       环境执行工具 → 返回结构化结果
第2步：模型看到工具结果 → 输出 <answer>...</answer>（需如实转述结果）
```
覆盖查物流 / 查订单 / 退款咨询 / 申请退款 / 政策问答 5 类意图。

## 三、AutoDL 上跑一遍（约 2–3 小时）

### 0. 准备
- GPU：RTX 4090（24G）
- 镜像：PyTorch 2.x + CUDA 12.x
- 基座模型：复用 `lora-finetune-lab` 已下载的 Qwen2.5-1.5B（在数据盘里，不用重下）

### 1. 装依赖
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 若报 torchvision::nms 不存在：pip uninstall -y torchvision
```

### 2. 生成任务集
```bash
python tasks.py                 # data/tasks_train.jsonl(160) + tasks_test.jsonl(40)
```

### 3. 采样轨迹（RL 的"探索"环节）
```bash
python rollout.py \
  --model /root/autodl-tmp/models/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master \
  --n 4                        # 每题采 4 条，温度 0.8 保证多样性
```
> 输出 `data/rollouts.jsonl`：每条含 工具调用 / 环境结果 / 最终答案 / **规则得分**。
> 关键看**成功率**——基座通常不高，这正是 RL 要提升的空间。

### 4. 加工成训练数据
```bash
python build_data.py
```
- `data/sft.jsonl` —— **只留满分轨迹**（拒绝采样）
- `data/dpo_pairs.jsonl` —— **同题最优 vs 最差** 自动配对（不需要人工标注）

### 5. 训练
```bash
# L1：拒绝采样 + 指令微调（QLoRA）
python train_rft.py --model <基座路径>

# L2：DPO 偏好优化
python train_dpo.py --model <基座路径>
```

### 6. 对比评测（关键一步）
```bash
python evaluate.py --base <基座路径>
```
输出：**基座 / RFT / DPO 三档的成功率、平均奖励、工具名正确率**，以及样例对比。

## 四、目录结构

```
env.py          工具环境 + 可验证奖励（工具名 0.4 / 参数 0.3 / 答案实体 0.3）
tasks.py        任务集生成（160 训练 / 40 测试）
rollout.py      轨迹采样（温度采样 → 规则打分）
build_data.py   轨迹 → RFT 数据 / DPO 偏好对
train_rft.py    L1：在成功轨迹上做 QLoRA 指令微调
train_dpo.py    L2：DPO 偏好优化（trl）
evaluate.py     基座 vs RFT vs DPO
data/  reports/ 产出
```

## 五、方法说明（面试可讲）

| 环节 | 做法 | 为什么 |
|---|---|---|
| 奖励 | **规则可验证**，不用模型打分 | 避免 reward hacking；信号干净 |
| L1 | **拒绝采样 + SFT** | 最简单的 agentic RL：不写 RL 算法，只用"成功的再学一遍"提升成功率 |
| L2 | **DPO** | 偏好对由采样**自动生成**；相比 PPO 不需要 value 网络和在线采样，小规模性价比高 |
| 评测 | 贪心解码、同一测试集、只换模型 | 排除采样随机性，结论可信 |

## 六、可继续做的
- 换更大基座（7B）看 RL 收益变化
- 引入 **GRPO**：同题采样一组、组内比相对优势（需要更大的采样预算）
- 用 LLM-as-a-Judge 处理开放式任务（配套项目 `llm-judge-eval`）

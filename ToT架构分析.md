# Tree of Thoughts (ToT) 思维树 Agent 搭建指南

> 从零理解如何把大语言模型当作搜索算法来用 —— 在 24 点游戏上实现思维树搜索。

---

## 一、什么是 Tree of Thoughts？

### 1.1 核心思想

传统 LLM 推理是"一条路走到黑"（Chain-of-Thought），但复杂问题有多个分支，一步选错就全盘皆输。

**ToT 的核心改动**：在每一步，LLM 同时生成 N 个候选下一步（Generate），然后独立评估每个候选的优劣（Evaluate），只保留最有希望的几个继续往下走（Select）。

```
问题: 4 5 6 10 → 凑 24

Step 0:  [""]                                        ← 初始空状态
           │
Step 1:  生成 19 个候选 → 评估 → 选 1 个最好的
           │
         4*5=20 (left: 6 10 20)   [得分 20, 选中]
           │
Step 2:  生成 10 个候选 → 评估 → 选 1 个最好的
           │
         10-6=4 (left: 4 20)      [得分 20, 选中]
           │
Step 3:  生成 4 个候选 → 评估 → 选 1 个最好的
           │
         4+20=24 (left: 24)       [得分 20, 选中]

答案: (4*5) + (10-6) = 24  ✓
```

### 1.2 和普通 LLM 推理的区别

| | Chain-of-Thought | Tree of Thoughts |
|---|---|---|
| **思路** | 一条链走到底 | 多分支探索 |
| **纠错** | 错了就错了 | 错了有备选分支 |
| **LLM 调用** | 1 次 | 每步 N 次（生成）+ M 次（评估）|
| **适合** | 简单推理 | 需要搜索/试错的问题 |
| **本质** | 贪心算法 | 束搜索（Beam Search） |

---

## 二、四层架构总览

整个 ToT 系统由四层组成，从底层到上层分别是：

```
┌─────────────────────────────────────────────────────────┐
│                      run.py  入口                       │
│  解析参数 → 加载任务 → 循环solve → 输出日志/准确率      │
├─────────────────────────────────────────────────────────┤
│                  methods/bfs.py  搜索算法                │
│  solve():  for step in steps:                          │
│    Generate (生成候选) → Evaluate (打分) → Select (剪枝)  │
├─────────────────────────────────────────────────────────┤
│              tasks/game24.py  任务适配                   │
│  get_input() / test_output() / propose_prompt_wrap()    │
│  把具体任务包装成 ToT 算法能调用的接口                    │
├─────────────────────────────────────────────────────────┤
│  models.py  LLM调用  │  prompts/game24.py  Prompt模板    │
│  gpt() → 豆包API     │  few-shot示例，告诉LLM怎么思考    │
└─────────────────────────────────────────────────────────┘
```

**依赖方向**：入口 → 算法 → 任务 → (模型 + Prompt)

---

## 三、第一层：models.py — LLM 调用封装

### 3.1 目标

把豆包 API 封装成一个干净的函数 `gpt(prompt) → [response1, response2, ...]`，上层代码不关心认证、重试、token 计费。

### 3.2 代码（完整）

```python
import backoff
from openai import OpenAI

# ===== 豆包 API 配置 =====
API_KEY = "your-api-key"            # 替换为你的 API Key
BASE_URL = "https://api.openai.com/v1"                 # 或其他兼容端点
MODEL = "gpt-4o"                                       # 模型名或端点ID

client = OpenAI(api_key=DOUBAO_API_KEY, base_url=DOUBAO_BASE_URL)

completion_tokens = 0   # 累计输出token
prompt_tokens = 0       # 累计输入token


@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def _completions_with_backoff(**kwargs):
    """指数退避重试：第1次等1秒，第2次等2秒，第3次等4秒...最多5次"""
    return client.chat.completions.create(**kwargs)


def gpt(prompt, model=DOUBAO_MODEL, temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    """发送一条 user prompt，返回 n 个回复的列表"""
    messages = [{"role": "user", "content": prompt}]
    return _chatgpt(messages, model, temperature, max_tokens, n, stop)


def _chatgpt(messages, model, temperature, max_tokens, n, stop) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    for _ in range(n):            # 豆包不支持 n>1，所以逐条请求
        res = _completions_with_backoff(
            model=model, messages=messages, temperature=temperature,
            max_tokens=max_tokens, stop=stop
        )
        outputs.extend([choice.message.content for choice in res.choices])
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
    return outputs
```

### 3.3 关键设计

| 设计点 | 代码 | 原因 |
|--------|------|------|
| **指数退避** | `@backoff.on_exception(backoff.expo, ...)` | API 偶发 429/500 错误，自动重试不崩 |
| **n 拆分为独立请求** | `for _ in range(n):` 逐个发请求 | 豆包端点不支持 `n>1` 参数 |
| **全局 token 计数** | `global completion_tokens, prompt_tokens` | 跨所有调用累计，最后算费用 |
| **stop 参数透传** | `stop=stop` 原样传 | 让 LLM 在特定位置停止（如换行） |

---

## 四、第二层：prompts/game24.py — Few-shot Prompt 模板

### 4.1 目标

ToT 需要 LLM 做三件事，每件事都需要不同的 few-shot 示例来"教"LLM 怎么回答：

| Prompt | 作用 | LLM 要输出什么 |
|--------|------|---------------|
| `standard_prompt` | 直接给答案 | `Answer: (a+b)*(c-d) = 24` |
| `cot_prompt` | 分步推理 | 每步的运算 + 最终答案 |
| `propose_prompt` | 生成候选下一步 | 从当前数字出发的所有可行运算 |
| `value_prompt` | 评估能否到 24 | `sure` / `likely` / `impossible` |

### 4.2 standard_prompt（5-shot 直接回答）

```python
standard_prompt = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Input: 1 4 8 8
Answer: (8 / 4 + 1) * 8 = 24
Input: 5 5 5 9
Answer: 5 + 5 + 5 + 9 = 24
Input: {input}
'''
```

### 4.3 propose_prompt（1-shot 生成候选）

```python
propose_prompt = '''Input: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 /  2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)
Input: {input}
Possible next steps:
'''
```

**关键**：给 LLM 展示"从三个数字里选两个做运算，得到一个新数字"的模式。LLM 会模仿这个格式输出所有可行运算。

### 4.4 value_prompt（7-shot 评估）

```python
value_prompt = '''Evaluate if given numbers can reach 24 (sure/likely/impossible)
10 14
10 + 14 = 24
sure
11 12
11 + 12 = 23
12 - 11 = 1
11 * 12 = 132
11 / 12 = 0.91
impossible
4 4 10
4 + 4 + 10 = 8 + 10 = 18
4 * 10 - 4 = 40 - 4 = 36
(10 - 4) * 4 = 6 * 4 = 24
sure
...
{input}
'''
```

**评分映射**（在 game24.py 中定义）：

```python
value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}
```

`sure` 得分是 `likely` 的 20 倍，`impossible` 接近 0。这样 greedy 选择会优先选"确定能到 24"的分支。

---

## 五、第三层：tasks/game24.py — 任务适配层

### 5.1 目标

实现 `Task` 基类定义的接口，让上层算法不关心具体是什么任务。

### 5.2 基类（base.py）

```python
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')

class Task:
    """所有任务的抽象基类。"""

    def __init__(self):
        self.value_cache = {}     # value prompt → score 的缓存，避免重复API调用
        self.steps = 4            # 24点最多4步
        self.stops = ['\n'] * 4   # 每步的停止符

    def __len__(self) -> int:
        raise NotImplementedError

    def get_input(self, idx: int) -> str:
        raise NotImplementedError

    def test_output(self, idx: int, output: str):
        raise NotImplementedError
```

### 5.3 Game24Task 核心方法

```python
import re
import sympy
import pandas as pd

class Game24Task(Task):
    """24点游戏 - ToT经典测试任务"""

    def __init__(self, file='24.csv'):
        super().__init__()
        path = os.path.join(DATA_PATH, '24', file)
        self.data = list(pd.read_csv(path)['Puzzles'])  # 1362 道题

    def get_input(self, idx: int) -> str:
        return self.data[idx]  # 如 "4 5 6 10"

    def test_output(self, idx: int, output: str):
        """验证最终答案是否正确"""
        # 1. 提取表达式（最后一行 = 号左边）
        expression = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]

        # 2. 校验数字是否一致（不能多也不能少）
        numbers = re.findall(r'\d+', expression)
        problem_numbers = re.findall(r'\d+', self.data[idx])
        if sorted(numbers) != sorted(problem_numbers):
            return {'r': 0}

        # 3. 用 sympy 验证表达式是否等于 24
        try:
            return {'r': int(sympy.simplify(expression) == 24)}
        except Exception:
            return {'r': 0}
```

### 5.4 Prompt 拼接方法

```python
@staticmethod
def propose_prompt_wrap(x: str, y: str = '') -> str:
    """y 是当前轨迹，从轨迹中提取剩余数字，拼入 propose prompt"""
    current_numbers = get_current_numbers(y if y else x)
    # y = "4*5=20 (left: 6 10 20)\n" → current_numbers = "6 10 20"
    if current_numbers == '24':
        prompt = cot_prompt.format(input=x) + 'Steps:' + y  # 已到24，收尾
    else:
        prompt = propose_prompt.format(input=current_numbers)
    return prompt
```

**`get_current_numbers()` 是关键辅助函数**：

```python
def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]
```

输入 `"4*5=20 (left: 6 10 20)\n"` → 切最后一行 → 切 `left: ` 右边 → 切 `)` 左边 → 得到 `"6 10 20"`。

### 5.5 接口全景图

| 方法 | 谁调用 | 做什么 |
|------|--------|--------|
| `get_input(idx)` | `run.py` | 取出第 idx 道题 |
| `test_output(idx, output)` | `run.py` | 验证答案对不对 |
| `propose_prompt_wrap(x, y)` | `bfs.py → get_proposals()` | 拼"生成候选下一步"的 prompt |
| `value_prompt_wrap(x, y)` | `bfs.py → get_values()` | 拼"评估可行性"的 prompt |
| `value_outputs_unwrap(x, y, outputs)` | `bfs.py → get_values()` | 把 LLM 回复转成分数 |
| `standard_prompt_wrap(x, y)` | `bfs.py → get_samples()` | 拼"直接回答"的 prompt |
| `cot_prompt_wrap(x, y)` | `bfs.py → get_samples()` / propose | 拼"分步推理"的 prompt |

---

## 六、第四层：methods/bfs.py — 搜索算法（核心）

### 6.1 目标

实现 ToT 的核心循环，每一轮做三件事：**生成 → 评估 → 选择**。

### 6.2 solve() 主函数

```python
def solve(args, task, idx, to_print=True):
    model = args.backend          # 豆包端点
    temperature = args.temperature
    x = task.get_input(idx)       # 题目，如 "4 5 6 10"
    ys = ['']                     # 当前候选解列表，初始为空
    infos = []

    for step in range(task.steps):  # 24点最多4步

        # ===== 1. GENERATE 生成候选 =====
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, ...) for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y, ...) for y in ys]
        new_ys = list(itertools.chain(*new_ys))  # 展开二维列表

        # ===== 2. EVALUATE 评估打分 =====
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, ...)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, ...)

        # ===== 3. SELECT 剪枝保留 =====
        select_new_ys = select(new_ys, values, args.method_select, args.n_select_sample)

        ys = select_new_ys  # 下一轮的起点

    return ys, {'steps': infos}
```

### 6.3 三个子函数的实现

#### 6.3.1 get_proposals() — 生成候选下一步

```python
def get_proposals(task, x, y, model, temperature):
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = gpt(propose_prompt, model=model, temperature=temperature, n=1)[0].split('\n')
    return [y + _ + '\n' for _ in proposals]
```

`y` 是之前的轨迹，如 `"4*5=20 (left: 6 10 20)\n"`。LLM 返回多行候选，每行是一个新运算。每个候选前面拼接之前的轨迹，形成完整路径。

#### 6.3.2 get_values() — 评估打分

```python
def get_values(task, x, ys, n_evaluate_sample, cache_value, model, temperature):
    values = []
    seen = {}
    for y in ys:
        if y in seen:            # 重复候选直接给0
            values.append(0)
            continue
        value_prompt = task.value_prompt_wrap(x, y)
        if cache_value and value_prompt in task.value_cache:
            value = task.value_cache[value_prompt]   # 命中缓存
        else:
            value_outputs = gpt(value_prompt, ...)
            value = task.value_outputs_unwrap(x, y, value_outputs)
            if cache_value:
                task.value_cache[value_prompt] = value
        seen[y] = value
        values.append(value)
    return values
```

**缓存机制**：同一个 prompt 不重复调 API。同一个问题同一道题跑多轮时，value_cache 跨轮复用。

#### 6.3.3 select() — 选择最优候选

```python
def select(ys, values, method, n_select_sample):
    ids = list(range(len(ys)))
    if method == 'sample':
        # 按分数比例随机采样（分数高的被选中的概率大）
        ps = np.array(values) / sum(values)
        select_ids = np.random.choice(ids, size=n, p=ps).tolist()
    elif method == 'greedy':
        # 直接取分数最高的 n 个
        select_ids = sorted(ids, key=lambda i: values[i], reverse=True)[:n]
    return [ys[i] for i in select_ids]
```

### 6.4 算法可视化

```
Step 1:
  [""]  ← ys
    │
    ├── generate: LLM 看了 "4 5 6 10" 生成 19 个候选
    │    ["4+5=9 (left: 6 9 10)", "4*5=20 (left: 6 10 20)", ...]
    │
    ├── evaluate: LLM 对每个候选打分
    │    [1, 20, 1, 0.001, 1, ...]
    │
    └── select (greedy, n=1): 取了 "4*5=20 (left: 6 10 20)" [得分20]

Step 2:
  ["4*5=20 (left: 6 10 20)"]  ← ys
    │
    ├── generate: LLM 看了 "6 10 20" 生成 10 个候选
    │    ["6+10=16 (left: 16 20)", "10-6=4 (left: 4 20)", ...]
    │
    ├── evaluate: LLM 打分
    │    [1, 20, 1, ...]
    │
    └── select: 取了 "10-6=4 (left: 4 20)"

...重复直到得到 24 或步数用尽
```

---

## 七、入口：run.py

### 7.1 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--task` | str | **必填** | 任务名，目前只支持 `game24` |
| `--task_start_index` | int | 900 | 起始题号 |
| `--task_end_index` | int | 1000 | 结束题号（不包含） |
| `--naive_run` | flag | False | 是否用朴素模式（单次调用，不搜索） |
| `--prompt_sample` | str | None | naive 模式用 standard/cot |
| `--method_generate` | str | propose | 生成方式：sample / propose |
| `--method_evaluate` | str | value | 评估方式：value / vote |
| `--method_select` | str | greedy | 选择方式：greedy / sample |
| `--n_generate_sample` | int | 1 | 每个候选生成几个样本 |
| `--n_evaluate_sample` | int | 1 | 每个候选评估几次 |
| `--n_select_sample` | int | 1 | 每步保留几个候选 |

### 7.2 运行命令

```bash
# 朴素模式（每题1次LLM调用，快但不准）
python src/run.py --task game24 --task_start_index 900 --task_end_index 902 --naive_run --prompt_sample standard

# 完整 ToT（propose + value + greedy，较慢但更准）
python src/run.py --task game24 --task_start_index 900 --task_end_index 902 --method_generate propose --method_evaluate value --method_select greedy
```

### 7.3 预期输出

```
Namespace(backend='ep-...', task='game24', task_start_index=900, task_end_index=902, ...)
functools.partial(<function gpt at ...>, model='ep-...', temperature=0.7)
-- new_ys --: ('4*5=20 (left: 6 10 20)\n', '4+5=9 (left: 6 9 10)\n', ...)
-- sol values --: (20, 1, 1, ...)
-- choices --: ['4*5=20 (left: 6 10 20)\n']
...
900 sum(accs) 1 cnt_avg 1.0 cnt_any 1
平均准确率: 1.000, 任意解准确率: 1.000
usage_so_far {'completion_tokens': ..., 'prompt_tokens': ..., 'cost': ...}
```

---

## 八、完整数据流追踪

以题目 `"4 5 6 10"` 为例：

```
run.py: task = get_task('game24')  →  Game24Task实例，加载1362道题
        solve(args, task, idx=900)
          │
bfs.py:  x = task.get_input(900)   →  "4 5 6 10"
         ys = [""]
         │
         Step 1:
         │  get_proposals(task, x="4 5 6 10", y="")
         │    │
         │    ├─ task.propose_prompt_wrap("4 5 6 10", "")
         │    │    └─ 当前数字 = "4 5 6 10"
         │    │    └─ 返回: propose_prompt.format(input="4 5 6 10")
         │    │
         │    ├─ gpt(prompt)  →  LLM 返回多行候选
         │    │     "4+5=9 (left: 6 9 10)\n
         │    │      4*5=20 (left: 6 10 20)\n
         │    │      ..."
         │    │
         │    └─ 每个候选前加 y 前缀，变成:
         │         ["4+5=9 (left: 6 9 10)\n",
         │          "4*5=20 (left: 6 10 20)\n", ...]
         │
         │  get_values(task, x, new_ys)
         │    │
         │    └─ 对每个候选调用 value_prompt_wrap → 拼 prompt → gpt()
         │       LLM 返回 "sure"/"likely"/"impossible"
         │       ↓
         │       value_outputs_unwrap 映射为分数 [1, 20, 1, 0.001, ...]
         │
         │  select(new_ys, values, "greedy", n=1)
         │    └─ 取最高分的候选: "4*5=20 (left: 6 10 20)\n"
         │
         Step 2: (重复上轮流程，但 y 变了)
         │  y = "4*5=20 (left: 6 10 20)\n"
         │  get_proposals(...) → LLM 看 "6 10 20" → 生成候选
         │  get_values(...) → 打分
         │  select(...) → 取 "10-6=4 (left: 4 20)\n"
         │
         Step 3:
         │  ...
         │  取 "4+20=24 (left: 24)\n"
         │
         Step 4:
            y = "4+20=24 (left: 24)\n"
            propose_prompt_wrap 检测到 24 → 切换为 cot_prompt
            LLM 输出最终答案: "Answer: (4*5)+(10-6) = 24"

run.py:  test_output(900, ys[0])
           ├─ 提取表达式: "(4*5)+(10-6)"
           ├─ 校验数字: [4,5,10,6] == [4,5,6,10] ✓
           └─ sympy.simplify: == 24 ✓  →  {'r': 1}
```

---

## 九、项目文件结构

```
tree-of-thought-llm-master/
├── setup.py                    # pip install -e .
├── requirements.txt            # 5个依赖包
├── ToT架构分析.md               # 本文档
├── src/
│   ├── run.py                  # 入口
│   └── tot/
│       ├── __init__.py
│       ├── models.py           # 第一层：LLM 调用
│       ├── methods/
│       │   ├── __init__.py
│       │   └── bfs.py          # 第四层：搜索算法（Generate→Evaluate→Select）
│       ├── tasks/
│       │   ├── __init__.py     # get_task() 工厂函数
│       │   ├── base.py         # Task 抽象基类
│       │   └── game24.py       # 第三层：24点任务适配
│       ├── prompts/
│       │   ├── __init__.py
│       │   └── game24.py       # 第二层：few-shot prompt 模板
│       └── data/
│           └── 24/
│               └── 24.csv      # 1362道24点题目
```

---

## 十、常见问题

### Q1: 为什么不是真正的 BFS？

真正的 BFS 每层展开所有节点。但 ToT 每层只保留 `n_select_sample` 个（默认 1 个），本质是**束搜索（Beam Search）**。叫 ToT 是因为它"像搜树一样多分支探索"。

### Q2: propose vs sample 有什么区别？

| | propose | sample |
|---|---|---|
| **LLM 输出** | 多个候选下一步的列表 | 一个完整答案 |
| **粒度** | 细粒度，每步一个运算 | 粗粒度，直接给最终答案 |
| **为什么用 propose** | 需要搜索多个分支 | 不需要搜索 |

### Q3: value vs vote 有什么区别？

| | value | vote |
|---|---|---|
| **评估方式** | 逐个独立打分 | 多个候选一起比较投票 |
| **适用场景** | 候选少（<20个）| 候选多（>20个） |
| **为什么用 value** | 24点每步候选不多 | — |

### Q4: 为什么 temperature 必须 > 0？

`temperature=0` 时 LLM 每次输出完全一样，Propose 生成的全是同一个候选，搜索退化为贪心，ToT 失去意义。建议 `0.7`。

### Q5: value_cache 会不会内存溢出？

不会。24 点一共就 4 步，每步候选不超过 20 个，同一个 Task 实例最多缓存几十条 prompt。

### Q6: 想换其他任务怎么改？

1. 在 `tasks/` 下新建 `mytask.py`，继承 `Task`，实现 `get_input` / `test_output` / 各种 prompt_wrap
2. 在 `tasks/__init__.py` 的 `get_task()` 中注册
3. 在 `prompts/` 下新建 `mytask.py`，写 few-shot 示例
4. `--task` 参数增加选项

**bfs.py 完全不用动**。

### Q7: 之前代码的 global gpt bug 是什么？

旧代码在 `solve()` 里写了 `global gpt; gpt = partial(gpt, model=...)`。每调用一次 `solve`，`gpt` 就被包一层 `partial`。第二次调用时 `model` 参数存在两次 → `TypeError: got multiple values for 'model'`。

**修复方法**：不再修改全局 `gpt`，直接把 `model` 和 `temperature` 作为参数传给每个子函数。

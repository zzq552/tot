import os
from openai import OpenAI
import backoff

completion_tokens = prompt_tokens = 0

# 豆包 API 配置
DOUBAO_API_KEY = "ark-003ae8fd-6959-4cce-924d-1288f12616b8-66ae3"
DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DOUBAO_MODEL = "ep-m-20260511182947-xmt9d"  # 豆包推理端点

client = OpenAI(
    api_key=DOUBAO_API_KEY,
    base_url=DOUBAO_BASE_URL,
)

@backoff.on_exception(backoff.expo, Exception)
def completions_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

def gpt(prompt, model=DOUBAO_MODEL, temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def chatgpt(messages, model=DOUBAO_MODEL, temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    # 豆包端点不支持 n>1，逐个发送独立请求
    for _ in range(n):
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=1, stop=stop)
        outputs.extend([choice.message.content for choice in res.choices])
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
    return outputs

def gpt_usage(backend=DOUBAO_MODEL):
    global completion_tokens, prompt_tokens
    # 豆包计费参考，按实际价格调整
    cost = (completion_tokens + prompt_tokens) / 1000 * 0.005
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}

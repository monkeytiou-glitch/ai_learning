import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 1. 加载 .env 文件中的环境变量
load_dotenv()

# 2. 初始化智谱模型 (使用免费的 glm-4-flash)
llm = ChatOpenAI(
    model="glm-4-flash", 
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    max_retries=0 
)

# 3. 发起单轮对话
print("正在呼叫智谱...")
try:
    res = llm.invoke("你好，请用一句话介绍一下你自己。")
    print("\n【模型回复】：")
    print(res.content)
except Exception as e:
    print(f"\n❌ 出错了: {e}")
# import os
# #dotenv 用于从.env文件中读取键值对配置到系统环境变量的库
# from dotenv import load_dotenv   #开保险箱的钥匙
# from langchain_openai import ChatOpenAI  #langchain_openai是LangChain的OpenAI模块，用于调用兼容OpenAI接口格式的大模型服务

# # 1. 加载 .env 密钥（打开保险箱，取出通行证）
# load_dotenv()   # dotenv库提供的方法，从 .env 文件中加载环境变量到当前的运行环境

# # 2. 初始化智谱模型
# #llm=Large Language Model 大语言模型
# llm = ChatOpenAI(
#     model="glm-4-flash", 
#     api_key=os.getenv("ZHIPU_API_KEY"),
#     base_url="https://open.bigmodel.cn/api/paas/v4/",#base_url是智谱大模型的API接口地址
#     max_retries=0 #max_retries是请求失败时的最大重试次数,=0失败后不再重试，防止死循环
# )

# # ==========================================
# # Week 2 实验 1：最原始的 Prompt 拼接
# # ==========================================

# # 测试文本
# text = "我叫李四，今年32岁，是个资深后端开发，精通Java和Go，之前在腾讯干了5年，现在想找个远程工作。"

# # 把要求和我们的话拼在一起，直接用普通的字符串发给模型
#  #prompt提示词，f-string格式化字符串
# prompt = f"""    
# 请从下面这段文字中提取信息，并输出为 JSON 格式。
# 包含字段：姓名(name)、年龄(age)、职位(job)、技能(skills)、工作经验(experience)、期望(target)。

# 文本内容：{text}
# """

# print("正在呼叫模型提取信息...\n")

# # 发起对话
# response = llm.invoke(prompt)   

# # 打印模型的原样输出
# print("【模型原始输出】：")
# print(response.content)


# #从 外联部 拿了电话机 (ChatOpenAI)。

# 从 核心大楼的话术车间 拿了表格 (ChatPromptTemplate)。

# 从 核心大楼的质检车间 拿了质检员 (JsonOutputParser)。
# 然后把他们三个串在流水线（chain）上，开始干活！
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# 1. 照常加载环境变量和模型
load_dotenv()
llm = ChatOpenAI(
    model="glm-4-flash", 
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    max_retries=0 
)

# 2. 准备测试文本
text = "我叫李四，今年32岁，是个资深后端开发，精通Java和Go，之前在腾讯干了5年，现在想找个远程工作。"

# 3. 设置一个“强制解析器”
# 这个解析器的任务就是：不管模型说什么，只要不是纯JSON，它就报错；如果是纯JSON，它自动帮你转换成 Python 字典
parser = JsonOutputParser()

# 4. 构建对话模板（注意这里的 {format_instructions}，是解析器自带的强力要求）
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的信息提取助手。请从用户输入的文本中提取信息。\n注意：请严格输出英文的键名，包含name、age、job、skills、experience、target。\n{format_instructions}"),
    ("human", "{input_text}")
])

# 5. 把模板、模型、解析器连成一条流水线（这就是 LangChain 的精髓：Chain）
chain = prompt_template | llm | parser

# 6. 运行流水线
print("正在呼叫模型并强制解析...\n")
try:
    # 注意：这里我们要把文本和格式指令一起传进去
    result = chain.invoke({
        "input_text": text,
        "format_instructions": parser.get_format_instructions()
    })
    
    print("【解析后的完美结果】：")
    print(result)
    
    # 验证一下：现在我们得到的是一个真正的 Python 字典
    print("\n【测试提取数据】：")
    print(f"这个人的名字是：{result['name']}")
    print(f"他的技能是：{result['skills']}")
    
except Exception as e:
    print(f"糟糕，解析失败了，报错信息：{e}")
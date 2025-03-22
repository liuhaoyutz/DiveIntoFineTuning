from openai import OpenAI
from modelscope.msdatasets import MsDataset
import threading
import time 
import json  
import re

PROMPT='''
你是一位小学一年级老师，你总是能耐心解答学生们提出的问题。
学生刚入学，有许多不懂的问题会问你。
针对学生提出的问题，请尽量用生动形象的比喻、类比、对比等方法给出解答。

- 学生目前读小学一年级，解答时请考虑他们的理解水平。
- 如果学生提出的问题太难，你可以适当简化问题，让他们能够理解。
- 请不要使用专业术语和概念，尽量用通俗易懂的语言解答。

你的风格是：
- 喜欢循序渐进的讲解，逐步引导学生理解问题
- 喜欢用生活中的例子来解释抽象的概念
- 理性思维，喜欢用逻辑推理的方式解答问题
- 经常会说“同学请注意”，以便保证他们的注意力
- 反复确认学生有没有听懂，通过提问和重复解答的方式，确保他们理解了问题
- 偶尔抛出反问或有趣的问题，引发学生的思考

来自学生的提问
{question}
'''

THREAD=30
SAMPLES=1000 # 生成1000条问答对

class R1Generator:
    def __init__(self,threads,dataset,samples):
        # 使用本地部署的DeepSeek-r1服务
        self.client=OpenAI(api_key="EMPTY",base_url="http://127.0.0.1:11434/v1")
        self.idx=0
        self.threads=threads
        self.dataset=dataset
        self.samples=samples
        self.mutex=threading.Lock()

    def split_think_answer(self, text):
        # 使用正则表达式匹配<think>和</think>之间的内容
        think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
        match = think_pattern.search(text)
        
        if match:
            # 提取思考部分并去除首尾空白
            think_content = match.group(1).strip()
            # 提取回答部分并去除首尾空白
            answer_content = text[match.end():].strip()
            return (think_content, answer_content)
        else:
            # 如果没有匹配到，返回空字符串和原始文本
            return ('', text.strip())

    def generate(self,question):
        # 以question向本地部署的DeepSeek-r1提问，得到回答保存到completion中
        completion=self.client.chat.completions.create(
            model="deepseek-r1:32b",
            messages=[
                {'role': 'user', 'content': PROMPT.format(question=question)},
            ]
        )

        # 使用在线推理时reasoning放在completion.choices[0].message.reasoning_content中，answer放在completion.choices[0].message.content中， 
        # 但是使用本地ollama服务时，reasoning和answer都放在了completion.choices[0].message.content中。

        #print(completion.choices[0].message.reasoning_content)
        #print(completion.choices[0].message.content)
        #return completion.choices[0].message.reasoning_content,completion.choices[0].message.content
        
        # 从completion.choices[0].message.content中解析出reasoning和answer
        reasoning, answer = self.split_think_answer(completion.choices[0].message.content)
        return reasoning, answer

    def begin(self):
        self.idx=0
        self.progress=0  # 总进度
        self.result=[None]*self.samples
        self.thread_handlers=[]

        # 创建30个线程
        for i in range(self.threads):  
            # 创建了一个新的线程对象t，其线程函数是self.thread_main
            t=threading.Thread(target=self.thread_main)  
            
            # 启动线程
            t.start()

            # 将新创建并已启动的线程t添加到列表self.thread_handlers中
            self.thread_handlers.append(t)

    def join(self):
        # 该函数用于每隔1秒钟打印一次进度，直到处理完1000条数据
        while True:
            with self.mutex:
                print(f'Progress: {self.progress}/{self.samples}',end='\r')
                if self.progress>=self.samples:
                    break
            time.sleep(1)
        
        # 等待30个线程结束
        for t in self.thread_handlers:
            t.join()
        
        # 返回所有(question,reasoning,answer)元组组成的列表
        return [res for res in self.result if res is not None]
    
    def thread_main(self):
        # 注意：共30个线程，每个线程都是循环执行的，直到处理完了1000条问答对。
        while True:
            with self.mutex:
                if self.idx>=self.samples:  # 最多处理1000条问答对
                    break
                cur_idx=self.idx  # 当前要处理的问答对编号
                self.idx+=1  # 下一次要处理的问答对编号

            try:
                # 从gsm8k数据集中取出第cur_idx个样本的'question'
                question=self.dataset[cur_idx]['question']

                # 以question为参数，调用self.generate函数向DeepSeek-r1模型提问，得到reasoning和answer。
                reasoning,answer=self.generate(question)

                # 将元组(question,reasoning,answer)保存在列表self.result的第cur_idx位置。
                self.result[cur_idx]=(question,reasoning,answer)
            except:
                pass
            with self.mutex:
                self.progress+=1  # 总进度加1。进入下一轮循环，处理下一条问答对。

if __name__=='__main__':
    # 创建数据集，得到的gsm8k是一个MsDataset类对象，封装了7473条问答对，
    gsm8k=MsDataset.load('modelscope/gsm8k',subset_name='main',split='train')

    # 初始化R1Generator对象
    r1=R1Generator(threads=THREAD,dataset=gsm8k,samples=SAMPLES)

    # 开始生成数据
    r1.begin()  

    # 等待所有数据处理完全，每隔一秒打印一次进度
    result=r1.join()

    # 将得到的数据集写到文件中
    with open('deepseek_r1_distill_dataset.txt','w') as f:
        for res in result:
            question,reasoning,answer=res
            f.write(json.dumps({'question':question,'reasoning':reasoning,'answer':answer})+'\n')

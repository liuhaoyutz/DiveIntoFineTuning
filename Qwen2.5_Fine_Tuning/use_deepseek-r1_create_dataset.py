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
SAMPLES=1000

class R1Generator:
    def __init__(self,threads,dataset,samples):
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
        completion=self.client.chat.completions.create(
            model="deepseek-r1:32b",
            #model="deepseek-reasoner",
            messages=[
                {'role': 'user', 'content': PROMPT.format(question=question)},
            ],
            extra_body={  # 某些平台允许扩展参数
                "return_reasoning": True
            }
        )

        # 使用在线推理时reasoning放在completion.choices[0].message.reasoning_content中，answer放在completion.choices[0].message.content中， 
        # 但是使用本地ollama服务时，reasoning和answer都放在了completion.choices[0].message.content中。

        #print(completion.choices[0].message.reasoning_content)
        #print(completion.choices[0].message.content)
        #return completion.choices[0].message.reasoning_content,completion.choices[0].message.content
        
        reasoning, answer = self.split_think_answer(completion.choices[0].message.content)
        return reasoning, answer

    def begin(self):
        self.idx=0
        self.progress=0
        self.result=[None]*self.samples
        self.thread_handlers=[]
        for i in range(self.threads):
            t=threading.Thread(target=self.thread_main)
            t.start()
            self.thread_handlers.append(t)

    def join(self):
        while True:
            with self.mutex:
                print(f'Progress: {self.progress}/{self.samples}',end='\r')
                if self.progress>=self.samples:
                    break
            time.sleep(1)
        for t in self.thread_handlers:
            t.join()
        return [res for res in self.result if res is not None]
    
    def thread_main(self):
        while True:
            with self.mutex:
                if self.idx>=self.samples:
                    break
                cur_idx=self.idx
                self.idx+=1
            try:
                question=self.dataset[cur_idx]['question']
                reasoning,answer=self.generate(question)
                self.result[cur_idx]=(question,reasoning,answer)
            except:
                pass
            with self.mutex:
                self.progress+=1

if __name__=='__main__':
    gsm8k=MsDataset.load('modelscope/gsm8k',subset_name='main',split='train')
    r1=R1Generator(threads=THREAD,dataset=gsm8k,samples=SAMPLES)
    r1.begin()
    result=r1.join()

    with open('deepseek_r1_distill_dataset.txt','w') as f:
        for res in result:
            question,reasoning,answer=res
            f.write(json.dumps({'question':question,'reasoning':reasoning,'answer':answer})+'\n')

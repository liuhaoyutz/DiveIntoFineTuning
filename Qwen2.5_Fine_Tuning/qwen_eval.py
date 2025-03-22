from modelscope import AutoModelForCausalLM, AutoTokenizer
import os

# Load base model
model_name='Qwen/Qwen2.5-3B-Instruct'
model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype="auto",device_map="auto")
tokenizer=AutoTokenizer.from_pretrained(model_name)

# Find latest checkpoint
checkpoints=os.listdir('qwen_distill/')

# 用filter函数过滤checkpoints列表中的元素，只保留那些以字符串'checkpoint'开头的元素。lambda x: x.startswith('checkpoint')是一个匿名函数，用于定义过滤条件。
# 用sorted函数对通过过滤后的检查点进行排序。key参数指定了一个用来从每个元素中提取比较键的函数。lambda x: int(x.split('-')[-1]) 这个匿名函数将每个检查点名称按 '-' 分割，并转换分割后最后一部分为整数作为排序依据。
# 最后，通过索引 [-1] 获取排序后列表中的最后一个元素，也就是具有最大步骤编号的最新检查点。
latest_checkpoints=sorted(filter(lambda x: x.startswith('checkpoint'),checkpoints),key=lambda x: int(x.split('-')[-1]))[-1]
lora_name=f'qwen_distill/{latest_checkpoints}'

SYSTEM_PROMPT='''
你是一位小学老师，给一年级学生解答问题。

用如下格式回答学生的问题：
<think>
针对问题，逐步拆解、分析、反思，整理解答思路。
</think>
以老师的第一人称视角，给学生开始讲解。
'''

def eval_qwen(model,query):
    # 准备发送给Qwen的消息
    messages=[
        {'role':'system','content':SYSTEM_PROMPT}, 
        {'role':'user','content': query}, 
        {'role':'assistant','content': '<think>'}
    ]

    # 应用模板并做tokenize
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,continue_final_message=True)
    model_inputs=tokenizer([text], return_tensors="pt").to(model.device)

    # 与Qwen对话，得到回复generated_ids
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=4000,
    )

    # 解码
    completion_ids=generated_ids[0][len(model_inputs.input_ids[0]):]
    completion_text=tokenizer.decode(completion_ids,skip_special_tokens=True)
    
    return '<think>'+completion_text

query='龟兔赛跑教给我们什么道理?'

# Base Model Test
completion=eval_qwen(model,query)
print('base model:',completion)

# Lora Model Test
print('merge lora:',lora_name)
model.load_adapter(lora_name)
completion=eval_qwen(model,query)
print('lora model:',completion)

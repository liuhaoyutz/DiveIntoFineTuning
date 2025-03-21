from modelscope import AutoModelForCausalLM, AutoTokenizer
import os

# Load base model
model_name='Qwen/Qwen2.5-3B-Instruct'
model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype="auto",device_map="auto")
tokenizer=AutoTokenizer.from_pretrained(model_name)

# Find latest checkpoint
checkpoints=os.listdir('qwen_distill/')
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
    messages=[
        {'role':'system','content':SYSTEM_PROMPT}, 
        {'role':'user','content': query}, 
        {'role':'assistant','content': '<think>'}
    ]
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,continue_final_message=True)
    model_inputs=tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=4000,
    )
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

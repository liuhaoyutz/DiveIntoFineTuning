import gradio as gr
import os 
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
from threading import Thread

MAX_HISTORY_LEN=3
SYSTEM_PROMPT='''
# 任务
你是一位小学老师，给学生解答问题。

请用如下格式回答学生的问题：
<think>
针对问题，逐步拆解、分析、反思，整理解答思路。
</think>
以老师的第一人称视角，给学生开始讲解。
'''

def chat_streaming(model_selector,query,history):
    messages=[
        {'role':'system','content':SYSTEM_PROMPT}, 
    ]
    for q,a in history:
        messages.append({'role':'user','content': q}, )
        messages.append({'role':'assistant','content': a}, )
    messages.append({'role':'user','content': query}, )
    messages.append({'role':'assistant','content': '<think>'})
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,continue_final_message=True)
    model_inputs=tokenizer([text], return_tensors="pt").to(model.device)

    if model_selector=='Qwen Base Model':
        model.disable_adapters()  # 使用原始Qwen模型
    else:
        model.enable_adapters()  # 使用经过SFT的Qwen模型

    streamer=TextIteratorStreamer(tokenizer,skip_prompt=True,skip_special_tokens=True)
    generation_kwargs=dict(model_inputs,streamer=streamer,max_new_tokens=2000)

    # 与模型进行对应
    thread=Thread(target=model.generate,kwargs=generation_kwargs)
    thread.start()
    for resp in streamer:
        yield resp
    thread.join()

with gr.Blocks(css='.qwen-logo img {height:200px; width:600px; margin:0 auto;}') as app:
    with gr.Row():
        chatbot=gr.Chatbot(label='DeepSeek Distill Qwen')
    with gr.Row():
        model_selector=gr.Dropdown(choices=['Qwen Base Model', 'Qwen Distll Model'],label='选择模型')
    with gr.Row():
        query_box=gr.Textbox(label='输入',autofocus=True,lines=2)
    with gr.Row():
        clear_btn=gr.ClearButton([query_box,chatbot],value='清空历史')
        submit_btn=gr.Button(value='提交')

    def chat(model_selector,query,history):
        full_resp='<think>'
        replace_resp=''
        for response in chat_streaming(model_selector,query,history):
            full_resp=full_resp+response
            replace_resp=full_resp.replace('<think>','[开始思考]\n').replace('</think>','\n[结束思考]\n')
            yield '',history+[(query,replace_resp)]
        history.append((query,replace_resp))
        while len(history)>MAX_HISTORY_LEN:
            history.pop(0)
    
    # 提交query
    submit_btn.click(chat,[model_selector,query_box,chatbot],[query_box,chatbot])

if __name__ == "__main__":
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
    
    model.load_adapter(lora_name)
    
    app.queue(200)  # 请求队列
    app.launch(server_name='0.0.0.0',max_threads=500) # 线程池

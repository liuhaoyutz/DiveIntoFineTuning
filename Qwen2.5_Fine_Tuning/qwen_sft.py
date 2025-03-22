from datasets import Dataset
from trl import SFTConfig, SFTTrainer
import json
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import LoraConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SYSTEM_PROMPT='''
# 任务
你是一位小学老师，给学生解答问题。

# 回答格式
<think>
针对问题，逐步拆解、分析、反思，整理解答思路。
</think>
以老师的第一人称视角，给学生开始讲解。
'''

# 从文本文件中加载数据集，并转换为Hugging Face的datasets库中的Dataset对象
def load_distill_dataset():
    # ds是一个字典，'messages'是key, value是一个初始为空的列表
    ds={'messages':[]}
    with open('deepseek_r1_distill_dataset.txt','r') as f:
        lines=f.readlines()
        for line in lines:
            line=json.loads(line)  # 每一行是一个以json格式保存的样本，json_loads将json格式数据解析为python字典

            # 创建sample列表，它代表一次对话，包括3个字典，分别代表'system', 'user', 'assistant'的消息
            sample=[
                    {'role':'system','content':SYSTEM_PROMPT}, 
                    {'role':'user','content': line['question']}, 
                    {'role':'assistant','content': f"<think>{line['reasoning']}</think>{line['answer']}"},
            ]

            # 把sample添加到ds['messages']列表中。
            ds['messages'].append(sample)

    # ds是一个字典，用ds初始化Dataset对象并返回
    return Dataset.from_dict(ds)

# 加载模型
model_name = 'Qwen/Qwen2.5-3B-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载数据集
dataset = load_distill_dataset()

# 配置SFT参数
sft_config = SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    lr_scheduler_type='linear',
    warmup_ratio=0.1,
    learning_rate=5e-6,
    max_seq_length=500,
    logging_steps=1,
    save_steps=0.1,
    num_train_epochs=2,
    report_to='tensorboard',
    fp16=True,
    max_grad_norm=0.1,
    output_dir='./qwen_distill/',
)

# 配置LoRA微调参数
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    lora_dropout=0.05,
    task_type='CAUSAL_LM',
)

# 生成训练器
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=sft_config,
    peft_config=lora_config,
)

# 开始训练
trainer.train()

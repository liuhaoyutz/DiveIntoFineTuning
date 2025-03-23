import json

def sharegpt_to_alpaca(sharegpt_data):
    alpaca_data = []
    
    for sample in sharegpt_data:
        conversations = sample.get("conversations", [])
        
        # 确保对话至少包含一轮 user 和 assistant
        if len(conversations) >= 2 and conversations[0]["from"] == "user" and conversations[1]["from"] == "assistant":
            instruction = conversations[0]["value"].strip()
            output = conversations[1]["value"].strip()
            
            # 构建 Alpaca 格式的样本
            alpaca_sample = {
                "instruction": instruction,
                "input": "",  # 如果有上下文信息，可以填充到这里
                "output": output
            }
            alpaca_data.append(alpaca_sample)
    
    return alpaca_data

# 示例：读取 ShareGPT 数据并转换为 Alpaca 格式
if __name__ == "__main__":
    # 读取 ShareGPT 格式的数据
    with open("sharegpt_dataset.json", "r", encoding="utf-8") as f:
        sharegpt_data = json.load(f)
    
    # 转换为 Alpaca 格式
    alpaca_data = sharegpt_to_alpaca(sharegpt_data)
    
    # 保存为 Alpaca 格式的数据
    with open("alpaca_dataset.json", "w", encoding="utf-8") as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
    
    print(f"Converted {len(alpaca_data)} samples to Alpaca format.")
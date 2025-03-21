# DiveIntoFineTuning

Qwen2.5_Fine_Tuning目录下的脚本实现如下功能：  
1、基于ollama本地部署DeepSeek-r1服务，通过DeepSeek-r1生成高质量问答对，对Qwen2.5模型进行蒸馏微调。  
2、验证Qwen2.5蒸馏微调模型与原始模型差异。  

基于DeepSeek-r1生成对话数据集：  
python use_deepseek-r1_create_dataset.py  

基于生成的对话数据集对Qwen2.5进行SFT微调：  
python qwen_sft.py  

命令行验证微调效果：  
python qwen_eval.py  

图形化界面验证微调效果：  
python qwen_webui.py  

Qwen目录下提供对Qwen模型进行SFT微调的脚本。  

Reference：  
https://github.com/owenliang/DeepSeek-Distill-Qwen-For-Child

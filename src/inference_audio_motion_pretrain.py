import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig, Gemma3ForConditionalGeneration, Gemma3Config


model_id = "/root/pengyang/codebase/SOLAMI/output/pretrain_audio_motion_final"


from peft import PeftModel

# Load Model base model




# 步骤2：初始化自定义模型
model = Gemma3ForConditionalGeneration()
model.resize_token_embeddings(263685)
state_dict = torch.load("your_trained_model/pytorch_model.bin")
model.load_state_dict(state_dict, strict=True) 

model = Gemma3ForConditionalGeneration.from_pretrained(model_id, low_cpu_mem_usage=True)

# Merge LoRA and base model and save
peft_model = PeftModel.from_pretrained(model, model_id)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("merged_model", safe_serialization=True, max_shard_size="2GB")

processor = AutoTokenizer.from_pretrained(model_id)
processor.save_pretrained("merged_model")



if torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float16

# Load Model with PEFT adapter
model = AutoModelForImageTextToText.from_pretrained(
  model_id,
  device_map="auto",
  torch_dtype=torch_dtype,
  attn_implementation="eager",
)



tokenizer = AutoTokenizer.from_pretrained(model_id)
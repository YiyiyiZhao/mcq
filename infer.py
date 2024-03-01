import json
import torch
import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="model_out_32_0.0001_20")
args = parser.parse_args()


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

print(torch.cuda.is_available())
print( torch.cuda.device_count())
base_model_path = "/data/zy/models/llama-2-chat-7b-hf"
adapter_path=args.output_dir
flag=adapter_path.replace('model_out_','')

base_model = AutoModelForCausalLM.from_pretrained(base_model_path,torch_dtype=torch.float16).cuda()
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.use_default_system_prompt = False


with open("data/test_data.json", "r") as f:
    test_dataset=json.load(f)
def chat_with_llama(model, tokenizer,prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to('cuda')
    output = model.generate(input_ids)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

model.eval()
with torch.no_grad():
    dataset_len=len(test_dataset)
    predictions=[]
    for i in range(dataset_len):
        ann=test_dataset[i]
        gt=ann['output']
        prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        response = chat_with_llama(model, tokenizer, prompt)
        pred= response[-1]
        predictions.append({"gt":gt, "pred":pred, "input":ann['input']})

with open("pred.json".format(flag), 'w') as f:
    json.dump(predictions,f)

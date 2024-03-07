import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default="/data/zy/models/llama-2-chat-7b-hf")
parser.add_argument("--adapter_dir", type=str, default="/data/zy/projects/adapters/adapter_llama_frame_summary")
args = parser.parse_args()

def get_hard_response_desc(sample):
    obj_list=[item['object'].replace('person','') for item in sample]
    if len(obj_list)==1:
        hard_res="There is a {} at your surrounding.".format(obj_list[0])
    else:
        hard_res="There is " +'a, '.join(obj_list[:-1])+ ' and a '+obj_list[-1] +' at your surrounding.'
    return hard_res

def chat_with_llama(model, tokenizer,prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to('cuda')
    output = model.generate(input_ids, num_beams=4, no_repeat_ngram_size=2)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


def infer_desc(in_obj_list):
    #load prompt template
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
    instruct_desc="Task requirements: Given the object detection results of an image with dimensions (720, 1280, 3), your task is to generate a scene description suitable for visually impaired individuals.\nInput: You'll be provided with a list of dictionaries, containing the upper-left (x1, y1) and bottom-right (x2, y2) coordinates of each object in the scene.\nOutput requirements: \n1. Transform the ((x1, y1), (x2, y2)) coordinates into descriptive words like 'in front,' 'left,' 'right,' and try to be as accurate as possible. Do not return coordinate numbers; you may use clock directions (e.g., 3 o'clock direction, 9 o'clock direction, etc.),but you can only at most use twice.\n2. Ensure clarity and spatial understanding for the target users. You can provide additional descriptions, such as the relative position of one object to another (e.g., above, to the right), allowing visually impaired individuals to create a complete 3D spatial representation in their minds.\n3. Only return the pure descriptive text; ensure it's of an appropriate length,less than 80 words.\n4. If a 'person' has significant overlap with the coordinates of a specific object, it may indicate that the user's hand has touched the specified object. Remember to include in the description that the hand has touched an object and specify which object it has touched."

    inp = {"instruction": instruct_desc, "input": in_obj_list}
    prompt = PROMPT_DICT["prompt_input"].format_map(inp)
    response = chat_with_llama(model, tokenizer, prompt)
    parts = response.split("### Response:")
    if len(parts) > 1:
        response_final = parts[1]
    else:
        response_final = get_hard_response_desc(in_obj_list)
    return response_final

if __name__ == '__main__':
    #load models and tokenizer
    base_model_path = args.base_model_path
    adapter_path=args.adapter_dir

    base_model = AutoModelForCausalLM.from_pretrained(base_model_path,torch_dtype=torch.float16).cuda()
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()


    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.use_default_system_prompt = False

    #get input
    in_obj_list = [{"idx": 0, "object": "laptop", "coordinate": [[0, 390], [610, 712]]}, {"idx": 1, "object": "bottle", "coordinate": [[161, 244], [300, 417]]}, {"idx": 2, "object": "bowl", "coordinate": [[266, 1], [598, 181]]}]

    #get inference description of this frame
    res=infer_desc(in_obj_list)
    print(res)






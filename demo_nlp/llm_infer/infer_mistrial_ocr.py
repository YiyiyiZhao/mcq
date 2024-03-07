import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import re
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default="/data/zy/models/mistrial-7b-instruct-v0.2")
args = parser.parse_args()


def response_post_processing(text):
    pattern = re.compile(r'\[/INST\](.*?)</s>', re.DOTALL)
    match = pattern.search(text)
    res=''
    if match:
        res=match[0].replace('[/INST]','').replace('</s>','').strip()
    return res

def get_hard_response_ocr(text):
    floats = re.findall(r'\d+\.\d+', text)
    if floats:
        hard_res=floats[0]
    else:
        hard_res="0.0"
    return hard_res

def chat_with_mistrial(model, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to("cuda")
    generated_ids = model.generate(model_inputs, max_new_tokens=128, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    response=decoded[0]
    return response

def infer_ocr_total(in_text):
    # load prompt template
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
    instruct_ocr = "You will be provided with a text string. Please address the question of calculating the total cost as indicated in the text string. You may find the numerical value of the cost following the keyword \"total.\" Here are the requirements:\n1. The floating-point number representing the cost may be found following the keyword \"total\" or its variants due to OCR errors. These variants may include \"TTOL\", \"totL\", among others. \n2. It is crucial not to alter the value; you must return the floating-point number exactly as it appears in the text string, excluding symbols like '$'.\n3. If the answer is not discernible, return the null string. Then, output the result purely answer (represented as a floating-point number.) without any other words. \n4. Please ensure to output only answer without additional words.\n\nhere is a demonstration:\ntext: \"PAPPAREAUX\nSEAPOOD KITCHEN\nPappadeaux Seafood Kitchen030\n1304Copeand RdArlingonTX76011\n817543-0544\nww.Pappadeaux.com\nO035Table 72#Party 3\nSARAH H\nSvrCk:212:1111/18/17\nDINE IN\n1 Lump Crab&Spinach Dip\n15.95\n18.95\n1 Cajun Combo\n1 Lunch Shrimp& Andouille\n13.95\nSub Total:\n48.85\nTax:\n3.91\nTotal:\n52.76\n11/1812:45TOTAL\n52.76\nThank you for dining at Pappadeaux\nTip Not Included\n15%\n7.91\n18\n9.50\n20%\n10.55\ne-Gift Card Payment19 Digits\nPresent e-Gift Card for validation.\ndioitc\n\"\nyou should return: 52.76\n"

    inp = {"instruction": instruct_ocr, "input": in_text}
    prompt = PROMPT_DICT["prompt_input"].format_map(inp)
    response=chat_with_mistrial(model, tokenizer, prompt)
    if response:
        res_final=response_post_processing(response)
    else:
        res_final=get_hard_response_ocr(in_text)
    return res_final






if __name__=="__main__":
    #load model and tokenizer
    base_model_path = args.base_model_path
    model = AutoModelForCausalLM.from_pretrained(base_model_path).half()
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    #input
    in_text="1660 India Street\n12/17/2017\nServerJake\n6:29 PM\nTable 24/1\n20018\nGuests2\n12.00\nGl Karma Cab Sauy\nGl Douglass Green SB\n11.00\nPKLI Green Curry Scallops\n27.95\nPKLI Piri Piri Chicken\n23.75\nSubtotal\n74.70\nTax\n6.00\nSurcharge 3.75% SC*\n2.80\nTotal\n83.50\nBalanceDue\n83.50\n*3.75% SC - In support\nof the increases to min wage\nTo find out whyvisit\nwww.wnlhosp.com/rttd\n"


    #output
    total_cost=infer_ocr_total(in_text)
    print(total_cost)


















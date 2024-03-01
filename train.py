from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import argparse
from transformers import TrainerCallback
from contextlib import nullcontext
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset
from llama_recipes.configs.datasets import alpaca_dataset
from transformers import default_data_collator, Trainer, TrainingArguments


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="/data/zy/models/llama-2-chat-7b-hf")
parser.add_argument("--lora_r", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--num_train_epochs", type=int, default=5)
parser.add_argument("--acc_step", type=int, default=1)
args = parser.parse_args()

#print GPT info
print(torch.cuda.is_available())
print( torch.cuda.device_count())


#load model
model_id=args.model_id
tokenizer = LlamaTokenizer.from_pretrained(model_id, padding='max_length')
model =LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)

#load data
train_dataset = get_preprocessed_dataset(tokenizer, alpaca_dataset, 'train')

#preparing model for PEFT
model.train()

def create_peft_config(model,lora_r):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_int8_training,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules = ["q_proj", "v_proj"]
    )

    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config

# create peft config
model, lora_config = create_peft_config(model,args.lora_r)

#define an optical profile


enable_profiler = False
output_dir = "model_out_{}_{}_{}".format(args.lora_r, args.learning_rate, args.num_train_epochs)

config = {
    'lora_config': lora_config,
    'learning_rate': args.learning_rate,
    'num_train_epochs': args.num_train_epochs,
    'gradient_accumulation_steps': args.acc_step,
    'per_device_train_batch_size': 1,
    'gradient_checkpointing': False,
}

# Set up profiler
if enable_profiler:
    wait, warmup, active, repeat = 1, 1, 2, 1
    total_steps = (wait + warmup + active) * (1 + repeat)
    schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
    profiler = torch.profiler.profile(
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{output_dir}/logs/tensorboard"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True)


    class ProfilerCallback(TrainerCallback):
        def __init__(self, profiler):
            self.profiler = profiler

        def on_step_end(self, *args, **kwargs):
            self.profiler.step()


    profiler_callback = ProfilerCallback(profiler)
else:
    profiler = nullcontext()


#train
# Define training args
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    bf16=True,  # Use BF16 if available
    # logging strategies
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="no",
    optim="adamw_torch_fused",
    max_steps=total_steps if enable_profiler else -1,
    **{k:v for k,v in config.items() if k != 'lora_config'}
)

with profiler:
    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        callbacks=[profiler_callback] if enable_profiler else [],
    )

    # Start training
    trainer.train()

#at last
model.save_pretrained(output_dir)
#
#
# #reference
# model.eval()
# # with torch.no_grad():
# #     print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))
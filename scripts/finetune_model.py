from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
from PIL import Image
from datasets import Dataset, load_dataset
from transformers import TextStreamer, TrainingArguments
from tqdm import tqdm
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import argparse

IMAGE_ROOT = "/home/lz457/zhoulab/vlm-ad/vl-finetuning/instruct_data_3/"
DATA_PATH = "/home/lz457/zhoulab/vlm-ad/vl-finetuning/instruct_data_3/final_annotations.jsonl"
EVAL_PATH = "/home/lz457/zhoulab/vlm-ad/vl-finetuning/instruct_data_3/eval.jsonl"

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, 
    finetune_language_layers   = True,
    finetune_attention_modules = True, 
    finetune_mlp_modules       = True,

    r = 8, # Keep this relatively small since we don't need dramatic change  
    lora_alpha = 8,  
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  
    loftq_config = None, 
)

# 3. DATA PREPARATION: Load and format your JSONL dataset
print("Step 3: Loading and preparing JSONL dataset...")

# --- Load the JSONL file ---
try:
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    eval_dataset = load_dataset("json", data_files=EVAL_PATH, split="train")
except FileNotFoundError:
    print("="*80)
    print("ERROR: dataset not found.")
    print("See the documentation for the expected format.")
    print("="*80)
    exit()

# --- Function to load images from paths ---
def load_image(example):
    try:
        image = Image.open(IMAGE_ROOT+example["file_name"]).convert("RGB")
        return {"image": image}
    except Exception as e:
        print(f"Error loading image {example['file_name']}: {e}")
        return {"image": None}
    
def get_image_path(example):
    try:
        path = IMAGE_ROOT + example["file_name"]
        return{"image":path}
    except Exception as e:
        print("Error loading full image path.")
        return {"image":None}

# Apply the function to load images.
dataset = dataset.map(get_image_path, remove_columns=["file_name"])
dataset = dataset.filter(lambda example: example["image"] is not None)
eval_dataset = eval_dataset.map(get_image_path, remove_columns=["file_name"])
eval_dataset = eval_dataset.filter(lambda example: example["image"] is not None)


# --- Formatting Function (this remains the same) ---
def formatting_prompts_func(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    return {"text": text}



def create_optimized_prompt():
    """
    Creates a shorter, more direct prompt for better and consistent performance.
    """
    return (
         "You are an AI autonomous driving navigator. Your job is to determine a safe, efficient path for the ego-vehicle "
        "to take in order to continue on its route based on the birds-eye-view image of the scene and the positions of other actors. The red box at the center of the image is the ego-vehicle."
        "The the numbered blue objects are other cars. All units are in terms of meters. The vehicle is currently following its planned route. Do not make any abrupt changes. You must obey all laws of the road such as stopping for red lights. You will generate a set of 10 waypoints that represent a safe, efficient action and path for the vehicle to take over the next 5 seconds, "
        "with 0.5 seconds between each waypoint. In your response, put the final waypoint output as a list of number coordinates (meters) inside of "
        "<waypoint></waypoint> tags and nothing else, e.g [[x1, y1], [x2, y2]]. When thinking, consider "
        "what action should the vehicle take (e.g merge, stop, continue, turn)? What is the vehicle's current heading, direction, and route? Think step by step, and put your thinking in <think></think> tags. Consider the following as additional context:"
    )

instruction = create_optimized_prompt()
context = lambda sample: sample["ego_description"] + "Navigation notes: " + sample["instruction"] + "." + sample["light_description"] + sample["closest"]

def convert_to_conversation(sample):
    """
    Formats a data sample into the standard conversational structure
    for VLM fine-tuning, ensuring type consistency for all fields.
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    # The image comes first
                    {
                        "type": "image",
                        "image": "file://" + sample["image"],
                    },
                    # The text prompt follows
                    {
                        "type": "text",
                        "text":  instruction + context(sample)
                    }
                ]
            },
            {
                "role": "assistant",
                # The assistant's content must also be a list for type consistency.
                "content": [
                    {
                        "type": "text",
                        "text": sample["output_reasoning"]
                    }
                ]
            },
        ]
    }


# converted_dataset = dataset.map(
#     convert_to_conversation,
#     remove_columns=["ego_description", "instruction", "output_reasoning", "light_description", "closest", "nav_goal", "actor_description", "velocity_description", "image"],
# )

# converted_eval = eval_dataset.map(
#     convert_to_conversation,
#     remove_columns=["ego_description", "instruction", "output_reasoning", "light_description", "closest", "nav_goal", "actor_description", "velocity_description", "image"],
# )


converted_dataset = [convert_to_conversation(example) for example in tqdm(dataset)]
converted_eval = [convert_to_conversation(example) for example in tqdm(eval_dataset)]

# converted_eval = eval_dataset.map(convert_to_conversation(sample) for sample in eval_dataset)

# converted_dataset = converted_dataset.map(formatting_prompts_func)

print("--- Inspecting the first data sample ---")
print(converted_dataset[0]["messages"])
print(f"(The converted dataset has {len(converted_dataset)}) examples.")


FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = converted_dataset,
    eval_dataset = converted_eval,
    args = SFTConfig(
        gradient_accumulation_steps = 4,
        warmup_steps = 50,
        per_device_train_batch_size=1,
        # max_steps = 30,
        num_train_epochs = 3, # Set this instead of max_steps for full training runs
        learning_rate = 1e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "new_outputs",
        report_to = "none",     # For Weights and Biases
        eval_strategy = "steps",     # Run evaluation at regular step intervals
        eval_steps = 10,                  # Run evaluation every 100 training steps
        save_strategy = "steps",           # It's good practice to save based on eval
        save_steps = 10,
        load_best_model_at_end = True,     # Automatically load the best performing checkpoint
        

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 2048,
    ),
)


# ==> END OF DEBUGGING CODE
# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

model.save_pretrained("final_lora_model")  # Local saving
tokenizer.save_pretrained("final_lora_model")

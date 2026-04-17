import base64
import json
import os
import random
import re
import torch
import torch.distributed as dist
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

from io import BytesIO
from PIL import Image
from pprint import pprint
from qwen_vl_utils import process_vision_info
from scipy.stats import pearsonr, spearmanr
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor


#################### What your should modify ⬇️ ####################
# put the root dir here
root_dir = "../../../PhotoFramer-Assessment/"
# put the model dir here
MODEL_PATH = os.path.join(root_dir, "src/open-r1-multimodal/output/Qwen2.5-VL-7B-GRPO-Composition-Score-Class/")
# put your output dir here
OUTPUT_DIR = os.path.join(root_dir, "src/open-r1-multimodal/result/Qwen2.5-VL-7B-GRPO-Composition-Score-Class/")
os.makedirs(OUTPUT_DIR, exist_ok=True)
# image root, os.path.join(image root, image name in json file) should be the final path
IMAGE_ROOT = os.path.join(root_dir, "Datasets")
# json files for test
DATA_JSONS = [
    os.path.join(IMAGE_ROOT, "CADB_Dataset/metas/test_cadb_score_1k.json"),
    os.path.join(IMAGE_ROOT, "CADB_Dataset/metas/test_cadb_class_1k.json"),
]
# tasks, the same order to json files, composition_score: rating task; composition_class: classification task
TASKS = [
    "composition_score",
    "composition_class"
]
# batch size
BSZ = 4
#################### What your should modify ⬆️ ####################


QUESTION_TEMPLATE_DICT = {
    "composition_score": (
        "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. "
        "The final answer should be a float between 1 and 5, rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality. "
        "Return the final answer in JSON format with the following keys: \"rating\": The score."
    ),
    "composition_class": (
        "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. "
        "The final answer should be the composition types in the shown image. "
        "The composition type should be in [Rule of thirds, Vertical, Horizontal, Diagonal, Curved, Triangle, Center, Symmetric, Pattern, "
        "Golden ratio, Radial, Vanishing point, Fill the frame, None]. "
        "You can select one or more composition types for a sinle image. "
        "Return the final answer in JSON format using a list: [A list of all composition types]."
    )
}

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank) 
    
    dist.init_process_group(backend="nccl")
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    return local_rank, world_size, rank

local_rank, world_size, rank = setup_distributed()
device = f"cuda:{local_rank}"
print(f"Process {rank} using {device}")

#We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": local_rank}, 
)

# default processer
processor = AutoProcessor.from_pretrained(MODEL_PATH)

def extract_answer(content, task):
    # Try to find the score within <answer> tags, if can not find, return None
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        if task == "composition_class":
            composition_pattern = r'\[(.*?)\]'
            composition_match = re.search(composition_pattern, content_answer, re.DOTALL)
            if composition_match:
                composition_match = composition_match.group(1).strip().strip('"\'[]').strip()
                compositions = [_.strip().strip('"\'').strip() for _ in composition_match.split(',')]
                return compositions
        elif task == "composition_score":
            score_pattern = r'\d+\.\d+|\.\d+|\d+'
            score_match = re.search(score_pattern, content_answer)
            if score_match:
                score = float(score_match.group(0))
                return score
    return None


for data_json, task in zip(DATA_JSONS, TASKS):
    output_path = os.path.join(OUTPUT_DIR, os.path.basename(data_json))

    if rank == 0:
        print(f"Processing {data_json}...")
    data = json.load(open(data_json, "r"))

    QUESTION_TEMPLATE = QUESTION_TEMPLATE_DICT[task]

    # Split data for distributed evaluation
    per_rank_data = len(data) // world_size
    start_idx = rank * per_rank_data
    end_idx = start_idx + per_rank_data if rank < world_size - 1 else len(data)
    rank_data = data[start_idx:end_idx]

    messages = []

    for x in rank_data:
        image_path = os.path.join(IMAGE_ROOT, x['image'])
        if "box" in x:
            image = Image.open(image_path)
            w, h = image.width, image.height
            x1, y1, x2, y2 = x["box"]
            assert x2 <= w and y2 <= h
            image = image.crop((x1, y1, x2, y2))

            temp_output_dir = os.path.join(output_dir, "temp")
            os.makedirs(temp_output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(x['image']))[0]
            filename = f"{base_name}_box{x1}_{y1}_{x2}_{y2}.jpg"
            image_path = os.path.join(temp_output_dir, filename)
            image.save(image_path)

        message = [
            # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": f"file://{image_path}"
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=x['problem'])
                }
            ]
        }]
        messages.append(message)

    rank_outputs = [] # List to store answers for this rank
    all_outputs = []  # List to store all answers

    # Process data
    for i in tqdm(range(0, len(messages), BSZ), disable=rank != 0):
        batch_messages = messages[i:i + BSZ]
    
        # Preparation for inference
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=256, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        rank_outputs.extend(batch_output_text)

    print(f"Rank {rank} has finished processing {len(rank_outputs)} examples")

    # Gather all outputs from all ranks
    all_outputs = [None] * len(data)
    rank_results = [(start_idx + i, output) for i, output in enumerate(rank_outputs)]

    gathered_results = [None] * world_size
    dist.all_gather_object(gathered_results, rank_results)
    
    assert gathered_results[-1][-1][0] == len(data) - 1

    # The main process will collect all results
    if rank == 0:
        for results in gathered_results:
            for idx, output in results:
                assert idx < len(all_outputs)
                all_outputs[idx] = output
        assert all_outputs[-1] is not None

        final_output = []

        for input_example, model_output in zip(data, all_outputs):
            original_output = model_output
            ground_truth = input_example['solution']
            model_answer = extract_answer(original_output, task)

            # Create a result dictionary for this example
            result = {
                'image': input_example['image'],
                'question': input_example['problem'],
                'ground_truth': ground_truth,
                'model_output': original_output,
                'extracted_answer': model_answer,
            }
            final_output.append(result)

        # Save results to a JSON file
        with open(output_path, "w") as f:
            json.dump(final_output, f, indent=2)

        print(f"Results saved to {output_path}")
        print("-"*100)

    # Synchronize all processes
    dist.barrier()

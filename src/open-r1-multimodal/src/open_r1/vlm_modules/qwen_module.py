import os
import re
import torch

from datetime import datetime
from open_r1.vlm_modules.vlm_module import VLMBaseModule
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from trl.data_utils import maybe_apply_chat_template
from typing import Dict, Any, Union


class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2-VL" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        elif "Qwen2.5-VL" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return ['max_pixels', 'min_pixels']
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs

    @staticmethod
    def format_reward(completions, **kwargs):
        pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    @staticmethod
    def reward_composition_class(completions, solution, tasks, **kwargs):
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        composition_pattern = r'\[(.*?)\]'
        for content, sol, task in zip(contents, solution, tasks):
            reward = 0.0
            if task == "composition_class":
                # Try symbolic verification first
                compositions = []
                try:
                    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                    if content_answer_match:
                        content_answer = content_answer_match.group(1).strip()
                        composition_match = re.search(composition_pattern, content_answer, re.DOTALL)
                        if composition_match:
                            composition_match = composition_match.group(1).strip().strip('"\'[]').strip()
                            compositions = [_.strip().strip('"\'').strip() for _ in composition_match.split(',')]
                            compositions = list(set(compositions))  # Prevent repeated prediction
                            sol = [_.lower() for _ in sol]
                            for composition in compositions:
                                if composition.lower() in sol:
                                    reward += 1. / len(sol)
                                else:
                                    reward -= 1. / len(sol)
                            reward = max(reward, 0.)
                except Exception:
                    pass  # Continue to next verification method if this fails

            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true" and task == "composition_class":
                log_path = os.getenv("LOG_PATH")
                # local_rank = int(os.getenv("LOCAL_RANK", 0))
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Composition reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Extracted compositions: {compositions}\n")
                    f.write(f"Solution: {sol}\n")
        return rewards

    @staticmethod
    def reward_composition_score(completions, solution, tasks, **kwargs):
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        score_pattern = r'\d+\.\d+|\.\d+|\d+'
        for content, sol, task in zip(contents, solution, tasks):
            reward = 0.0
            if task == "composition_score":
                # Try symbolic verification first
                score = None
                try:
                    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                    if content_answer_match:
                        content_answer = content_answer_match.group(1).strip()
                        score_match = re.search(score_pattern, content_answer)
                        if score_match:
                            score = float(score_match.group(0))
                            diff = abs(score - sol)
                            reward = max((4. - diff) / 4., 0)
                except Exception:
                    pass  # Continue to next verification method if this fails

            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true" and task == "composition_score":
                log_path = os.getenv("LOG_PATH")
                # local_rank = int(os.getenv("LOCAL_RANK", 0))
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Score reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Extracted score: {score}\n")
                    f.write(f"Solution: {sol}\n")
        return rewards

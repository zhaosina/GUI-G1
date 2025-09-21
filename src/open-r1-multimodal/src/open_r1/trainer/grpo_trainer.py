# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import json
import os
import textwrap
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Union, Sized

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

from accelerate.utils import is_peft_model, set_seed
import PIL.Image
from PIL import ImageDraw

import copy
from torch.utils.data import Sampler
import warnings

from torch.nn.utils.rnn import pad_sequence

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility.
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class Qwen2VLGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        freeze_vision_modules: Optional[bool] = False,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "bfloat16",
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = torch_dtype
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "altas" in model_id.lower() or "atlas" in model_id.lower():
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        self.vision_modules_keywords = ["visual"]
        if peft_config is not None:
            def find_all_linear_names(model, multimodal_keywords):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    # LoRA is not applied to the vision modules
                    if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                        continue
                    if isinstance(module, cls):
                        lora_module_names.add(name)
                for m in lora_module_names:  # needed for 16-bit
                    if "embed_tokens" in m:
                        lora_module_names.remove(m)
                return list(lora_module_names)
            target_modules = find_all_linear_names(model, self.vision_modules_keywords)
            peft_config.target_modules = target_modules
            model = get_peft_model(model, peft_config)

        if freeze_vision_modules:
            print("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.vision_modules_keywords):
                    p.requires_grad = False

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "altas" in model_id.lower() or "atlas" in model_id.lower():
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id or "altas" in model_id or "atlas" in model_id.lower():
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_prompt_length = None
        if args.max_prompt_length is not None:
            warnings.warn("Setting max_prompt_length is currently not supported, it has been set to None")

        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            temperature=args.temperature,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta
        self.epsilon = args.epsilon

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # Multi-turn training configuration
        self.enable_multi_turn = args.enable_multi_turn
        self.multi_turn_max_turns = max(1, args.multi_turn_max_turns)
        self.multi_turn_success_threshold = args.multi_turn_success_threshold
        self.multi_turn_visible_ratio = max(0.0, min(1.0, args.multi_turn_visible_ratio))
        self.log_multi_turn_conversations = args.log_multi_turn_conversations
        if self.log_multi_turn_conversations:
            default_dir = Path(args.output_dir) / "multi_turn_logs"
            log_dir = Path(args.conversation_log_dir) if args.conversation_log_dir is not None else default_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            self.conversation_log_dir = log_dir
        else:
            self.conversation_log_dir = None

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model
    
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw):
        logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)


    def _prepare_inputs(self, inputs):
        # Simple pass-through, just like original
        return inputs

    def _prepare_image_for_prompt(self, image):
        if image is None:
            return None
        if isinstance(image, PIL.Image.Image):
            img = image.copy()
        else:
            img = PIL.Image.open(image)
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        if w < 28 or h < 28:
            if w < h:
                scale = 28 / max(w, 1)
                new_w = 28
                new_h = max(28, int(h * scale))
            else:
                scale = 28 / max(h, 1)
                new_h = 28
                new_w = max(28, int(w * scale))
            img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
        return img

    def _prepare_prompt_batch(self, examples: list[dict[str, Any]]):
        prompts = [example["prompt"] for example in examples]
        rates = [example["rate"] for example in examples]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in examples]
        images = []
        for example in examples:
            img = example.get("prompt_image", example.get("image"))
            if img is None and "image_path" in example:
                img = example["image_path"]
            images.append(self._prepare_image_for_prompt(img))
        return prompts, rates, prompts_text, images

    def _generate_single_turn_outputs(self, inputs: list[dict[str, Any]], model):
        device = self.accelerator.device
        prompts, rates, prompts_text, images = self._prepare_prompt_batch(inputs)

        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        pixel_values = prompt_inputs.get("pixel_values")
        image_grid_thw = prompt_inputs.get("image_grid_thw")

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_inputs["input_ids"] = prompt_ids
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_mask

        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(
                **prompt_inputs,
                generation_config=self.generation_config,
            )
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        with torch.no_grad():
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw
                )
                old_per_token_logps = old_per_token_logps[:, prompt_length - 1:]
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw
                    )
        if ref_per_token_logps is not None:
            ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        return {
            "prompts": prompts,
            "rates": rates,
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "rewards_per_func": rewards_per_func,
            "completions": completions,
        }

    def _strip_previous_image_reference(self, conversation: list[dict[str, Any]]):
        for message in reversed(conversation):
            content = message.get("content")
            if message.get("role") == "user" and isinstance(content, list):
                if any(part.get("type") == "image" for part in content):
                    text_segments = [part.get("text", "") for part in content if part.get("type") == "text"]
                    text = text_segments[0] if text_segments else ""
                    replacement = text if text else "[Image omitted]"
                    message["content"] = [{"type": "text", "text": f"[Image omitted] {replacement}".strip()}]
                    return text
        return ""

    def _as_text_content(self, text: str):
        return [{"type": "text", "text": text}]

    def _parse_bbox_from_completion(self, completion: str | None):
        if not completion:
            return None
        text = completion
        if "<answer>" in text and "</answer>" in text:
            text = text.split("<answer>", 1)[1].split("</answer>", 1)[0]
        if "```json" in text:
            text = text.split("```json", 1)[1]
            text = text.split("```", 1)[0]
        text = text.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                return None
        if isinstance(parsed, dict):
            candidate = parsed.get("bbox_2d") or parsed.get("bbox") or parsed.get("box")
            if candidate and len(candidate) == 4:
                return tuple(float(x) for x in candidate)
        elif isinstance(parsed, list) and parsed:
            first = parsed[0]
            if isinstance(first, dict):
                candidate = first.get("bbox_2d") or first.get("bbox") or first.get("box")
                if candidate and len(candidate) == 4:
                    return tuple(float(x) for x in candidate)
            elif isinstance(first, (list, tuple)) and len(first) == 4:
                return tuple(float(x) for x in first)
        return None

    def _create_masked_image(self, image: PIL.Image.Image | None, bbox: Any):
        if image is None or bbox is None:
            return None
        img = image.copy().convert("RGBA")
        width, height = img.size
        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        min_w = width * self.multi_turn_visible_ratio
        min_h = height * self.multi_turn_visible_ratio
        target_w = max(box_w, min_w)
        target_h = max(box_h, min_h)
        left = max(0, int(center_x - target_w / 2))
        right = min(width, int(center_x + target_w / 2))
        top = max(0, int(center_y - target_h / 2))
        bottom = min(height, int(center_y + target_h / 2))
        overlay = PIL.Image.new("RGBA", img.size, (0, 0, 0, 180))
        overlay.paste(PIL.Image.new("RGBA", (right - left, bottom - top), (0, 0, 0, 0)), (left, top))
        combined = PIL.Image.alpha_composite(img, overlay)
        draw = ImageDraw.Draw(combined)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=3)
        return combined.convert("RGB")

    def _build_error_hint_text(self, predicted_bbox, target_bbox, image_size):
        if target_bbox is None:
            return "Focus on the highlighted region and return the bbox_2d coordinates in JSON format."
        gx1, gy1, gx2, gy2 = [float(v) for v in target_bbox]
        width, height = image_size
        if predicted_bbox is None:
            return (
                "Hint: The previous response was invalid. Use the highlighted area and report bbox_2d coordinates close to "
                f"{[round(gx1, 2), round(gy1, 2), round(gx2, 2), round(gy2, 2)]}."
            )
        px1, py1, px2, py2 = predicted_bbox
        px_center = (px1 + px2) / 2
        py_center = (py1 + py2) / 2
        gx_center = (gx1 + gx2) / 2
        gy_center = (gy1 + gy2) / 2
        tol_x = width * 0.01
        tol_y = height * 0.01
        hints = []
        if px_center < gx_center - tol_x:
            hints.append("shift the box to the right")
        elif px_center > gx_center + tol_x:
            hints.append("shift the box to the left")
        if py_center < gy_center - tol_y:
            hints.append("move the box downward")
        elif py_center > gy_center + tol_y:
            hints.append("move the box upward")
        pred_w = max(1.0, px2 - px1)
        pred_h = max(1.0, py2 - py1)
        tgt_w = gx2 - gx1
        tgt_h = gy2 - gy1
        if pred_w < tgt_w * 0.9:
            hints.append("widen the box")
        elif pred_w > tgt_w * 1.1:
            hints.append("narrow the width")
        if pred_h < tgt_h * 0.9:
            hints.append("increase the height")
        elif pred_h > tgt_h * 1.1:
            hints.append("reduce the height")
        if not hints:
            hints.append("align the box with the highlighted region")
        guidance = ", ".join(hints)
        return (
            f"Hint: The previous box { [round(px1, 2), round(py1, 2), round(px2, 2), round(py2, 2)] } missed the target. "
            f"Please {guidance} and report bbox_2d close to {[round(gx1, 2), round(gy1, 2), round(gx2, 2), round(gy2, 2)]}."
        )

    def _persist_masked_image(self, image: PIL.Image.Image | None, sample_idx: int, turn_idx: int):
        if image is None or self.conversation_log_dir is None:
            return None
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        file_name = f"step{self.state.global_step:06d}_sample{sample_idx:04d}_turn{turn_idx + 1}.png"
        path = self.conversation_log_dir / file_name
        try:
            image.save(path)
        except Exception:
            return None
        return str(path)

    def _log_multi_turn_conversations(self, entries: list[dict[str, Any]]):
        if not entries or self.conversation_log_dir is None:
            return
        log_path = self.conversation_log_dir / "conversations.jsonl"
        with log_path.open("a", encoding="utf-8") as fp:
            for entry in entries:
                fp.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _generate_single_turn(self, inputs: list[dict[str, Any]], model):
        outputs = self._generate_single_turn_outputs(inputs, model)
        prompts = outputs["prompts"]
        rates = outputs["rates"]
        prompt_ids = outputs["prompt_ids"]
        prompt_mask = outputs["prompt_mask"]
        completion_ids = outputs["completion_ids"]
        completion_mask = outputs["completion_mask"]
        old_per_token_logps = outputs["old_per_token_logps"]
        ref_per_token_logps = outputs["ref_per_token_logps"]
        pixel_values = outputs["pixel_values"]
        image_grid_thw = outputs["image_grid_thw"]
        rewards_per_func_local = outputs["rewards_per_func"]

        rewards_per_func_global = self.accelerator.gather(rewards_per_func_local)
        rewards = rewards_per_func_global.sum(dim=1)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        local_advantages = advantages[process_slice]

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_metrics = self.accelerator.gather_for_metrics(rewards_per_func_local).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_metrics[i].item())

        self._metrics["reward"].append(
            self.accelerator.gather_for_metrics(rewards_per_func_local.sum(dim=1)).mean().item()
        )
        self._metrics["reward_std"].append(
            self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item()
        )

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": local_advantages,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "rates": rates,
        }

    def _generate_multi_turn(self, inputs: list[dict[str, Any]], model):
        device = self.accelerator.device
        num_samples = len(inputs)
        if num_samples == 0:
            return {
                "prompt_ids": torch.empty(0, dtype=torch.long, device=device),
                "prompt_mask": torch.empty(0, dtype=torch.long, device=device),
                "completion_ids": torch.empty(0, dtype=torch.long, device=device),
                "completion_mask": torch.empty(0, dtype=torch.long, device=device),
                "old_per_token_logps": None,
                "ref_per_token_logps": None,
                "advantages": torch.empty(0, device=device),
                "pixel_values": None,
                "image_grid_thw": None,
                "rates": [],
            }

        for example in inputs:
            if example.get("prompt_image") is None:
                example["prompt_image"] = example.get("image")

        conversation_states = [copy.deepcopy(example["prompt"]) for example in inputs]
        original_images = [example.get("image") for example in inputs]
        solutions = [example.get("solution") for example in inputs]
        conversation_logs: list[list[dict[str, Any]]] = [[] for _ in range(num_samples)]
        final_outputs: list[dict[str, Any] | None] = [None] * num_samples
        rewards_records: list[torch.Tensor | None] = [None] * num_samples
        success_records = [False] * num_samples
        remaining = list(range(num_samples))

        for turn_idx in range(self.multi_turn_max_turns):
            if not remaining:
                break
            active_examples = []
            for idx in remaining:
                example = dict(inputs[idx])
                example["prompt"] = conversation_states[idx]
                example["prompt_image"] = example.get("prompt_image", inputs[idx].get("prompt_image"))
                active_examples.append(example)

            outputs = self._generate_single_turn_outputs(active_examples, model)
            completions = outputs["completions"]
            rewards_per_func_local = outputs["rewards_per_func"]
            prompt_ids = outputs["prompt_ids"]
            prompt_mask = outputs["prompt_mask"]
            completion_ids = outputs["completion_ids"]
            completion_mask = outputs["completion_mask"]
            old_per_token_logps = outputs["old_per_token_logps"]
            ref_per_token_logps = outputs["ref_per_token_logps"]
            pixel_values = outputs["pixel_values"]
            image_grid_thw = outputs["image_grid_thw"]

            aggregated_rewards = rewards_per_func_local.sum(dim=1)
            success_mask = aggregated_rewards >= self.multi_turn_success_threshold
            if turn_idx == self.multi_turn_max_turns - 1:
                success_mask = torch.ones_like(success_mask, dtype=torch.bool, device=success_mask.device)

            new_remaining = []
            for local_idx, sample_idx in enumerate(remaining):
                completion_entry = completions[local_idx]
                completion_text = completion_entry[0]["content"] if is_conversational(inputs[0]) else completion_entry
                reward_row = rewards_per_func_local[local_idx].detach()
                aggregated_reward = aggregated_rewards[local_idx].item()
                parsed_bbox = self._parse_bbox_from_completion(completion_text)
                log_entry: dict[str, Any] = {
                    "turn": turn_idx + 1,
                    "response": completion_text,
                    "reward_per_func": reward_row.tolist(),
                    "reward": aggregated_reward,
                }

                if success_mask[local_idx]:
                    final_outputs[sample_idx] = {
                        "prompt_ids": prompt_ids[local_idx],
                        "prompt_mask": prompt_mask[local_idx],
                        "completion_ids": completion_ids[local_idx],
                        "completion_mask": completion_mask[local_idx],
                        "old_per_token_logps": None if old_per_token_logps is None else old_per_token_logps[local_idx],
                        "ref_per_token_logps": None if ref_per_token_logps is None else ref_per_token_logps[local_idx],
                        "pixel_values": None if pixel_values is None else pixel_values[local_idx],
                        "image_grid_thw": None if image_grid_thw is None else image_grid_thw[local_idx],
                    }
                    rewards_records[sample_idx] = reward_row
                    success_records[sample_idx] = True
                    if parsed_bbox is not None:
                        log_entry["predicted_bbox"] = [round(float(v), 2) for v in parsed_bbox]
                    if solutions[sample_idx] is not None:
                        log_entry["ground_truth_bbox"] = [round(float(v), 2) for v in solutions[sample_idx]]
                    log_entry["success"] = True
                    conversation_logs[sample_idx].append(log_entry)
                else:
                    previous_text = self._strip_previous_image_reference(conversation_states[sample_idx])
                    conversation_states[sample_idx].append(
                        {"role": "assistant", "content": self._as_text_content(completion_text)}
                    )
                    hint_text = self._build_error_hint_text(
                        parsed_bbox,
                        solutions[sample_idx],
                        original_images[sample_idx].size if original_images[sample_idx] is not None else (1, 1),
                    )
                    masked_image = self._create_masked_image(original_images[sample_idx], solutions[sample_idx])
                    if masked_image is None:
                        masked_image = self._prepare_image_for_prompt(inputs[sample_idx].get("prompt_image"))
                    inputs[sample_idx]["prompt_image"] = masked_image
                    conversation_states[sample_idx].append(
                        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": hint_text}]}
                    )
                    saved_path = self._persist_masked_image(masked_image, sample_idx, turn_idx)
                    if parsed_bbox is not None:
                        log_entry["predicted_bbox"] = [round(float(v), 2) for v in parsed_bbox]
                    if solutions[sample_idx] is not None:
                        log_entry["ground_truth_bbox"] = [round(float(v), 2) for v in solutions[sample_idx]]
                    log_entry.update(
                        {
                            "success": False,
                            "hint": hint_text,
                            "masked_image_path": saved_path,
                            "previous_instruction": previous_text,
                        }
                    )
                    conversation_logs[sample_idx].append(log_entry)
                    new_remaining.append(sample_idx)

            remaining = new_remaining

        for idx, record in enumerate(rewards_records):
            if record is None:
                raise ValueError("Multi-turn generation did not produce a completion for every sample.")

        pad_token_id = self.processing_class.pad_token_id
        prompt_ids = pad_sequence(
            [output["prompt_ids"] for output in final_outputs],
            batch_first=True,
            padding_value=pad_token_id,
        ).to(device)
        prompt_mask = pad_sequence(
            [output["prompt_mask"] for output in final_outputs],
            batch_first=True,
            padding_value=0,
        ).to(device)
        completion_ids = pad_sequence(
            [output["completion_ids"] for output in final_outputs],
            batch_first=True,
            padding_value=pad_token_id,
        ).to(device)
        completion_mask = pad_sequence(
            [output["completion_mask"] for output in final_outputs],
            batch_first=True,
            padding_value=0,
        ).to(device)

        if final_outputs[0]["old_per_token_logps"] is None:
            old_per_token_logps = None
        else:
            old_per_token_logps = pad_sequence(
                [output["old_per_token_logps"] for output in final_outputs],
                batch_first=True,
                padding_value=0.0,
            ).to(device)

        if final_outputs[0]["ref_per_token_logps"] is None:
            ref_per_token_logps = None
        else:
            ref_per_token_logps = pad_sequence(
                [output["ref_per_token_logps"] for output in final_outputs],
                batch_first=True,
                padding_value=0.0,
            ).to(device)

        if final_outputs[0]["pixel_values"] is None:
            pixel_values = None
        else:
            pixel_values = torch.stack([output["pixel_values"] for output in final_outputs]).to(device)

        if final_outputs[0]["image_grid_thw"] is None:
            image_grid_thw = None
        else:
            image_grid_thw = torch.stack([output["image_grid_thw"] for output in final_outputs]).to(device)

        rewards_tensor = torch.stack(rewards_records)
        rewards_global = self.accelerator.gather(rewards_tensor)
        rewards = rewards_global.sum(dim=1)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        process_slice = slice(
            self.accelerator.process_index * num_samples,
            (self.accelerator.process_index + 1) * num_samples,
        )
        local_advantages = advantages[process_slice]

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_metrics = self.accelerator.gather_for_metrics(rewards_tensor).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_metrics[i].item())

        self._metrics["reward"].append(
            self.accelerator.gather_for_metrics(rewards_tensor.sum(dim=1)).mean().item()
        )
        self._metrics["reward_std"].append(
            self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item()
        )

        if self.log_multi_turn_conversations:
            log_entries = []
            for idx, turns in enumerate(conversation_logs):
                if not turns:
                    continue
                log_entries.append(
                    {
                        "global_step": int(self.state.global_step),
                        "sample_index": idx,
                        "problem": inputs[idx].get("problem"),
                        "solution": inputs[idx].get("solution"),
                        "turns": turns,
                        "final_success": success_records[idx],
                    }
                )
            self._log_multi_turn_conversations(log_entries)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": local_advantages,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "rates": [example["rate"] for example in inputs],
        }

    def _generate_and_score_completions(self, inputs: list[dict[str, Any]], model):
        if self.enable_multi_turn:
            return self._generate_multi_turn(inputs, model)
        return self._generate_single_turn(inputs, model)


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
    
        # Check if we need to generate new completions or use buffered ones
        if self.state.global_step % self.num_iterations == 0:
            inputs = self._generate_and_score_completions(inputs, model)
            self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
        else:
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        self._step += 1

        # Get the prepared inputs
        rates = inputs["rates"]
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]
        
        # Concatenate for full sequence
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # Get the current policy's log probabilities
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, pixel_values, image_grid_thw)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_ids.size(1) - 1:]

        # Get the advantages from inputs
        advantages = inputs["advantages"]

        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its computation
        # and use per_token_logps.detach() instead
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()

        # Compute the policy ratio and clipped version
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # Add KL penalty if beta > 0
        if self.beta > 0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            per_token_loss = per_token_loss + self.beta * per_token_kl

            # Log KL divergence
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # Compute final loss
        ## FIX
        loss = ((per_token_loss * completion_mask).sum(dim=1) * torch.tensor(rates).to(per_token_loss.device) / 64).mean()
        # Log clip ratio
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    def _get_train_sampler(self) -> Sampler:
        """Returns a sampler that ensures proper data sampling for GRPO training."""
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        """Returns a sampler for evaluation."""
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )


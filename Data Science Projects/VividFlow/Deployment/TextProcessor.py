import gc
import traceback
from typing import Optional

import torch
import ftfy
import sentencepiece
from transformers import AutoModelForCausalLM, AutoTokenizer


class TextProcessor:
    """
    Semantic expansion engine using Qwen2.5-0.5B.
    Transforms user inputs into motion-rich prompts for video generation.
    """

    MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
    MAX_OUTPUT_LENGTH = 100  # Token limit to ensure ~50 words

    def __init__(self, resource_manager: Optional[object] = None):
        """
        Initialize TextProcessor with optional resource management.

        Args:
            resource_manager: Optional resource manager instance
        """
        self.resource_manager = resource_manager

        # Determine device
        if resource_manager is not None:
            self.device = resource_manager.get_device()
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.is_loaded = False

    def load_model(self) -> None:
        """Load Qwen model and tokenizer."""
        if self.is_loaded:
            print("⚠ TextProcessor already loaded, skipping...")
            return

        try:
            print("→ Loading Qwen2.5-0.5B-Instruct...")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_ID,
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )

            if self.resource_manager is not None:
                self.resource_manager.register_model("TextProcessor", self.model)

            self.is_loaded = True
            print("✓ TextProcessor loaded successfully")

        except Exception as e:
            print(f"✗ Error loading TextProcessor: {str(e)}")
            raise

    def unload_model(self) -> None:
        """Unload model and free GPU memory."""
        if not self.is_loaded:
            return

        try:
            if self.model is not None:
                self.model.to('cpu')
                del self.model
                self.model = None

            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            if self.resource_manager is not None:
                self.resource_manager.unregister_model("TextProcessor")
                self.resource_manager.clear_cache(aggressive=True)
            else:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self.is_loaded = False
            print("✓ TextProcessor unloaded")

        except Exception as e:
            print(f"⚠ Error during TextProcessor unload: {str(e)}")

    def expand_prompt(self, user_input: str) -> str:
        """
        Convert user's brief instruction into detailed motion description.

        Args:
            user_input: User's original instruction

        Returns:
            str: Expanded prompt for video generation (≤50 words)
        """
        if not self.is_loaded:
            raise RuntimeError("TextProcessor not loaded. Call load_model() first.")

        system_prompt = """You are a motion description expert. Convert the user's brief instruction into a detailed, dynamic prompt for video generation.

Focus on:
- Camera movements (pan, zoom, tilt, tracking)
- Subject actions and motions
- Scene dynamics and atmosphere
- Temporal flow and transitions

Keep output under 50 words. Use vivid, cinematic language. English only."""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.MAX_OUTPUT_LENGTH,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1
                )

            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            expanded_prompt = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()

            # Enforce word limit
            words = expanded_prompt.split()
            if len(words) > 50:
                expanded_prompt = " ".join(words[:50]) + "..."

            print(f"✓ Prompt expanded: '{user_input}' → '{expanded_prompt}'")
            return expanded_prompt

        except Exception as e:
            print(f"✗ Error during prompt expansion: {str(e)}")
            return user_input

    def process(self, user_input: str, auto_unload: bool = True) -> str:
        """
        Main processing pipeline: load → expand → (optionally unload).

        Args:
            user_input: User's instruction
            auto_unload: Whether to unload model after processing

        Returns:
            str: Expanded prompt
        """
        try:
            if not self.is_loaded:
                self.load_model()

            expanded = self.expand_prompt(user_input)

            if auto_unload:
                self.unload_model()

            return expanded

        except Exception as e:
            print(f"✗ TextProcessor pipeline error: {str(e)}")
            return user_input

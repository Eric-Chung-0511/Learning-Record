import os
import torch
import logging
from typing import Dict, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login

class ModelLoadingError(Exception):
    """Custom exception for model loading failures"""
    pass


class ModelGenerationError(Exception):
    """Custom exception for model generation failures"""
    pass


class LLMModelManager:
    """
    負責LLM模型的載入、設備管理和文本生成。
    管理模型、記憶體優化和設備配置。
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 tokenizer_path: Optional[str] = None,
                 device: Optional[str] = None,
                 max_length: int = 2048,
                 temperature: float = 0.3,
                 top_p: float = 0.85):
        """
        初始化模型管理器

        Args:
            model_path: LLM模型的路徑或HuggingFace模型名稱，默認使用Llama 3.2
            tokenizer_path: tokenizer的路徑，通常與model_path相同
            device: 運行設備 ('cpu'或'cuda')，None時自動檢測
            max_length: 輸入文本的最大長度
            temperature: 生成文本的溫度參數
            top_p: 生成文本時的核心採樣機率閾值
        """
        # 設置專屬logger
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # 模型配置
        self.model_path = model_path or "meta-llama/Llama-3.2-3B-Instruct"
        self.tokenizer_path = tokenizer_path or self.model_path

        # 設備管理
        self.device = self._detect_device(device)
        self.logger.info(f"Device selected: {self.device}")

        # 生成參數
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p

        # 模型狀態
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
        self.call_count = 0

        # HuggingFace認證
        self.hf_token = self._setup_huggingface_auth()

    def _detect_device(self, device: Optional[str]) -> str:
        """
        檢測並設置運行設備

        Args:
            device: 用戶指定的設備，None時自動檢測

        Returns:
            str: ('cuda' or 'cpu')
        """
        if device:
            if device == 'cuda' and not torch.cuda.is_available():
                self.logger.warning("CUDA requested but not available, falling back to CPU")
                return 'cpu'
            return device

        detected_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if detected_device == 'cuda':
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.logger.info(f"CUDA detected with {gpu_memory:.2f} GB GPU memory")

        return detected_device

    def _setup_huggingface_auth(self) -> Optional[str]:
        """
        設置HuggingFace認證

        Returns:
            Optional[str]: HuggingFace token，如果可用
        """
        hf_token = os.environ.get("HF_TOKEN")

        if hf_token:
            try:
                login(token=hf_token)
                self.logger.info("Successfully authenticated with HuggingFace")
                return hf_token
            except Exception as e:
                self.logger.error(f"HuggingFace authentication failed: {e}")
                return None
        else:
            self.logger.warning("HF_TOKEN not found. Access to gated models may be limited")
            return None

    def _load_model(self):
        """
        載入LLM模型和tokenizer，使用8位量化以節省記憶體

        Raises:
            ModelLoadingError: 當模型載入失敗時
        """
        if self._model_loaded:
            return

        try:
            self.logger.info(f"Loading model from {self.model_path} with 8-bit quantization")

            # 清理GPU記憶體
            self._clear_gpu_cache()

            # 設置8位量化配置
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )

            # 載入tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                padding_side="left",
                use_fast=False,
                token=self.hf_token
            )

            # 設置特殊標記
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 載入模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                token=self.hf_token
            )

            self._model_loaded = True
            self.logger.info("Model loaded successfully")

        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            self.logger.error(error_msg)
            raise ModelLoadingError(error_msg) from e

    def _clear_gpu_cache(self):
        """清理GPU記憶體緩存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.debug("GPU cache cleared")

    def generate_response(self, prompt: str, **generation_kwargs) -> str:
        """
        生成LLM回應

        Args:
            prompt: 輸入提示詞
            **generation_kwargs: 額外的生成參數，可覆蓋預設值

        Returns:
            str: 生成的回應文本

        Raises:
            ModelGenerationError: 當生成失敗時
        """
        # 確保模型已載入
        if not self._model_loaded:
            self._load_model()

        try:
            self.call_count += 1
            self.logger.info(f"Generating response (call #{self.call_count})")

            # clean GPU
            self._clear_gpu_cache()

            # 設置固定種子以提高一致性
            torch.manual_seed(42)

            # prepare input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            ).to(self.device)

            # 準備生成參數
            generation_params = self._prepare_generation_params(**generation_kwargs)
            generation_params.update({
                "pad_token_id": self.tokenizer.eos_token_id,
                "attention_mask": inputs.attention_mask,
                "use_cache": True,
            })

            # resposne
            with torch.no_grad():
                outputs = self.model.generate(inputs.input_ids, **generation_params)

            # 解碼回應
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = self._extract_generated_response(full_response, prompt)

            if not response or len(response.strip()) < 10:
                raise ModelGenerationError("Generated response is too short or empty")

            self.logger.info(f"Response generated successfully ({len(response)} characters)")
            return response

        except Exception as e:
            error_msg = f"Text generation failed: {str(e)}"
            self.logger.error(error_msg)
            raise ModelGenerationError(error_msg) from e

    def _prepare_generation_params(self, **kwargs) -> Dict[str, Any]:
        """
        準備生成參數，支援模型特定的優化

        Args:
            **kwargs: 用戶提供的生成參數

        Returns:
            Dict[str, Any]: 完整的生成參數配置
        """
        # basic parameters
        params = {
            "max_new_tokens": 120,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": True,
        }

        # 針對Llama模型的特殊優化
        if "llama" in self.model_path.lower():
            params.update({
                "max_new_tokens": 600,
                "temperature": 0.35, # not too big
                "top_p": 0.75,
                "repetition_penalty": 1.5,
                "num_beams": 5,
                "length_penalty": 1,
                "no_repeat_ngram_size": 3
            })
        else:
            params.update({
                "max_new_tokens": 300,
                "temperature": 0.6,
                "top_p": 0.9,
                "num_beams": 1,
                "repetition_penalty": 1.05
            })

        # 用戶參數覆蓋預設值
        params.update(kwargs)

        return params

    def _extract_generated_response(self, full_response: str, prompt: str) -> str:
        """
        從完整回應中提取生成的部分

        Args:
            full_response: 模型的完整輸出
            prompt: 原始提示詞

        Returns:
            str: 提取的生成回應
        """
        # 尋找assistant標記
        assistant_tag = "<|assistant|>"
        if assistant_tag in full_response:
            response = full_response.split(assistant_tag)[-1].strip()

            # 檢查是否有未閉合的user標記
            user_tag = "<|user|>"
            if user_tag in response:
                response = response.split(user_tag)[0].strip()

            return response

        # 移除輸入提示詞
        if full_response.startswith(prompt):
            return full_response[len(prompt):].strip()

        return full_response.strip()

    def reset_context(self):
        """重置模型上下文，清理GPU緩存"""
        if self._model_loaded:
            self._clear_gpu_cache()
            self.logger.info("Model context reset")
        else:
            self.logger.info("Model not loaded, no context to reset")

    def get_current_device(self) -> str:
        """
        獲取當前運行設備

        Returns:
            str: 當前設備名稱
        """
        return self.device

    def is_model_loaded(self) -> bool:
        """
        檢查模型是否已載入

        Returns:
            bool: 模型載入狀態
        """
        return self._model_loaded

    def get_call_count(self) -> int:
        """
        獲取模型調用次數

        Returns:
            int: 調用次數
        """
        return self.call_count

    def get_model_info(self) -> Dict[str, Any]:
        """
        獲取模型信息

        Returns:
            Dict[str, Any]: 包含模型路徑、設備、載入狀態等信息
        """
        return {
            "model_path": self.model_path,
            "device": self.device,
            "is_loaded": self._model_loaded,
            "call_count": self.call_count,
            "has_hf_token": self.hf_token is not None
        }

"""
Qwen Image 2.1 Prompt Enhancer Nodes

核心节点实现:
1. QwenImage21PromptEnhancer - 主增强节点
2. QwenImage21PromptEnhancerConfig - 配置节点 (可选)
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image

import comfy.model_management
from comfy_api.latest import io

from .enhancer import (
    EnhanceMode,
    PromptEnhancer,
    ImageWithSize,
    parse_enhanced_result,
    load_system_prompt_t2i,
    load_system_prompt_edit,
)

# ============== 自定义类型定义 ==============
EnhancedPromptResult = io.Custom("ENHANCED_PROMPT_RESULT")


# ============== 主增强节点 ==============
class QwenImage21PromptEnhancer:
    """Qwen Image 2.1 提示词增强节点"""
    
    CATEGORY = "QwenImage21"
    EXPERIMENTAL = False
    DEPRECATED = False
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (io.String.TYPE, {
                    "multiline": True,
                    "dynamic_prompts": True,
                    "default": "",
                    "display_name": "Prompt",
                    "tooltip": "原始提示词，支持中英文"
                }),
                "mode": (["t2i", "i2i"], {
                    "default": "t2i",
                    "display_name": "Mode",
                    "tooltip": "t2i=文生图, i2i=图生图"
                }),
                "language": (["en", "zh"], {
                    "default": "en",
                    "display_name": "Output Language",
                    "tooltip": "输出语言"
                }),
            },
            "optional": {
                "images": (io.Image.TYPE, {
                    "optional": True,
                    "display_name": "Reference Images",
                    "tooltip": f"参考图片，最多16张 (Qwen Image 2.1 最大支持)"
                }),
                "director_config": ("HR_DIRECTOR_CONFIG", {
                    "optional": True,
                    "display_name": "HR Director Config",
                    "tooltip": "HR Qwen Director Config 节点配置"
                }),
                "pe_model_path": (io.String.TYPE, {
                    "optional": True,
                    "default": "",
                    "display_name": "PE Model Path",
                    "tooltip": "本地 PE 模型路径，留空使用默认路径"
                }),
                "max_new_tokens": (io.Int.TYPE, {
                    "optional": True,
                    "default": 4096,
                    "min": 256,
                    "max": 16384,
                    "step": 256,
                    "display_name": "Max New Tokens"
                }),
                "temperature": (io.Float.TYPE, {
                    "optional": True,
                    "default": 1.0,
                    "min": 0.01,
                    "max": 2.0,
                    "step": 0.01,
                    "display_name": "Temperature"
                }),
                "top_p": (io.Float.TYPE, {
                    "optional": True,
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display_name": "Top P"
                }),
                "top_k": (io.Int.TYPE, {
                    "optional": True,
                    "default": 20,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                    "display_name": "Top K"
                }),
                "presence_penalty": (io.Float.TYPE, {
                    "optional": True,
                    "default": 0.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "display_name": "Presence Penalty"
                }),
                "seed": (io.Int.TYPE, {
                    "optional": True,
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "display_name": "Seed"
                }),
                "thinking": (io.Boolean.TYPE, {
                    "optional": True,
                    "default": False,
                    "display_name": "Enable Thinking",
                    "tooltip": "启用思考模式 (仅 PE 模型支持)"
                }),
                "unload_after": (io.Boolean.TYPE, {
                    "optional": True,
                    "default": True,
                    "display_name": "Unload Model After",
                    "tooltip": "生成后卸载模型释放显存"
                }),
            },
        }

    RETURN_TYPES = (
        io.String.TYPE,  # enhanced_prompt
        io.String.TYPE,  # thinking
        io.String.TYPE,  # wh_ratio
        io.String.TYPE,  # ratio_follow
        EnhancedPromptResult.TYPE,  # full_result
    )
    RETURN_NAMES = (
        "enhanced_prompt",
        "thinking",
        "wh_ratio",
        "ratio_follow",
        "full_result",
    )
    FUNCTION = "enhance"
    
    def __init__(self):
        self.enhancer: Optional[PromptEnhancer] = None
        self._last_mode = None
        self._last_pe_model_path = None
    
    def enhance(
        self,
        prompt: str,
        mode: str,
        language: str,
        images: Optional[torch.Tensor] = None,
        director_config: Optional[dict] = None,
        pe_model_path: str = "",
        max_new_tokens: int = 4096,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 20,
        presence_penalty: float = 0.0,
        seed: int = 42,
        thinking: bool = False,
        unload_after: bool = True,
    ) -> tuple:
        """执行提示词增强"""
        
        # 空提示词检查
        if not prompt.strip():
            empty_result = {
                "enhanced_prompt": "",
                "thinking": "",
                "wh_ratio": "",
                "ratio_follow": "",
                "parse_ok": True
            }
            return ("", "", "", "", json.dumps(empty_result))
        
        # 解析参考图片
        image_list: list[ImageWithSize] = []
        if images is not None:
            image_list = self._process_images(images)
            logging.info(f"Processing {len(image_list)} reference images")
        
        # 图生图模式必须要有图片
        if mode == "i2i" and len(image_list) == 0:
            logging.warning("i2i mode requires reference images, falling back to t2i")
            mode = "t2i"
        
        # 限制图片数量 (Qwen Image 2.1 最大支持 16 张)
        max_images = 16
        if len(image_list) > max_images:
            logging.warning(f"Too many images ({len(image_list)}), truncating to {max_images}")
            image_list = image_list[:max_images]
        
        # 初始化或复用 enhancer
        pe_path = pe_model_path.strip() if pe_model_path else None
        
        # 构建配置
        config = None
        if director_config:
            try:
                from ComfyUI_MiniMax_H3_Sampler_Unlimited.director_config import normalize_qwen38_config
                config = normalize_qwen38_config(director_config)
            except ImportError:
                pass
        
        if (self.enhancer is None or 
            self._last_mode != mode or 
            self._last_pe_model_path != pe_path):
            self.enhancer = PromptEnhancer(
                mode=EnhanceMode.T2I if mode == "t2i" else EnhanceMode.I2I,
                pe_model_path=pe_path,
                director_config=config,
            )
            self._last_mode = mode
            self._last_pe_model_path = pe_path
        
        # 构建系统提示词
        if mode == "t2i":
            system_prompt = load_system_prompt_t2i()
        else:
            system_prompt = load_system_prompt_edit()
        
        # 执行增强
        start_time = time.time()
        try:
            raw_result = self.enhancer.enhance(
                prompt=prompt,
                images=image_list,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                seed=seed,
                thinking=thinking,
            )
            elapsed = time.time() - start_time
            logging.info(f"Prompt enhancement completed in {elapsed:.2f}s")
            
        except Exception as e:
            logging.error(f"Prompt enhancement failed: {e}")
            raise
        
        finally:
            if unload_after and self.enhancer is not None:
                self.enhancer.unload()
        
        # 解析结果
        enhanced_prompt, thinking_text, wh_ratio, ratio_follow, parse_ok = parse_enhanced_result(
            raw_result, mode == "i2i"
        )
        
        # 构建完整结果
        full_result = {
            "enhanced_prompt": enhanced_prompt,
            "thinking": thinking_text,
            "wh_ratio": wh_ratio,
            "ratio_follow": ratio_follow,
            "parse_ok": parse_ok,
            "mode": mode,
            "image_count": len(image_list),
            "elapsed_seconds": round(time.time() - start_time, 2),
        }
        
        return (
            enhanced_prompt,
            thinking_text,
            wh_ratio,
            ratio_follow,
            json.dumps(full_result),
        )
    
    def _process_images(self, images: torch.Tensor) -> list[ImageWithSize]:
        """将 ComfyUI 图像张量转换为 PIL Image 列表"""
        result = []
        
        # images shape: [B, H, W, C] 或 [B, C, H, W]
        if images.ndim == 4:
            if images.shape[-1] in (1, 3, 4):  # [B, H, W, C]
                for i in range(images.shape[0]):
                    img_tensor = images[i]
                    if img_tensor.shape[-1] == 1:
                        img_tensor = img_tensor.repeat(1, 1, 3)
                    elif img_tensor.shape[-1] == 4:  # RGBA -> RGB
                        img_tensor = img_tensor[..., :3]
                    # 转换到 [0, 255] 范围
                    img_np = (img_tensor.cpu().numpy() * 255).astype('uint8')
                    pil_img = Image.fromarray(img_np)
                    result.append(ImageWithSize(image=pil_img, original_size=pil_img.size))
            else:  # [B, C, H, W]
                for i in range(images.shape[0]):
                    img_tensor = images[i]
                    if img_tensor.shape[0] == 1:
                        img_tensor = img_tensor.repeat(3, 1, 1)
                    elif img_tensor.shape[0] == 4:
                        img_tensor = img_tensor[:3]
                    img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                    pil_img = Image.fromarray(img_np)
                    result.append(ImageWithSize(image=pil_img, original_size=pil_img.size))
        
        return result


# ============== 简单模式节点 ==============
class QwenImage21PromptEnhancerSimple:
    """简化版提示词增强节点 - 只需输入提示词"""
    
    CATEGORY = "QwenImage21"
    EXPERIMENTAL = False
    DEPRECATED = False
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (io.String.TYPE, {
                    "multiline": True,
                    "dynamic_prompts": True,
                    "default": "",
                    "display_name": "Prompt",
                }),
            },
            "optional": {
                "images": (io.Image.TYPE, {
                    "optional": True,
                    "display_name": "Reference Images",
                }),
            },
        }

    RETURN_TYPES = (io.String.TYPE, io.String.TYPE, io.String.TYPE)
    RETURN_NAMES = ("enhanced_prompt", "wh_ratio", "thinking")
    FUNCTION = "enhance"
    
    def __init__(self):
        self.node = QwenImage21PromptEnhancer()
    
    def enhance(
        self,
        prompt: str,
        images: Optional[torch.Tensor] = None,
    ) -> tuple:
        """简化版增强 - 使用默认参数"""
        enhanced, thinking, wh_ratio, _, _ = self.node.enhance(
            prompt=prompt,
            mode="i2i" if images is not None else "t2i",
            language="en",
            images=images,
        )
        return (enhanced, wh_ratio, thinking)


# ============== 翻译节点 ==============
class QwenImage21Translator:
    """中英互译节点 - 使用 HRQwen Director 模型"""
    
    CATEGORY = "QwenImage21"
    EXPERIMENTAL = False
    DEPRECATED = False
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (io.String.TYPE, {
                    "multiline": True,
                    "dynamic_prompts": True,
                    "default": "",
                    "display_name": "Text",
                    "tooltip": "待翻译文本"
                }),
                "direction": (["auto", "zh2en", "en2zh"], {
                    "default": "auto",
                    "display_name": "Direction",
                    "tooltip": "翻译方向: auto=自动检测, zh2en=中文译英文, en2zh=英文译中文"
                }),
            },
            "optional": {
                "director_config": ("HR_DIRECTOR_CONFIG", {
                    "optional": True,
                    "display_name": "Director Config",
                    "tooltip": "HR Qwen Director Config，使用配置的模型进行翻译"
                }),
            }
        }

    RETURN_TYPES = (io.String.TYPE, io.String.TYPE)
    RETURN_NAMES = ("translated_text", "detected_lang")
    FUNCTION = "translate"
    
    def __init__(self):
        self._director = None
        self._last_config = None
    
    def translate(
        self,
        text: str,
        direction: str,
        director_config: Optional[dict] = None,
    ) -> tuple:
        """执行翻译"""
        
        if not text.strip():
            return ("", "")
        
        # 自动检测语言
        if direction == "auto":
            detected = self._detect_language(text)
            if detected == "zh":
                direction = "zh2en"
            else:
                direction = "en2zh"
        else:
            detected = "zh" if direction == "zh2en" else "en"
        
        # 构建翻译 prompt
        if direction == "zh2en":
            translate_prompt = f"""Translate the following Chinese text to English.
Only output the translated text, nothing else.

Chinese text:
{text}"""
        else:
            translate_prompt = f"""Translate the following English text to Chinese.
Only output the translated text, nothing else.

English text:
{text}"""
        
        # 使用 Director 模型翻译
        if director_config:
            result = self._translate_with_director(translate_prompt, director_config)
        else:
            result = text  # 没有配置时返回原文
        
        return (result.strip(), detected)
    
    def _detect_language(self, text: str) -> str:
        """简单语言检测"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(re.findall(r'[\w\u4e00-\u9fff]', text))
        
        if total_chars == 0:
            return "en"
        
        chinese_ratio = chinese_chars / total_chars
        return "zh" if chinese_ratio > 0.3 else "en"
    
    def _translate_with_director(self, prompt: str, director_config: dict) -> str:
        """使用 Director 模型翻译"""
        try:
            from ComfyUI_MiniMax_H3_Sampler_Unlimited.director_config import normalize_qwen38_config
            from ComfyUI_MiniMax_H3_Sampler_Unlimited.director_backend import resolve_director_selection
            from ComfyUI_MiniMax_H3_Sampler_Unlimited.qwen35 import Qwen35ContinuityDirector
            
            # 检查是否复用同一个 Director
            config_str = str(director_config)
            if self._director is None or self._last_config != config_str:
                config = normalize_qwen38_config(director_config)
                selection = resolve_director_selection(
                    config["backend"],
                    config["model"],
                    config["mmproj"]
                )
                
                if selection.model_path is None or selection.mmproj_path is None:
                    raise ValueError("Director requires local Qwen model and mmproj")
                
                self._director = Qwen35ContinuityDirector(
                    model_path=selection.model_path,
                    mmproj_path=selection.mmproj_path,
                    backend=config["backend"],
                    mtp_enabled=config["mtp"],
                    mtp_draft_tokens=config["mtp_draft_tokens"],
                    reasoning_effort=config["reasoning_effort"],
                    debug=config["debug"],
                )
                self._last_config = config_str
            
            # 调用 Director 生成
            result = self._director.observe_and_plan(
                story=prompt,
                pictures=None,
                style="translation",
                prompt_lang="auto",
            )
            
            return result.get("detailed_description", prompt)
            
        except ImportError as e:
            logging.error(f"Cannot import H3 Sampler Unlimited: {e}")
            return f"[HRQwen38DirectorConfig not available: {e}]"
        except Exception as e:
            logging.error(f"Director translation failed: {e}")
            return f"[Translation Error: {e}]"


# ============== 节点映射 ==============
NODE_CLASS_MAPPINGS = {
    "QwenImage21PromptEnhancer": QwenImage21PromptEnhancer,
    "QwenImage21PromptEnhancerSimple": QwenImage21PromptEnhancerSimple,
    "QwenImage21Translator": QwenImage21Translator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImage21PromptEnhancer": "Qwen Image 2.1 Prompt Enhancer",
    "QwenImage21PromptEnhancerSimple": "Qwen Image 2.1 Prompt Enhancer (Simple)",
    "QwenImage21Translator": "Qwen Image 2.1 Translator",
}

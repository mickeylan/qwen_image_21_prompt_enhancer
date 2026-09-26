"""
Qwen Image 2.1 Prompt Enhancer Core

核心增强引擎，支持三种模式:
1. PE模式 - 使用 Qwen 官方 PE 模型 (Qwen3.5-VL 9B)
2. Director模式 - 使用 HRQwen38DirectorConfig 配置的 Qwen 模型
3. API模式 - 调用远程 API
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image

# ============== 路径配置 ==============
DEFAULT_PE_MODEL_PATHS = {
    "t2i": "models/text_encoders/Qwen-Image-2.1-PE-T2I",
    "i2i": "models/text_encoders/Qwen-Image-2.1-PE-I2I",
}

PROMPT_RULES_PATH = Path(__file__).parent.parent / "prompt_rewrite" / "prompts"


# ============== 枚举定义 ==============
class EnhanceMode(str, Enum):
    T2I = "t2i"
    I2I = "i2i"


@dataclass
class ImageWithSize:
    """带原始尺寸的图像"""
    image: Image.Image
    original_size: tuple[int, int]


# ============== 核心增强器 ==============
class PromptEnhancer:
    """
    Qwen Image 2.1 提示词增强器
    """

    def __init__(
        self,
        mode: EnhanceMode = EnhanceMode.T2I,
        pe_model_path: Optional[str] = None,
        director_config: Optional[dict] = None,
        api_config: Optional[dict] = None,
    ):
        self.mode = mode
        self.pe_model_path = pe_model_path
        self.director_config = director_config
        self.api_config = api_config
        self._process = None
        self._server_url = None

        self._detect_runtime()

    def _detect_runtime(self):
        """检测可用的运行环境"""
        self._runtime_mode = None

        # 1. 优先检查 PE 模型
        if self.pe_model_path:
            pe_dir = Path(self.pe_model_path)
        else:
            default_key = "t2i" if self.mode == EnhanceMode.T2I else "i2i"
            pe_dir = Path("K:/ComfyUI") / DEFAULT_PE_MODEL_PATHS[default_key]
            if not pe_dir.exists():
                pe_dir = Path("K:/ComfyUI/models/text_encoders")

        if pe_dir.exists() and (pe_dir / "config.json").exists():
            self._runtime_mode = "pe"
            self._pe_model_dir = pe_dir
            logging.info(f"Using PE model at: {pe_dir}")
            return

        # 2. 检查 Director Config
        if self.director_config:
            self._runtime_mode = "director"
            logging.info("Using HRQwen Director Config")
            return

        # 3. 检查 API 配置
        if self.api_config and self.api_config.get("api_base"):
            self._runtime_mode = "api"
            logging.info("Using API")
            return

        # 4. 检查 vLLM 服务
        if self._check_vllm_service():
            self._runtime_mode = "vllm"
            return

        self._runtime_mode = "fallback"
        logging.warning("No runtime available, using fallback")

    def _check_vllm_service(self) -> bool:
        """检查 vLLM 服务是否可用"""
        try:
            import requests
            response = requests.get("http://localhost:8100/health", timeout=2)
            if response.status_code == 200:
                self._server_url = "http://localhost:8100"
                return True
        except:
            pass
        return False

    def enhance(
        self,
        prompt: str,
        images: list[ImageWithSize] | None = None,
        system_prompt: str | None = None,
        **sampling_params,
    ) -> str:
        """执行提示词增强"""
        if self._runtime_mode == "pe":
            return self._enhance_pe(prompt, images, system_prompt, **sampling_params)
        elif self._runtime_mode == "director":
            return self._enhance_director(prompt, images, system_prompt, **sampling_params)
        elif self._runtime_mode == "api":
            return self._enhance_api(prompt, images, system_prompt, **sampling_params)
        elif self._runtime_mode == "vllm":
            return self._enhance_vllm(prompt, images, system_prompt, **sampling_params)
        else:
            return self._enhance_fallback(prompt, images, system_prompt, **sampling_params)

    def _enhance_pe(
        self,
        prompt: str,
        images: list[ImageWithSize] | None,
        system_prompt: str | None,
        **sampling_params,
    ) -> str:
        """使用 PE 模型增强"""
        # 构建输入
        case = {
            "id": "comfyui_single",
            "prompt": prompt,
            "input_images": [],
            "task_type": "comfyui",
        }

        temp_images = []
        if images:
            for i, img_with_size in enumerate(images):
                temp_dir = Path(tempfile.gettempdir()) / "qwen_pe_images"
                temp_dir.mkdir(exist_ok=True)
                temp_path = temp_dir / f"image_{i}.png"
                img_with_size.image.save(temp_path)
                temp_images.append(str(temp_path))
            case["input_images"] = temp_images

        task = "edit" if temp_images else "t2i"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            json.dump(case, f, ensure_ascii=False)
            input_path = f.name

        output_path = input_path.replace('.jsonl', '_out.jsonl')

        try:
            cmd = [
                sys.executable,
                str(Path(__file__).parent.parent / "prompt_rewrite" / "run_vllm.py"),
                "--task", task,
                "--ckpt", str(self._pe_model_dir),
                "--input", input_path,
                "--output", output_path,
            ]

            if system_prompt:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                    f.write(system_prompt)
                    cmd.extend(["--system-prompt", f.name])

            for key, value in [('temperature', '--temperature'), ('top_p', '--top-p'),
                               ('top_k', '--top-k'), ('seed', '--seed'),
                               ('max_new_tokens', '--max-new-tokens')]:
                if sampling_params.get(key):
                    cmd.extend([value, str(sampling_params[key])])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                logging.error(f"PE model error: {result.stderr}")
                raise RuntimeError(f"PE model failed: {result.stderr}")

            with open(output_path, 'r', encoding='utf-8') as f:
                output = json.loads(f.readline())

            thinking = output.get('thinking', '')
            positive_prompt = output.get('positive_prompt', '')
            return f"<think>\n{thinking}\n</think>\n{positive_prompt}"

        finally:
            for path in [input_path, output_path]:
                try:
                    Path(path).unlink()
                except:
                    pass
            for path in temp_images:
                try:
                    Path(path).unlink()
                except:
                    pass

    def _enhance_director(
        self,
        prompt: str,
        images: list[ImageWithSize] | None,
        system_prompt: str | None,
        **sampling_params,
    ) -> str:
        """使用 HRQwen Director Config 增强"""
        try:
            # 导入 H3 Sampler Unlimited 的模块
            from ComfyUI_MiniMax_H3_Sampler_Unlimited.director_config import normalize_qwen38_config
            from ComfyUI_MiniMax_H3_Sampler_Unlimited.director_backend import resolve_director_selection
            from ComfyUI_MiniMax_H3_Sampler_Unlimited.qwen35 import Qwen35ContinuityDirector

            config = normalize_qwen38_config(self.director_config)
            selection = resolve_director_selection(
                config["backend"],
                config["model"],
                config["mmproj"]
            )

            if selection.model_path is None or selection.mmproj_path is None:
                raise ValueError("Director requires local Qwen model and mmproj")

            # 创建 Director
            director = Qwen35ContinuityDirector(
                model_path=selection.model_path,
                mmproj_path=selection.mmproj_path,
                backend=config["backend"],
                mtp_enabled=config["mtp"],
                mtp_draft_tokens=config["mtp_draft_tokens"],
                reasoning_effort=config["reasoning_effort"],
                debug=config["debug"],
            )

            # 构建图片输入
            pil_images = [img.image for img in images] if images else None

            # 调用 Director 生成增强提示词
            result = director.observe_and_plan(
                story=prompt,
                pictures=pil_images,
                style="photorealistic",
                prompt_lang="auto",
            )

            return result.get("detailed_description", prompt)

        except ImportError as e:
            logging.error(f"Cannot import H3 Sampler Unlimited: {e}")
            raise RuntimeError("HRQwen38DirectorConfig not available")
        except Exception as e:
            logging.error(f"Director enhancement failed: {e}")
            raise

    def _enhance_api(
        self,
        prompt: str,
        images: list[ImageWithSize] | None,
        system_prompt: str | None,
        **sampling_params,
    ) -> str:
        """使用 API 增强"""
        import openai

        client = openai.OpenAI(
            api_base=self.api_config["api_base"],
            api_key=self.api_config["api_key"]
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = []
        if images:
            for img_with_size in images:
                img_bytes = io.BytesIO()
                img_with_size.image.save(img_bytes, format='PNG')
                img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                })

        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})

        response = client.chat.completions.create(
            model=self.api_config.get("model", "gpt-4o"),
            messages=messages,
            temperature=sampling_params.get('temperature', 1.0),
            max_tokens=sampling_params.get('max_new_tokens', 4096),
        )

        return response.choices[0].message.content

    def _enhance_vllm(
        self,
        prompt: str,
        images: list[ImageWithSize] | None,
        system_prompt: str | None,
        **sampling_params,
    ) -> str:
        """使用 vLLM 服务增强"""
        import requests

        url = f"{self._server_url}/v1/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = []
        if images:
            for img_with_size in images:
                img_bytes = io.BytesIO()
                img_with_size.image.save(img_bytes, format='PNG')
                img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                })

        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})

        response = requests.post(url, json={
            "model": "Qwen-Image-2.1-PE",
            "messages": messages,
            "temperature": sampling_params.get('temperature', 1.0),
            "max_tokens": sampling_params.get('max_new_tokens', 4096),
        })

        return response.json()['choices'][0]['message']['content']

    def _enhance_fallback(
        self,
        prompt: str,
        images: list[ImageWithSize] | None,
        system_prompt: str | None,
        **sampling_params,
    ) -> str:
        """回退模式：使用内置提示词返回格式化请求"""
        from .system_prompts import get_t2i_system_prompt, get_i2i_system_prompt

        if system_prompt is None:
            system_prompt = get_t2i_system_prompt() if self.mode == EnhanceMode.T2I else get_i2i_system_prompt()

        # 返回带有结构的文本，供外部处理
        result = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "mode": self.mode.value,
            "image_count": len(images) if images else 0,
        }

        return json.dumps(result)

    def unload(self):
        """卸载模型，释放显存"""
        if self._process:
            self._process.terminate()
            self._process = None
        torch.cuda.empty_cache()


# ============== 结果解析 ==============
def split_thinking(text: str) -> tuple[str, str]:
    """分割思考和答案"""
    if "</think>" in text:
        think, _, answer = text.partition("</think>")
        if "<think>" in think:
            think = think.partition("<think>")[2]
        return think.strip(), answer.strip()
    if "<think>" in text:
        return text.partition("<think>")[2].strip(), ""
    return "", text.strip()


def _balanced_spans(answer: str) -> list[str]:
    """提取所有平衡的 JSON 对象"""
    spans = []
    depth = 0
    start = -1
    in_str = False
    escaped = False

    for i, ch in enumerate(answer):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(answer[start:i + 1])

    return spans


def parse_enhanced_result(raw_result: str, has_ratio_follow: bool = False) -> tuple[str, str, str, str, bool]:
    """解析增强结果"""
    thinking, answer = split_thinking(raw_result)

    # 尝试解析 JSON
    for candidate in reversed(_balanced_spans(answer)):
        try:
            obj = json.loads(candidate)
            rewritten = obj.get("rewritten_prompt") or obj.get("rewrited_prompt")
            if not isinstance(rewritten, str) or not rewritten.strip():
                continue

            wh_ratio = str(obj.get("wh_ratio") or "").strip()
            ratio_follow = str(obj.get("ratio_follow") or "").strip() if has_ratio_follow else ""

            return rewritten.strip(), thinking, wh_ratio, ratio_follow, True
        except json.JSONDecodeError:
            continue

    return answer.strip(), thinking, "", "", False


def load_system_prompt_t2i() -> str:
    """加载 T2I 系统提示词"""
    path = PROMPT_RULES_PATH / "system_prompt_t2i.txt"
    if path.exists():
        return path.read_text(encoding='utf-8').strip()

    from .system_prompts import T2I_SYSTEM_PROMPT
    return T2I_SYSTEM_PROMPT


def load_system_prompt_edit() -> str:
    """加载 I2I 系统提示词"""
    path = PROMPT_RULES_PATH / "system_prompt_edit.txt"
    if path.exists():
        return path.read_text(encoding='utf-8').strip()

    from .system_prompts import I2I_SYSTEM_PROMPT
    return I2I_SYSTEM_PROMPT

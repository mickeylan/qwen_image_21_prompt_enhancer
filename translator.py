"""
翻译功能模块

支持多种翻译方式:
1. 本地 transformers 模型
2. API 调用
3. 简单规则翻译 (备选)
"""

import re
import logging
from typing import Optional

# ============== 本地翻译器 ==============
class LocalTranslator:
    """使用 transformers 本地翻译"""
    
    def __init__(self, model_name: str = "facebook/mbart-large-50-many-to-many-mmt"):
        self.model_name = model_name
        self._pipeline = None
    
    def translate(self, text: str, source_lang: str = "zh_XX", target_lang: str = "en_XX") -> str:
        """执行翻译"""
        if self._pipeline is None:
            self._load_model()
        
        try:
            # mbart 支持多语言
            result = self._pipeline(
                text,
                src_lang=source_lang,
                tgt_lang=target_lang,
                max_length=1024,
            )
            return result[0]["translation_text"]
        except Exception as e:
            logging.error(f"Translation failed: {e}")
            return text
    
    def _load_model(self):
        """加载模型"""
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "translation",
                model=self.model_name,
                device_map="auto",
            )
        except ImportError:
            logging.warning("transformers not installed, local translation unavailable")
            self._pipeline = None


# ============== API 翻译器 ==============
class APITranslator:
    """使用 API 翻译"""
    
    def __init__(self, api_base: str, api_key: str, model: str = "gpt-4o"):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self._client = None
    
    def translate(self, text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
        """执行翻译"""
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_base=self.api_base, api_key=self.api_key)
        
        # 构建翻译 prompt
        if source_lang == "auto":
            source_lang = self._detect_language(text)
        
        lang_map = {
            "zh": ("Chinese", "Chinese"),
            "en": ("English", "English"),
            "ja": ("Japanese", "Japanese"),
            "ko": ("Korean", "Korean"),
        }
        
        src_name, tgt_name = lang_map.get(source_lang, ("Chinese", "English"))
        
        system_prompt = f"""You are a professional translator. Translate the following {src_name} text to {tgt_name}.

Rules:
- Only output the translated text, nothing else
- Preserve the original meaning and style
- Keep proper nouns and brand names unchanged
- If there are quotes or special formatting, preserve them
- Output in one paragraph without explanation"""
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"API translation failed: {e}")
            return f"[Translation Error: {e}]"
    
    def _detect_language(self, text: str) -> str:
        """检测语言"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(re.findall(r'[\w\u4e00-\u9fff]', text))
        
        if total_chars == 0:
            return "en"
        
        chinese_ratio = chinese_chars / total_chars
        return "zh" if chinese_ratio > 0.3 else "en"


# ============== 简单翻译规则 ==============
# 常用中英文对照 (用于快速翻译)
COMMON_TRANSLATIONS = {
    # 形容词
    "美丽的": "beautiful",
    "漂亮的": "pretty",
    "可爱的": "cute",
    "帅气的": "handsome",
    "温柔的": "gentle",
    "温暖的": "warm",
    "凉爽的": "cool",
    "明亮的": "bright",
    "黑暗的": "dark",
    "安静的": "quiet",
    "热闹的": "lively",
    "古老的": "ancient",
    "现代的": "modern",
    "繁华的": "prosperous",
    "宁静的": "peaceful",
    
    # 名词
    "猫": "cat",
    "狗": "dog",
    "鸟": "bird",
    "花": "flower",
    "树": "tree",
    "山": "mountain",
    "水": "water",
    "天空": "sky",
    "大海": "sea",
    "森林": "forest",
    "城市": "city",
    "村庄": "village",
    "房子": "house",
    "建筑": "building",
    "人物": "person",
    "女人": "woman",
    "男人": "man",
    "孩子": "child",
    "女孩": "girl",
    "男孩": "boy",
    
    # 场景
    "在": "in",
    "上": "on",
    "下": "under",
    "前": "front",
    "后": "back",
    "左": "left",
    "右": "right",
    "中间": "center",
    "旁边": "beside",
    "里面": "inside",
    "外面": "outside",
    
    # 动作
    "坐着": "sitting",
    "站着": "standing",
    "走着": "walking",
    "跑着": "running",
    "飞着": "flying",
    "看着": "looking",
    "笑着": "smiling",
    "跳舞": "dancing",
    "唱歌": "singing",
    "玩耍": "playing",
    
    # 时间
    "白天": "daytime",
    "夜晚": "night",
    "早晨": "morning",
    "傍晚": "evening",
    "日出": "sunrise",
    "日落": "sunset",
    
    # 天气
    "晴天": "sunny",
    "雨天": "rainy",
    "雪天": "snowy",
    "多云": "cloudy",
    "有雾": "foggy",
    
    # 风格
    "写实的": "realistic",
    "抽象的": "abstract",
    "卡通的": "cartoon",
    "水彩": "watercolor",
    "油画": "oil painting",
    "素描": "sketch",
    "摄影": "photograph",
    "电影感": "cinematic",
}


def simple_translate(text: str, direction: str = "zh2en") -> str:
    """
    简单翻译 - 使用规则替换
    
    注意: 这是一个基础的翻译实现，复杂文本建议使用 API 翻译
    """
    if direction == "zh2en":
        result = text
        # 按长度排序，优先替换长的词组
        sorted_phrases = sorted(COMMON_TRANSLATIONS.items(), key=lambda x: len(x[0]), reverse=True)
        for zh, en in sorted_phrases:
            result = result.replace(zh, en)
        return result
    else:
        # en2zh - 简单反向替换
        result = text
        reverse_dict = {v: k for k, v in COMMON_TRANSLATIONS.items()}
        sorted_phrases = sorted(reverse_dict.items(), key=lambda x: len(x[0]), reverse=True)
        for en, zh in sorted_phrases:
            result = result.replace(en, zh)
        return result


# ============== 翻译工具函数 ==============
def detect_language(text: str) -> str:
    """检测文本语言"""
    if not text.strip():
        return "en"
    
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))
    korean_chars = len(re.findall(r'[\uac00-\ud7af]', text))
    total_chars = len(re.findall(r'[\w\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', text))
    
    if total_chars == 0:
        return "en"
    
    # 计算各种语言字符比例
    max_ratio = max(chinese_chars, japanese_chars, korean_chars) / total_chars
    
    if chinese_chars / total_chars == max_ratio:
        return "zh"
    elif japanese_chars / total_chars == max_ratio:
        return "ja"
    elif korean_chars / total_chars == max_ratio:
        return "ko"
    else:
        return "en"

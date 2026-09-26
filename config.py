"""
配置管理
"""

from pathlib import Path
from typing import Optional
import json


class Config:
    """插件配置"""
    
    # 默认配置
    DEFAULT_CONFIG = {
        "pe_model_paths": {
            "t2i": "models/text_encoders/Qwen-Image-2.1-PE-T2I",
            "i2i": "models/text_encoders/Qwen-Image-2.1-PE-I2I",
        },
        "api_config": {
            "api_base": "",
            "api_key": "",
            "model_name": "gpt-4o",
        },
        "sampling_defaults": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "max_new_tokens": 4096,
            "presence_penalty": 0.0,
            "seed": 42,
        },
        "max_images": 16,
        "unload_after": True,
    }
    
    _instance: Optional['Config'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = cls._instance._load_config()
        return cls._instance
    
    def _load_config(self) -> dict:
        """加载配置"""
        config_path = self._get_config_path()
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                # 合并默认配置和用户配置
                config = self.DEFAULT_CONFIG.copy()
                config.update(user_config)
                return config
            except Exception:
                pass
        return self.DEFAULT_CONFIG.copy()
    
    def _get_config_path(self) -> Path:
        """获取配置路径"""
        return Path(__file__).parent / "config.json"
    
    def save(self):
        """保存配置"""
        config_path = self._get_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)
    
    def get(self, key: str, default=None):
        """获取配置项"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value
    
    def set(self, key: str, value):
        """设置配置项"""
        keys = key.split('.')
        target = self._config
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value


# 全局配置实例
config = Config()

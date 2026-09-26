"""
Qwen Image 2.1 Prompt Enhancer - ComfyUI Custom Node

专门针对 Qwen Image 2.1 的提示词增强节点，支持:
- 文生图 (T2I) 和 图生图 (I2I) 提示词增强
- 最多16张参考图片 (Qwen Image 2.1 最大支持)
- 配合 HR Qwen Director Config 使用
- 内置官方提示词工程规则
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
__version__ = "1.0.0"

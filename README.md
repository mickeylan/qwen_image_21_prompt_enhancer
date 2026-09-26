# Qwen Image 2.1 Prompt Enhancer

ComfyUI 自定义节点，专门针对 Qwen Image 2.1 的提示词增强。

## 功能特性

- **T2I 文生图提示词增强**: 将简短的用户描述扩展为详细的英文提示词
- **I2I 图生图提示词增强**: 结合参考图片，将编辑指令重写为精确的编辑指令
- **中英互译**: 使用 Qwen 模型进行中英互译
- **最多16张参考图片**: 支持 Qwen Image 2.1 最大图片数量
- **内置官方提示词工程规则**: 基于 Qwen 官方 PE 模型训练规则
- **利用 HR Qwen Director Config**: 复用 Qwen3.5/3.6/3.8 模型配置，无需重复加载

## 节点列表

| 节点 | 说明 |
|------|------|
| **QwenImage21PromptEnhancer** | 主增强节点，支持 T2I/I2I 增强 |
| **QwenImage21PromptEnhancerSimple** | 简化版增强节点 |
| **QwenImage21Translator** | 中英互译节点 |

## 运行模式

### 1. PE 模式 (推荐)
使用 Qwen 官方 Prompt Enhancer 模型，效果最佳。

**模型下载**: https://pan.quark.cn/s/b1ded16831e8
- T2I 模型: `Qwen-Image-2.1-PE-T2I`
- I2I 模型: `Qwen-Image-2.1-PE-I2I`

将模型放入: `ComfyUI/models/text_encoders/`

### 2. Director 模式 (推荐)
使用 HRQwen38DirectorConfig 配置的 Qwen 模型。

```
[HRQwen38DirectorConfig] → [QwenImage21PromptEnhancer] → [其他节点]
        ↓
   (director_config)
```

### 3. API 模式
调用远程 API 服务。

## 安装

1. 将整个目录复制到 `ComfyUI/custom_nodes/`
2. 确保 `prompt_rewrite/` 目录存在（包含官方提示词规则）

## 使用方法

### 基本用法

```
[Text] → [QwenImage21PromptEnhancer] → [Load Model] → [KSampler] → [Save Image]
```

### 带参考图片

```
[Load Image] → [QwenImage21PromptEnhancer]
                          ↓
                    (enhanced_prompt)
                          ↓
              [Load Model] → [KSampler] → [Save Image]
```

### 配合 HR Qwen Director Config

```
[HR Qwen Director Config] → [QwenImage21PromptEnhancer] → [其他节点]
        ↓
   (director_config)
```

## 输入输出

### 主增强节点输入
- `prompt`: 原始提示词 (支持中英文)
- `mode`: 模式选择 (`t2i` / `i2i`)
- `language`: 输出语言 (`en` / `zh`)
- `images`: 参考图片 (最多16张)
- `director_config`: HR Director Config (可选)
- `pe_model_path`: PE 模型路径 (可选)
- 采样参数: temperature, top_p, top_k, etc.

### 主增强节点输出
- `enhanced_prompt`: 增强后的提示词
- `thinking`: 思考过程
- `wh_ratio`: 宽高比 (如 "16:9")
- `ratio_follow`: 跟随的图片 (如 "<image1>")
- `full_result`: 完整结果 JSON

### 翻译节点

**输入**:
- `text`: 待翻译文本
- `direction`: 翻译方向 (`auto` / `zh2en` / `en2zh`)
- `director_config`: HR Director Config (可选)

**输出**:
- `translated_text`: 翻译后的文本
- `detected_lang`: 检测到的语言

**工作流示例**:
```
[HRQwen38DirectorConfig]─────┬──→ [QwenImage21PromptEnhancer] ──→ [Model]
                            │
                            └──→ [QwenImage21Translator] ──→ [Prompt Enhancer]
```

## 提示词工程规则

### T2I 规则
- 8步增强流程
- 详细的视觉描述
- 精确的光线、色彩、构图指导
- 输出格式: `{"rewritten_prompt": "...", "wh_ratio": "..."}`

### I2I 规则
- 属性解耦原则
- 图片角色识别
- 尺寸/比例确定
- 支持多图编辑场景

## 依赖

- ComfyUI
- transformers (PE 模式)
- vLLM (可选，PE 模式)
- torch

## License

MIT

# 基于YOLO符号化感知与DeepSeek大模型的开源开环VLA自动驾驶决策系统

本项目为大模型与感知课程的结课大作业。系统采用**模块化符号化 VLA (Symbolic Vision-Language-Action)** 架构，在算力和算力时间受限的条件下，实现了一种高效、可解释性强且具备时序记忆能力的开源自动驾驶决策流水线。

---

## 🌟 核心设计思想

直接将高分辨率视频流输入多模态大模型（VLM）存在巨大的显存和推理延迟瓶颈。本项目选择将任务解耦为**感知（眼睛）**、**记忆（时序）**、**决策（大脑）**和**行动（HUD）**四个模块：
1. **视觉感知层 (Perception):** 使用轻量化的 `YOLOv8-seg` 作为“眼睛”，在 Cityscapes 数据集上实时捕捉关键障碍物（行人、车辆等），将其降维为绝对尺寸、方位、画面占比等“符号化文本数据（JSON）”。
2. **时序动态记忆层 (Temporal Pseudo-TTC):** 引入跨帧短时记忆机制，通过计算正前方最威胁目标的面积变化率（$\Delta Area$），估算碰撞趋势（接近率），有效解决了单帧静态输入导致的“幽灵刹车”痛点。
3. **大脑决策层 (Reasoning):** 将符号化 JSON 与硬编码的《自动驾驶行为专家规则库》组装成 Prompt 发送给 `DeepSeek LLM`。利用其强大的思维链（CoT）能力做出可解释的安全驾驶指令。
4. **行动控制层 (Action HUD):** 接收规范化的 JSON 动作指令，利用 OpenCV 将检测框、匹配规则和最终动作（急刹、减速、巡航）以赛博朋克风的虚拟 HUD 仪表盘实时叠显到视频流中，用于开环评测。

---

## 💻 硬件与环境配置

* **硬件平台:** NVIDIA GeForce RTX 3090 (24GB VRAM)
* **基础环境:** PyTorch 2.1.0 / CUDA 12.1 / Python 3.10.13

### 环境搭建步骤 (规避 C API 冲突天坑)
由于 NumPy 2.0 升级会导致旧版 PyTorch 与 OpenCV 底层冲突报错 `RuntimeError: Numpy is not available`，请严格按照以下步骤搭建纯净环境：


```

```text
Successfully generated README.md

```bash
# 1. 创建隔离环境
conda create -n yolo_autodrive python=3.10.13 -y
conda activate yolo_autodrive

# 2. 升级 pip 工具
pip install --upgrade pip

# 3. 安装服务器无头版 OpenCV (规避 libGL.so.1 缺失导致的报错)
pip install opencv-python-headless

# 4. 安装 YOLO 官方生态库与大模型 API 客户端
pip install ultralytics openai Pillow

# 5. 【核心避坑】强制将 NumPy 降级回 1.x 版本
pip install "numpy<2" --force-reinstall

```

---

## 📂 项目目录结构

```text
yolo_autodrive/
├── SimHei.ttf              # 中文字体文件（用于PIL渲染，可选）
├── llm_driver.py           # 大脑模块：封装专家规则库并调用 DeepSeek API
├── main.py                 # 主控流水线：读取序列 -> YOLO感知 -> 趋势计算 -> LLM决策 -> 导出HUD视频
├── README.md               # 项目说明文档
└── data/
    └── cityscapes/
        └── leftImg8bit/
            └── demoVideo/
                └── stuttgart_00/  # Cityscapes 官方连续帧图像序列 (.png)

```

---

## 🛠️ 核心模块代码实现

### 1. 大脑决策层 (`llm_driver.py`)

请确保在此文件中配置您的 `DEEPSEEK_API_KEY`。为彻底规避 Linux 无头服务器在 OpenCV 下渲染中文字体导致的 `??????` 豆腐块乱码，本模块强制大模型以 **纯英文 JSON** 结构返回决策。

```python
import json
from openai import OpenAI

DEEPSEEK_API_KEY = "您的DeepSeek_API_Key" 

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="[https://api.deepseek.com](https://api.deepseek.com)"
)

def get_driving_decision(perception_dict):
    system_prompt = \"\"\"
    You are a high-level autonomous driving decision brain. You must strictly base your decisions on the provided perception data and follow this Expert Knowledge Base:
    
    1. CRITICAL DANGER (EMERGENCY BRAKE): If an obstacle is in the "正前方" AND distance is "极近 (危险)"/"中等距离" AND the approach trend is "快速接近", output "EMERGENCY BRAKE".
    2. SAFE FOLLOWING (CRUISE): If an obstacle is in the "正前方" AND the approach trend is "远离" or "相对静止/无目标", output "CRUISE".
    3. CROWDED TRAFFIC (DECELERATE): If the total number of obstacles is >= 8, output "DECELERATE" regardless of immediate front danger.
    4. SIDE PRE-WARNING (CAUTION): If all obstacles are on the "左侧" or "右侧", output "CAUTION".
    5. CLEAR ROAD (CRUISE): If no obstacles or all are far away, output "CRUISE".

    You must output in a strict JSON format with the following keys:
    {
      "环境理解": "Brief English description of current traffic",
      "匹配专家库": "Rule X Triggered",
      "最终决策": "EMERGENCY BRAKE | DECELERATE | CAUTION | CRUISE"
    }
    \"\"\"
    user_prompt = f"Current perception data:\\n{json.dumps(perception_dict, ensure_ascii=False)}"
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"环境理解": "API Error", "匹配专家库": "Fallback", "最终决策": "EMERGENCY BRAKE"}

```

### 2. 主控流水线与时序动态层 (`main.py`)

主要负责读取 Cityscapes 连续图像序列，提取 YOLO 分割特征，计算时序面积变化率（$\Delta Area$）并渲染最终的 HUD 控制台。完整代码参见仓库中的 `main.py`。

---

## 🚀 运行与启动

1. 将 Cityscapes 的 `demoVideo` 图像序列解压至 `data` 目录下。
2. 检查 `llm_driver.py` 中的 API Key 是否填写正确。
3. 检查 `main.py` 中的 `demo_video_dir` 路径是否正确指向您的图像序列文件夹。
4. 启动主流水线：

```bash
python main.py

```

5. 运行完毕后，系统会在根线下导出 `output_hud.mp4` 视频文件。你可以将该视频下载到本地，直接放入结课答辩 PPT 中展示。

---



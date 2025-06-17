# ros_lm

**ROS 2 Humble Package for Interacting with Open Large (Vision-)Language Models**

ros_lm provides a native ROS 2 interface to load, query, and unload LLM and LVLM models via a single service.

---

## 🌟 Features

- **Service API** (`llm_generate_text`) for:
    - Loading and unloading models (resource management)
    - Generating text (optionally with images for LVLM)
- Supports **LLMs** (text) and **LVLMs** (vision + text)
- Example client included to get started quickly

---

## 🛠️ Dependencies

- **ROS 2 Humble**
- **Python 3.10+**
- ROS packages:
    - `rclpy`
    - `ros_lm_interfaces`
- Key Python libraries:
    - `transformers`
    - `torch`
- The rest of the dependencies are in [`requirements.txt`](./requirements.txt)

---

## ⚙️ Installation & Build

```bash
# 1. Clone the main repository in your workspace
cd ~/ros2_ws
git clone https://github.com/MAPIRlab/ros_lm.git src/

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Build with colcon
colcon build

# 4. Source the overlay
source install/setup.bash
```

---

## 🚀 Usage

### Run the service server

Start the node that exposes the `llm_generate_text` service:

```bash
ros2 run ros_lm server
```

### `OpenLLMRequest` Interface

The service uses the `ros_lm_interfaces/srv/OpenLLMRequest` message, with these fields:

| Field         | Type       | Description                                                      |
|---------------|------------|------------------------------------------------------------------|
| `action`      | `int32`    | `1=LOAD`, `2=GENERATE`, `3=UNLOAD`                              |
| `model_id`    | `string`   | Identifier (e.g. `meta-llama/Llama-3.1-8B-Instruct`)            |
| `prompt`      | `string`   | Input text                                                      |
| `max_length`  | `uint32`   | Maximum length (e.g. `200`)                                     |
| `temperature` | `float32`  | Sampling temperature (e.g. `0.7`)                               |
| `top_k`       | `uint32`   | Top-k sampling (e.g. `50`)                                      |
| `top_p`       | `float32`  | Nucleus sampling (e.g. `0.9`)                                   |
| `images`      | `string[]` | Base64 list of images (LVLM only)                               |

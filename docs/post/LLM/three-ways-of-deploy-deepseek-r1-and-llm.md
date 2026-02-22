---
title: DeepSeek-R1大模型本地部署的三种方式，总有一种适合你
date: 2025-02-03 18:06:20
tag:
  - LLM
  - transformer
description: 三种不同的方式部署大模型（deepseek r1），分别是 ollama, LM Studio 和 vllm，从个人测试部署到工业产品使用，让你一次性掌握大模型的不同部署方式。
publish: true
permalink: /post/three-ways-of-deploy-deepseek-r1-and-llm.html
---

由于 DeepSeek-R1 爆火，导致 DeepSeek 官网用起来非常卡（至 2025 年 2 月 2 日），因此催生除了很多本地部署的需求。而这里我选用了三种最常用的部署方式，从普通人测试使用到工业界部署，让你一次性掌握大模型的部署方式。

## 对比总结

| 特性        | Ollama    | LM Studio | vLLM       |
| --------- | --------- | --------- | ---------- |
| **定位**    | 本地快速体验    | 图形化交互工具   | 生产级推理引擎    |
| **用户群体**  | 开发者/爱好者   | 非技术用户     | 企业/工程师     |
| **部署复杂度** | 低         | 极低        | 中高         |
| **性能优化**  | 基础        | 一般        | 极致         |
| **适用场景**  | 开发测试、原型验证 | 个人使用、教育演示 | 高并发生产环境    |
| **扩展性**   | 有限        | 无         | 强（分布式/云原生） |

- 建议
  - 想快速体验模型：**Ollama**
  - 需要图形界面和隐私保护：**LM Studio**
  - 企业级高并发需求：**vLLM**

## 方式1： ollama

Ollama 是一个开源的本地化大模型部署工具，旨在简化大型语言模型（LLM）的安装、运行和管理。它支持多种模型架构，并提供与 **OpenAI 兼容的 API 接口**，适合开发者和企业快速搭建私有化 AI 服务。

- 官网
  - <https://ollama.com/>
- 开源链接
  - <https://github.com/ollama/ollama>

Ollama 的**特点**：

- **轻量化部署**：完全的本地化部署。
- **多模型支持**：兼容各种开源模型，包括 qwen、deepseek、LLaMA 等。
- **跨平台支持**：支持主流的 Windows、Mac、Linux。

使用 Ollama 安装 DeepSeek-R1 等大模型一共就三个步骤。

- 步骤 1：下载 ollama
  - windows 和 mac 进入 <https://ollama.com/download> 下载对应的安装包，然后安装即可
  - Linux 的话使用如下命令安装
    - `curl -fsSL https://ollama.com/install.sh | sh`

![DeepSeek-R1大模型本地部署的三种方式，总有一种适合你-20250202210452603](https://cfcdn.bruceyuan.com/blog/2025/DeepSeek-R1大模型本地部署的三种方式，总有一种适合你-20250202210452603.webp)

- 步骤 2：启动 ollama （一般都默认启动了，点击运行或者命令行启动）
  - 如果是 mac 和 windows 中，点击启动即可；
  - 如果是 linux 中，理论上上方的脚本会自动启动 ollama，但是如果发生意外，可以使用
    - `ollama serve` 进行启动；
- 步骤 3：运行对应的模型
  - 运行 deepseek-r1 模型，这里选用的是 deepseek-r1 的蒸馏小模型，`deepseek-r1:1.5b`
    - `ollama run deepseek-r1:1.5b`
  - 如果使用其他模型则是
    - `ollama run {model_name}`
    - `{model_name}`  替换成真实的模型名字，名字可以在 <https://ollama.com/search> 中获取。
  - 注意⚠️：这样启动模型具有对应的上下文，本质上是启动了 chat 的接口。
  - 具体效果如下：可以看到模型具有 `think` 的能力，但是由于模型比较小，效果依然不是特别好。

![DeepSeek-R1大模型本地部署的三种方式，总有一种适合你-20250202213103381](https://cfcdn.bruceyuan.com/blog/2025/DeepSeek-R1大模型本地部署的三种方式，总有一种适合你-20250202213103381.webp)
> 从上面也可以看出，比较适合本地快速测试大模型。
>
## 方式2：LM Studio

LM Studio 是一款桌面应用程序，用于在您的计算机上开发和试验 LLMs。

LM Studio 的**特点**：

- 运行本地 LLMs 的桌面应用程序
- 熟悉的聊天界面
- 搜索和下载功能（通过 Hugging Face 🤗）
- 可以侦听类似 OpenAI 端点的本地服务器
- 用于管理本地模型和配置的系统

- LM Studio 是一个可视化的软件（ <https://lm-studio.cn> ），基本上没有任何的学习成本。具体操作界面如下：

![DeepSeek-R1大模型本地部署的三种方式，总有一种适合你-20250202215154305](https://cfcdn.bruceyuan.com/blog/2025/DeepSeek-R1大模型本地部署的三种方式，总有一种适合你-20250202215154305.webp)

- 模型下载和运行的步骤如下

![DeepSeek-R1大模型本地部署的三种方式，总有一种适合你-20250202214613708](https://cfcdn.bruceyuan.com/blog/2025/DeepSeek-R1大模型本地部署的三种方式，总有一种适合你-20250202214613708.webp)

因此 LM Studio 最适合普通人使用，没有任何的使用成本，全部都是可视化操作。比如适合个人学习、内容创作、教育演示。以及需要隐私保护的本地对话场景。

## 方式 3：vLLM

vLLM 是加州大学伯克利分校开发的高性能大模型推理框架，**专为生产环境优化**，支持分布式部署和极低延迟的模型服务，**适合企业级应用**。这也是在业界用的最多的推理框架之一，如果你需要稳定的、优化性能更强的、社区服务支持更好的，那么 vLLM 一定是不二的候选。

vLLM 的**特点**

- **极致性能**：通过各种算法优化，显著提升吞吐量。
- **生产级功能**：动态批处理（Continuous Batching）、分布式推理、多 GPU 并行。
- **模型兼容性**：支持 HuggingFace Transformers 模型架构（如 Llama、deepseek等）。
- **开放生态**：与 OpenAI API 兼容，可无缝替换商业模型。

通过 vllm 对外提供一个服务，这样你就可以一次部署，在多个不同的地方使用，比如家里、公司内、甚至在星巴克内；当然也可以把这个服务提供给其他人。

通过 `pip install vllm` [安装](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)之后，一般有两种形式使用。

### 方式1：提供 openai 的API接口的 server

下面我用 [AIStackDC](https://aistackdc.com) 平台演示（当然也可以本地使用），这是一个云服务器平台，最大的特点就是便宜好用，如果用我的邀请链接：[https://aistackdc.com/phone-register?invite_code=D872A9](https://aistackdc.com/phone-register?invite_code=D872A9)，可以额外获得 1张 1 折（5 小时）和几张5 折（36小时）优惠券。

#### 服务器启动

- 首先在容器管理中，点击**创建实例**，选择 4090 和 PyTorch 2.30 以及 python 3.12 对应的镜像。

![DeepSeek-R1大模型本地部署的三种方式，总有一种适合你-20250203122902695](https://cfcdn.bruceyuan.com/blog/2025/DeepSeek-R1大模型本地部署的三种方式，总有一种适合你-20250203122902695.webp)

- 进入 jupyter lab （similar to jupyter notebook)
![DeepSeek-R1大模型本地部署的三种方式，总有一种适合你-20250203123030296](https://cfcdn.bruceyuan.com/blog/2025/DeepSeek-R1大模型本地部署的三种方式，总有一种适合你-20250203123030296.webp)

- 然后选择一个 terminal 进入命令行，安装对应的 vllm
  - `pip install vllm`
  - 如果执行失败可以使用阿里云的镜像
    - `pip install vllm --index-url https://mirrors.aliyun.com/pypi/simple/`
- 安装完毕之后，启动 vllm 服务，下面以 deepseek r1 的蒸馏模型为例
  - `HF_ENDPOINT=https://hf-mirror.com vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
  - 其中 HF_ENDPOINT 是为了设置代理，为了在国内下载更快。
  - 后面的`vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` 才是真正的启动命令。

#### 服务器中使用

具体参考 <https://docs.vllm.ai/en/latest/getting_started/quickstart.html#openai-compatible-server>

- 方式 1：使用 curl 使用 genererate 接口

```shell
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

- 方式 2： 代码实现

```python
from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ]
)
print("Chat response:", chat_response)
```

#### 本地电脑（mac/windows）中使用

由于 [AIStackDC](https://aistackdc.com) 平台有对应的 IP，我们是可以 ssh 登录的，具体看[https://aistackdc.com/help/instanceserviceportssh](https://aistackdc.com/help/instanceserviceportssh#ssh)，

在本地电脑执行 `ssh -i <SSH私钥文件路径> -CNg -L <本地监听端口>:127.0.0.1:<容器内服务监听端口> root@221.178.84.158 -p <实例SSH连接端口>`

- SSH私钥文件路径：本地存放SSH私钥文件的路径
- 容器内服务监听端口：容器实例中启动的服务监听的端口
- 本地监听端口：本地监听的转发端口（即后续想要在本地浏览器中访问容器实例中服务的访问端口，该端口自定义，确保本地系统中该端口未被占用即可）
- 实例SSH连接端口：SSH连接容器实例使用的端口

举例：`ssh -i id_rsa -CNg -L 9000:127.0.0.1:8000 root@221.178.85.21 -p 34538`

![DeepSeek-R1大模型本地部署的三种方式，总有一种适合你-20250203124957011](https://cfcdn.bruceyuan.com/blog/2025/DeepSeek-R1大模型本地部署的三种方式，总有一种适合你-20250203124957011.webp)

执行完之后，就可以在本地使用了，同时也可以通过 `localhost:9000` 配置到 [open-webui](https://github.com/open-webui/open-webui)等聊天界面中，这样可以实现

![DeepSeek-R1大模型本地部署的三种方式，总有一种适合你-20250203124819972](https://cfcdn.bruceyuan.com/blog/2025/DeepSeek-R1大模型本地部署的三种方式，总有一种适合你-20250203124819972.webp)

### 方式 2：批量离线推理

还有一种是使用 python api 在代码内部 load，然后做各种定制化的开发，但是如果你采用这种方式，那么说明你不是本教程的受众，你应该自己有足够的能力完成模型的部署和开发。

- Example.

```python
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="facebook/opt-125m")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## 交个朋友🤣

最后欢迎关注我，基本全网同名 [chaofa用代码打点酱油](https://yuanchaofa.com/)

- 公众号： ![chaofa用代码打点酱油](https://yuanchaofa.com/llms-zero-to-hero/chaofa-wechat-official-account.png)
- [B站-chaofa用代码打点酱油](https://space.bilibili.com/12420432)
- [YouTube-chaofa用代码打点酱油](https://www.youtube.com/@bbruceyuan)
- [chaofa 的 notion 简介](https://chaofa.notion.site/11a569b3ecce49b2826d679f5e2fdb54)

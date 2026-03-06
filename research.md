# Real-ESRGAN 项目深度研究报告

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 软件架构总览](#2-软件架构总览)
- [3. 目录结构](#3-目录结构)
- [4. 依赖关系](#4-依赖关系)
- [5. 核心模块详解](#5-核心模块详解)
  - [5.1 网络架构 (archs/)](#51-网络架构-archs)
  - [5.2 数据集 (data/)](#52-数据集-data)
  - [5.3 模型训练 (models/)](#53-模型训练-models)
  - [5.4 推理工具 (utils.py)](#54-推理工具-utilspy)
  - [5.5 训练入口 (train.py)](#55-训练入口-trainpy)
- [6. 推理脚本详解](#6-推理脚本详解)
  - [6.1 图像超分辨率推理](#61-图像超分辨率推理)
  - [6.2 视频超分辨率推理](#62-视频超分辨率推理)
  - [6.3 Cog 云端推理 API](#63-cog-云端推理-api)
- [7. 训练流程详解](#7-训练流程详解)
  - [7.1 数据准备流程](#71-数据准备流程)
  - [7.2 退化模型（Degradation Model）](#72-退化模型degradation-model)
  - [7.3 训练配置文件](#73-训练配置文件)
  - [7.4 训练策略](#74-训练策略)
- [8. API 接口详解](#8-api-接口详解)
- [9. C++ API 详解 (NCNN-Vulkan)](#9-c-api-详解-ncnn-vulkan)
  - [9.1 概述与架构](#91-概述与架构)
  - [9.2 C++ 类接口](#92-c-类接口)
  - [9.3 C++ 图像推理完整示例](#93-c-图像推理完整示例)
  - [9.4 编译与构建](#94-编译与构建)
  - [9.5 命令行工具使用](#95-命令行工具使用)
- [10. Java/Android API 详解 (NCNN-Android)](#10-javaandroid-api-详解-ncnn-android)
  - [10.1 概述与架构](#101-概述与架构)
  - [10.2 JNI 桥接层](#102-jni-桥接层)
  - [10.3 Java/Android 图像推理完整示例](#103-javaandroid-图像推理完整示例)
  - [10.4 Android 项目集成指南](#104-android-项目集成指南)
- [11. ONNX 导出与跨平台部署](#11-onnx-导出与跨平台部署)
- [12. 调用流程图](#12-调用流程图)
  - [12.1 推理调用流程](#121-推理调用流程)
  - [12.2 训练调用流程](#122-训练调用流程)
- [13. 关键设计模式与工程决策](#13-关键设计模式与工程决策)
- [14. 预训练模型一览](#14-预训练模型一览)
- [15. 测试体系](#15-测试体系)
- [16. 总结](#16-总结)

---

## 1. 项目概述

**Real-ESRGAN**（Real-World Enhanced Super-Resolution Generative Adversarial Network）是一个用于通用图像/视频恢复的实用超分辨率算法。它将 ESRGAN 扩展到真实世界应用场景，采用纯合成数据训练，能够处理真实世界中常见的复杂退化（模糊、噪声、JPEG压缩伪影等）。

### 核心特性

| 特性 | 说明 |
|------|------|
| **通用超分辨率** | 支持 2x 和 4x 上采样 |
| **多模型支持** | 通用模型、动漫模型、视频模型 |
| **人脸增强** | 集成 GFPGAN 进行人脸修复 |
| **多平台** | Windows、Linux、macOS |
| **灵活部署** | PyTorch 推理、ONNX 导出、NCNN 移动端 |
| **视频处理** | 支持多GPU并行的视频超分辨率 |
| **云端API** | 通过 Cog/Replicate 提供云端推理服务 |

---

## 2. 软件架构总览

Real-ESRGAN 采用分层架构，建立在 **BasicSR** 框架之上：

```
┌──────────────────────────────────────────────────────────┐
│                     应用层 (Application Layer)             │
│  inference_realesrgan.py | inference_realesrgan_video.py  │
│  cog_predict.py                                           │
├──────────────────────────────────────────────────────────┤
│                     工具层 (Utility Layer)                 │
│  realesrgan/utils.py                                      │
│  ├── RealESRGANer (核心推理引擎)                           │
│  ├── PrefetchReader (后台预读取)                           │
│  └── IOConsumer (后台写入)                                 │
├──────────────────────────────────────────────────────────┤
│                     模型层 (Model Layer)                   │
│  realesrgan/models/                                       │
│  ├── RealESRGANModel (GAN训练)                            │
│  └── RealESRNetModel (非对抗训练)                          │
├──────────────────────────────────────────────────────────┤
│                   网络架构层 (Architecture Layer)           │
│  realesrgan/archs/                                        │
│  ├── SRVGGNetCompact (轻量级生成器)                        │
│  └── UNetDiscriminatorSN (判别器)                          │
│  BasicSR提供:                                              │
│  └── RRDBNet (RRDB生成器)                                  │
├──────────────────────────────────────────────────────────┤
│                     数据层 (Data Layer)                    │
│  realesrgan/data/                                         │
│  ├── RealESRGANDataset (在线合成退化)                      │
│  └── RealESRGANPairedDataset (离线配对数据)                 │
├──────────────────────────────────────────────────────────┤
│                   框架层 (Framework Layer)                  │
│  BasicSR (训练框架, 注册机制, 基础模型)                     │
│  PyTorch (深度学习引擎)                                    │
└──────────────────────────────────────────────────────────┘
```

---

## 3. 目录结构

```
Real-ESRGAN/
├── realesrgan/                    # 核心 Python 包
│   ├── __init__.py                # 包初始化，导出所有子模块
│   ├── version.py                 # 版本信息
│   ├── train.py                   # 训练入口
│   ├── utils.py                   # 推理工具类 (RealESRGANer等)
│   ├── archs/                     # 网络架构定义
│   │   ├── __init__.py            # 自动注册所有 *_arch.py
│   │   ├── discriminator_arch.py  # UNet判别器 (频谱归一化)
│   │   └── srvgg_arch.py          # SRVGGNetCompact 轻量生成器
│   ├── data/                      # 数据集定义
│   │   ├── __init__.py            # 自动注册所有 *_dataset.py
│   │   ├── realesrgan_dataset.py  # 在线合成退化数据集
│   │   └── realesrgan_paired_dataset.py  # 配对数据集
│   └── models/                    # 训练模型定义
│       ├── __init__.py            # 自动注册所有 *_model.py
│       ├── realesrgan_model.py    # GAN训练模型
│       └── realesrnet_model.py    # 回归训练模型
├── inference_realesrgan.py        # 图像推理脚本
├── inference_realesrgan_video.py  # 视频推理脚本
├── cog_predict.py                 # Replicate 云端推理
├── scripts/                       # 数据准备脚本
│   ├── extract_subimages.py       # 裁切子图
│   ├── generate_meta_info.py      # 生成元数据
│   ├── generate_meta_info_pairdata.py  # 配对数据元数据
│   ├── generate_multiscale_DF2K.py     # 多尺度数据增强
│   └── pytorch2onnx.py           # ONNX模型转换
├── options/                       # 训练配置文件 (YAML)
│   ├── train_realesrgan_x4plus.yml
│   ├── train_realesrgan_x2plus.yml
│   ├── train_realesrnet_x4plus.yml
│   ├── train_realesrnet_x2plus.yml
│   ├── finetune_realesrgan_x4plus.yml
│   └── finetune_realesrgan_x4plus_pairdata.yml
├── tests/                         # 单元测试
│   ├── test_dataset.py            # 数据集测试
│   ├── test_discriminator_arch.py # 判别器测试
│   ├── test_model.py              # 模型测试
│   └── test_utils.py              # 工具类测试
├── docs/                          # 文档
├── inputs/                        # 示例输入图像
├── weights/                       # 模型权重存放目录
├── setup.py                       # 包安装配置
├── requirements.txt               # Python依赖
└── cog.yaml                       # Replicate容器配置
```

---

## 4. 依赖关系

### 4.1 核心依赖

```
Real-ESRGAN
├── basicsr >= 1.4.2        # 底层训练框架（提供 RRDBNet, SRModel, SRGANModel, 注册机制等）
├── torch >= 1.7             # 深度学习引擎
├── torchvision              # 视觉工具
├── facexlib >= 0.2.5        # 人脸检测/解析库
├── gfpgan >= 1.3.5          # 人脸增强模型
├── numpy                    # 数值计算
├── opencv-python            # 图像/视频处理
├── Pillow                   # 图像处理
└── tqdm                     # 进度条
```

### 4.2 依赖调用关系

```
Real-ESRGAN ──依赖──> BasicSR
    │                    ├── 提供: RRDBNet (生成器架构)
    │                    ├── 提供: SRModel, SRGANModel (基础训练模型)
    │                    ├── 提供: train_pipeline() (训练管线)
    │                    ├── 提供: ARCH_REGISTRY, MODEL_REGISTRY, DATASET_REGISTRY (注册机制)
    │                    ├── 提供: DiffJPEG, USMSharp (退化工具)
    │                    ├── 提供: random_mixed_kernels, circular_lowpass_kernel (核生成)
    │                    ├── 提供: filter2D, random_add_gaussian_noise_pt 等 (退化操作)
    │                    └── 提供: load_file_from_url (模型下载)
    │
    ├──依赖──> GFPGAN
    │              └── 提供: GFPGANer (人脸增强推理引擎)
    │
    ├──依赖──> facexlib
    │              └── 提供: 人脸检测和解析模型
    │
    └──依赖──> PyTorch
                   ├── torch.nn (神经网络层)
                   ├── torch.nn.functional (函数式操作)
                   └── torch.utils.data (数据加载)
```

### 4.3 BasicSR 注册机制

Real-ESRGAN 深度依赖 BasicSR 的注册机制，这是整个框架的核心设计：

```python
# 架构注册 - realesrgan/archs/discriminator_arch.py
@ARCH_REGISTRY.register()
class UNetDiscriminatorSN(nn.Module): ...

# 模型注册 - realesrgan/models/realesrgan_model.py
@MODEL_REGISTRY.register()
class RealESRGANModel(SRGANModel): ...

# 数据集注册 - realesrgan/data/realesrgan_dataset.py
@DATASET_REGISTRY.register()
class RealESRGANDataset(data.Dataset): ...
```

各子模块的 `__init__.py` 通过动态导入实现自动注册：

```python
# 以 archs/__init__.py 为例
import importlib
from os import path as osp

arch_folder = osp.dirname(osp.abspath(__file__))
# 扫描所有 *_arch.py 文件并导入
for f in scandir(arch_folder):
    if f.name.endswith('_arch.py'):
        importlib.import_module(f'realesrgan.archs.{f.name[:-3]}')
```

这种机制使得添加新的架构/模型/数据集只需：
1. 创建对应的 `*_arch.py` / `*_model.py` / `*_dataset.py` 文件
2. 使用 `@REGISTRY.register()` 装饰器
3. 在配置文件 (YAML) 中引用类名即可

---

## 5. 核心模块详解

### 5.1 网络架构 (archs/)

#### 5.1.1 SRVGGNetCompact — 轻量级超分辨率生成器

**文件**: `realesrgan/archs/srvgg_arch.py`

这是一个紧凑的 VGG 风格超分辨率网络，专为推理速度优化：

```
输入图像 (B, C_in, H, W)
    │
    ├── Conv2d(C_in → num_feat, 3×3)  ──── 特征提取
    ├── Activation (PReLU/ReLU/LeakyReLU)
    │
    ├── [Conv2d(num_feat → num_feat, 3×3) + Activation] × num_conv  ──── 特征变换
    │
    ├── Conv2d(num_feat → C_out × upscale², 3×3)  ──── 上采样准备
    ├── PixelShuffle(upscale)  ──── 亚像素上采样: (B, C×r², H, W) → (B, C, H×r, W×r)
    │
    └── + F.interpolate(input, scale_factor=upscale)  ──── 残差学习（跳跃连接）
    │
    输出图像 (B, C_out, H×upscale, W×upscale)
```

**关键参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_in_ch` | 3 | 输入通道数 |
| `num_out_ch` | 3 | 输出通道数 |
| `num_feat` | 64 | 中间特征通道数 |
| `num_conv` | 16 | 卷积层数量 |
| `upscale` | 4 | 上采样因子 |
| `act_type` | `'prelu'` | 激活函数类型 |

**设计亮点**:
- 全部使用 3×3 卷积，结构简洁，推理高效
- PixelShuffle 上采样避免棋盘格伪影
- 全局残差学习（输入双线性插值 + 网络输出）提高训练稳定性

#### 5.1.2 UNetDiscriminatorSN — U-Net 频谱归一化判别器

**文件**: `realesrgan/archs/discriminator_arch.py`

采用 U-Net 结构配合频谱归一化（Spectral Normalization）的判别器，能够对每个像素给出真/假判断：

```
输入 (B, 3, H, W)
    │
    ▼ 下采样路径 (Encoder)
    ├── conv0: 3 → F (3×3, s=1)       → x0
    ├── conv1: F → 2F (4×4, s=2, SN)  → x1  [H/2]
    ├── conv2: 2F → 4F (4×4, s=2, SN) → x2  [H/4]
    └── conv3: 4F → 8F (4×4, s=2, SN) → x3  [H/8]
    │
    ▼ 上采样路径 (Decoder)
    ├── Upsample(2x) + conv4: 8F → 4F (3×3, SN)  + skip(x2)  [H/4]
    ├── Upsample(2x) + conv5: 4F → 2F (3×3, SN)  + skip(x1)  [H/2]
    └── Upsample(2x) + conv6: 2F → F  (3×3, SN)  + skip(x0)  [H]
    │
    ▼ 输出层
    ├── conv7: F → F (3×3, SN)
    ├── conv8: F → F (3×3, SN)
    └── conv9: F → 1 (3×3)  ──── 逐像素真/假分数
    │
    输出 (B, 1, H, W)
```

**设计亮点**:
- U-Net 结构提供逐像素判别，而非传统判别器的单一标量输出
- 跳跃连接保留多尺度信息
- 频谱归一化稳定 GAN 训练
- 所有层使用 LeakyReLU(0.2) 激活

#### 5.1.3 RRDBNet — RRDB 生成器（由 BasicSR 提供）

Real-ESRGAN 的主力生成器架构由 BasicSR 提供，是 ESRGAN 中 RRDB（Residual-in-Residual Dense Block）网络的实现：

```
输入 (B, 3, H, W)
    │
    ├── conv_first: 3 → 64 (3×3)
    │
    ├── [RRDB Block] × 23  ──── 残差嵌套密集块
    │       │
    │       └── 每个 RRDB 包含 3 个残差密集块 (RDB)
    │           每个 RDB 包含 5 个密集连接的卷积层
    │
    ├── conv_body: 64 → 64 (3×3)
    ├── + skip connection (conv_first 输出)
    │
    ├── Upsample × 2 (PixelShuffle 或 nearest + conv)
    │
    ├── conv_last: 64 → 3 (3×3)
    │
    输出 (B, 3, H×4, W×4)
```

---

### 5.2 数据集 (data/)

#### 5.2.1 RealESRGANDataset — 在线合成退化数据集

**文件**: `realesrgan/data/realesrgan_dataset.py`

这是 Real-ESRGAN 的核心创新——在训练时在线合成退化图像，而非使用预先准备的配对数据：

**数据加载流程**:

```
磁盘/LMDB → 读取GT图像 → 数据增强 → 生成退化核 → 返回数据字典
```

**退化核生成过程**:

```
第一退化核 (kernel1):
    ├── 随机选择核大小: [7, 9, 11, 13, 15, 17, 19, 21]
    ├── 以 sinc_prob 概率选择 sinc 核:
    │   ├── kernel_size < 13: ω_c ∈ [π/3, π]
    │   └── kernel_size ≥ 13: ω_c ∈ [π/5, π]
    │   └── circular_lowpass_kernel(ω_c, kernel_size)
    └── 否则使用 random_mixed_kernels():
        └── 从 [iso, aniso, generalized_iso, generalized_aniso,
             plateau_iso, plateau_aniso] 中随机选择

第二退化核 (kernel2):
    └── 同样流程，但使用独立的参数 (*_2 后缀)

最终 sinc 核 (sinc_kernel):
    ├── 以 final_sinc_prob 概率:
    │   └── circular_lowpass_kernel(ω_c ∈ [π/3, π], kernel_size)
    └── 否则: 使用脉冲核 (无模糊效果)
```

所有核填充至 21×21 大小，保证张量维度一致。

**输出数据字典**:

```python
{
    'gt': Tensor[3, 400, 400],       # 高质量GT图像
    'kernel1': Tensor[21, 21],        # 第一退化模糊核
    'kernel2': Tensor[21, 21],        # 第二退化模糊核
    'sinc_kernel': Tensor[21, 21],    # 最终sinc滤波核
    'gt_path': str                    # 原始图像路径
}
```

#### 5.2.2 RealESRGANPairedDataset — 配对数据集

**文件**: `realesrgan/data/realesrgan_paired_dataset.py`

用于微调场景的简单配对数据集，直接加载预先退化的 LQ-GT 图像对：

```
磁盘/LMDB → 读取GT和LQ图像 → 配对随机裁切 → 数据增强 → 归一化 → 返回
```

**输出数据字典**:

```python
{
    'gt': Tensor[3, gt_size, gt_size],         # 高质量图像
    'lq': Tensor[3, gt_size//scale, gt_size//scale],  # 低质量图像
    'gt_path': str,
    'lq_path': str
}
```

---

### 5.3 模型训练 (models/)

#### 5.3.1 RealESRNetModel — 回归训练模型

**文件**: `realesrgan/models/realesrnet_model.py`  
**继承**: `BasicSR.SRModel`

这是第一阶段训练使用的非对抗模型，只使用像素级损失（L1）训练生成器：

**核心方法**:

- **`feed_data(data)`**: 在线合成退化流水线（详见 §7.2）
- **`_dequeue_and_enqueue()`**: 训练对队列管理，保证退化多样性
- 训练优化由父类 `SRModel` 处理

**训练损失**: 仅 L1 像素损失

#### 5.3.2 RealESRGANModel — GAN 训练模型

**文件**: `realesrgan/models/realesrgan_model.py`  
**继承**: `BasicSR.SRGANModel`

这是第二阶段训练使用的对抗训练模型，产生视觉质量更高的结果：

**核心方法**:

- **`feed_data(data)`**: 与 RealESRNetModel 相同的退化流水线，但始终对 GT 应用 USM 锐化
- **`_dequeue_and_enqueue()`**: 同上
- **`optimize_parameters(current_iter)`**: 完整的 GAN 训练优化

**训练损失**:

```
总生成器损失 = λ_pixel × L1_Loss(SR, GT)
             + λ_percep × PerceptualLoss(SR, GT)    # VGG19 特征匹配
             + λ_gan × GANLoss(D(SR), True)          # 对抗损失

总判别器损失 = GANLoss(D(GT), True)                    # 真实图像
             + GANLoss(D(SR.detach()), False)          # 生成图像
```

**USM 锐化控制**: 可通过配置文件选择是否对不同损失使用 USM 锐化后的 GT：
- `l1_gt_usm`: L1 损失使用 USM GT
- `percep_gt_usm`: 感知损失使用 USM GT
- `gan_gt_usm`: GAN 损失使用 USM GT

**训练对队列机制**:

```
队列容量: 180 (默认)

当队列未满时:
    → 直接添加当前批次到队列

当队列已满时:
    → 随机打乱队列
    → 取出队列前 batch_size 个样本作为当前训练数据
    → 将当前批次放入队列替换被取出的位置

目的: 打破批次内退化参数的一致性，增加训练多样性
```

---

### 5.4 推理工具 (utils.py)

**文件**: `realesrgan/utils.py`

#### 5.4.1 RealESRGANer — 核心推理引擎

这是整个项目最重要的推理类，所有推理脚本都通过它进行图像超分辨率：

**构造函数参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `scale` | int | - | 上采样倍数 |
| `model_path` | str | - | 模型权重路径或URL |
| `dni_weight` | list | None | 深度网络插值权重 |
| `model` | nn.Module | None | 网络架构实例 |
| `tile` | int | 0 | 分块大小 (0=不分块) |
| `tile_pad` | int | 10 | 分块间填充像素 |
| `pre_pad` | int | 10 | 边界预填充像素 |
| `half` | bool | False | 是否使用半精度 (FP16) |
| `device` | torch.device | None | 推理设备 |
| `gpu_id` | int | None | GPU ID |

**核心方法**:

```python
class RealESRGANer:
    def dni(net_a, net_b, dni_weight):
        """深度网络插值: 混合两个模型的权重"""
        # net_interp = (1 - weight) × net_a + weight × net_b

    def pre_process(img):
        """预处理: numpy→tensor, 预填充, 模数填充"""
        # 1. HWC→CHW, BGR→RGB, numpy→tensor
        # 2. 反射填充至 scale 的倍数

    def process():
        """标准推理: output = model(input)"""

    def tile_process():
        """分块推理: 将大图分成小块分别处理"""
        # 1. 将图像划分为 tile×tile 的网格
        # 2. 每块加上 tile_pad 填充
        # 3. 分别推理每块
        # 4. 加权融合到完整输出图像

    def post_process():
        """后处理: 移除填充"""

    def enhance(img, outscale=None, alpha_upsampler='realesrgan'):
        """主入口: 完整的图像增强流程"""
        # 支持: 灰度/RGB/RGBA, 8-bit/16-bit/32-bit
```

**`enhance()` 方法详细流程**:

```
输入图像 (numpy, HWC, BGR, uint8/uint16/float32)
    │
    ├── 1. 检测图像格式 (灰度/RGB/RGBA, 位深度)
    ├── 2. 分离 Alpha 通道 (如果是 RGBA)
    ├── 3. 灰度图转为3通道
    │
    ├── 4. pre_process()
    │      ├── numpy → torch.Tensor
    │      ├── 归一化到 [0, 1]
    │      └── 反射填充
    │
    ├── 5. process() 或 tile_process()
    │      └── 通过模型前向传播
    │
    ├── 6. post_process()
    │      └── 移除填充, tensor → numpy
    │
    ├── 7. 处理 Alpha 通道
    │      ├── 'realesrgan': 用模型上采样 Alpha
    │      └── 'bicubic': 用双三次插值上采样 Alpha
    │
    ├── 8. 合并通道, 恢复位深度
    └── 9. 输出缩放到 outscale
```

#### 5.4.2 PrefetchReader — 预取读取器

后台线程预读取图像列表，使用队列缓冲：

```python
class PrefetchReader(threading.Thread):
    # 后台线程从磁盘读取图像，放入队列
    # 主线程从队列取图像处理
    # 实现流水线并行，减少 I/O 等待
```

#### 5.4.3 IOConsumer — 写入消费者

后台线程负责将处理完的图像写入磁盘：

```python
class IOConsumer(threading.Thread):
    # 从输出队列取图像和路径
    # 后台调用 cv2.imwrite() 保存
    # 主线程可以继续处理下一张图像
```

---

### 5.5 训练入口 (train.py)

**文件**: `realesrgan/train.py`

训练入口极其精简，仅 12 行代码：

```python
import os.path as osp
from basicsr.train import train_pipeline
import realesrgan.archs      # 触发架构注册
import realesrgan.data        # 触发数据集注册
import realesrgan.models      # 触发模型注册

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
```

**设计理念**: 所有训练逻辑委托给 BasicSR 的 `train_pipeline()`，Real-ESRGAN 只负责注册自定义组件（架构、数据集、模型），通过 YAML 配置文件指定使用哪些组件。

---

## 6. 推理脚本详解

### 6.1 图像超分辨率推理

**文件**: `inference_realesrgan.py`

**命令行参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-i, --input` | `'inputs'` | 输入图像文件或目录 |
| `-n, --model_name` | `'RealESRGAN_x4plus'` | 模型名称 |
| `-o, --output` | `'results'` | 输出目录 |
| `-dn, --denoise_strength` | `0.5` | 降噪强度 (0-1) |
| `-s, --outscale` | `4` | 输出放大倍数 |
| `--model_path` | `None` | 自定义模型路径 |
| `-t, --tile` | `0` | 分块大小 |
| `--face_enhance` | `False` | 启用人脸增强 |
| `--fp32` | `False` | 使用全精度 |
| `--ext` | `'auto'` | 输出格式 |
| `-g, --gpu-id` | `None` | GPU ID |

**支持的模型**:

| 模型名称 | 架构 | 倍数 | 用途 |
|----------|------|------|------|
| `RealESRGAN_x4plus` | RRDBNet(64, 23) | 4x | 通用图像 |
| `RealESRNet_x4plus` | RRDBNet(64, 23) | 4x | 通用图像(无GAN) |
| `RealESRGAN_x4plus_anime_6B` | RRDBNet(64, 6) | 4x | 动漫图像 |
| `RealESRGAN_x2plus` | RRDBNet(64, 23) | 2x | 通用图像 |
| `realesr-animevideov3` | SRVGGNetCompact(64, 16) | 4x | 动漫视频 |
| `realesr-general-x4v3` | SRVGGNetCompact(64, 16) | 4x | 通用(支持DNI) |

**`realesr-general-x4v3` 的降噪强度机制**:

当 `denoise_strength < 1` 时，使用深度网络插值（DNI）混合两个模型：
```python
dni_weight = [denoise_strength, 1 - denoise_strength]
# 模型 = denoise_strength × 降噪模型 + (1 - denoise_strength) × 标准模型
```

### 6.2 视频超分辨率推理

**文件**: `inference_realesrgan_video.py`

在图像推理基础上增加了视频处理能力：

**额外参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--fps` | `None` | 输出视频帧率 (自动检测) |
| `--ffmpeg_bin` | `'ffmpeg'` | ffmpeg 路径 |
| `--extract_frame_first` | `False` | 先提取帧再处理 |
| `--num_process_per_gpu` | `1` | 每GPU并行进程数 |

**多GPU并行处理流程**:

```
视频文件
    │
    ├── get_video_meta_info() ──── 获取视频元数据 (分辨率, 帧率, 帧数, 音频)
    │
    ├── 单GPU模式:
    │   ├── Reader (ffmpeg管道/图像序列) → 逐帧读取
    │   ├── RealESRGANer.enhance() → 逐帧处理
    │   └── Writer (ffmpeg管道) → 编码输出 (保留音频)
    │
    └── 多GPU模式:
        ├── get_sub_video() → 按帧数分割视频片段
        ├── multiprocessing.Pool → 并行处理各片段
        └── ffmpeg concat → 合并输出片段 + 原始音频
```

**关键类**:
- **`Reader`**: 视频帧读取器，通过 ffmpeg 管道读取原始帧数据
- **`Writer`**: 视频帧编码器，通过 ffmpeg 管道写入处理后的帧

### 6.3 Cog 云端推理 API

**文件**: `cog_predict.py`

Replicate 平台的容器化推理服务：

```python
class Predictor(BasePredictor):
    def setup(self):
        """容器启动时下载所有模型权重"""

    def choose_model(self, version, tile):
        """根据版本参数选择模型"""

    def predict(self, img, version, scale, face_enhance, tile) -> Path:
        """处理单张图像并返回结果路径"""
```

**云端模型版本**:
- `'General - RealESRGANplus'` → RealESRGAN_x4plus
- `'General - v3'` → realesr-general-x4v3
- `'Anime - anime6B'` → RealESRGAN_x4plus_anime_6B
- `'AnimeVideo - v3'` → realesr-animevideov3

---

## 7. 训练流程详解

### 7.1 数据准备流程

```
原始高清图像 (DF2K, OST等)
    │
    ├── scripts/generate_multiscale_DF2K.py
    │   └── 生成多尺度版本 (0.75x, 0.5x, 1/3x, 短边400)
    │
    ├── scripts/extract_subimages.py
    │   └── 裁切为 480×480 子图 (步长240, 重叠裁切)
    │
    ├── scripts/generate_meta_info.py
    │   └── 生成 meta_info.txt (图像路径列表)
    │
    └── 可选: LMDB 打包 (BasicSR工具)
        └── 将图像打包为 LMDB 数据库加速 I/O
```

### 7.2 退化模型（Degradation Model）

Real-ESRGAN 的核心创新是**二阶退化模型**（Second-Order Degradation Model），在 `feed_data()` 中实现：

```
高清GT图像
    │
    ├── USM锐化 (Unsharp Mask)
    │
    ▼ ══════ 第一阶段退化 ══════
    │
    ├── 1. 模糊 (Blur)
    │   └── filter2D(img, kernel1)
    │       kernel1: iso/aniso/generalized/plateau/sinc 核
    │
    ├── 2. 随机缩放 (Resize)
    │   ├── 上采样概率: resize_prob[0]
    │   ├── 下采样概率: resize_prob[1]
    │   └── 保持概率: resize_prob[2]
    │   插值方式: area / bilinear / bicubic (随机选择)
    │   缩放范围: resize_range (如 [0.15, 1.5])
    │
    ├── 3. 噪声 (Noise)
    │   ├── 高斯噪声概率: gaussian_noise_prob
    │   │   └── σ ∈ noise_range, 灰度噪声概率: gray_noise_prob
    │   └── 泊松噪声概率: 1 - gaussian_noise_prob
    │       └── scale ∈ poisson_scale_range
    │
    └── 4. JPEG压缩
        └── quality ∈ jpeg_range (如 [30, 95])
    │
    ▼ ══════ 第二阶段退化 ══════
    │
    ├── 1. 模糊 (可选, 概率: second_blur_prob)
    │   └── filter2D(img, kernel2)
    │
    ├── 2. 随机缩放 (独立参数)
    │   └── resize_range2, resize_prob2
    │
    ├── 3. 噪声 (独立参数)
    │   └── noise_range2, gray_noise_prob2
    │
    └── 4. 最终退化 (两种顺序随机选择, 各50%概率):
        │
        ├── 顺序A: 缩放到目标尺寸 → sinc滤波 → JPEG压缩
        │
        └── 顺序B: JPEG压缩 → 缩放到目标尺寸 → sinc滤波
    │
    ▼
    裁切为 gt_size/scale 大小的 LQ 图像
```

**退化模型的设计理念**:
- 单阶段退化无法模拟真实世界中的复杂退化
- 二阶段退化可以产生更丰富的退化组合
- 随机化退化顺序增加数据多样性
- sinc 滤波模拟常见的振铃伪影（ringing artifacts）

### 7.3 训练配置文件

项目提供 6 种训练配置：

| 配置文件 | 模型 | 倍数 | 迭代次数 | 特点 |
|----------|------|------|----------|------|
| `train_realesrnet_x4plus.yml` | RealESRNet | 4x | 1000K | 第一阶段: L1训练 |
| `train_realesrnet_x2plus.yml` | RealESRNet | 2x | 1000K | 第一阶段: L1训练 |
| `train_realesrgan_x4plus.yml` | RealESRGAN | 4x | 400K | 第二阶段: GAN训练 |
| `train_realesrgan_x2plus.yml` | RealESRGAN | 2x | 400K | 第二阶段: GAN训练 |
| `finetune_realesrgan_x4plus.yml` | RealESRGAN | 4x | - | 微调 |
| `finetune_realesrgan_x4plus_pairdata.yml` | RealESRGAN | 4x | - | 配对数据微调 |

### 7.4 训练策略

Real-ESRGAN 采用**两阶段训练策略**:

```
阶段一: RealESRNet 训练 (1000K 迭代)
    ├── 模型: RealESRNetModel (继承 SRModel)
    ├── 损失: L1 像素损失
    ├── 学习率: 2e-4 (余弦退火)
    ├── 数据: 在线合成退化 (RealESRGANDataset)
    └── 目的: 学习基础的超分辨率映射

        │
        ▼ 用阶段一的权重初始化阶段二的生成器

阶段二: RealESRGAN 训练 (400K 迭代)
    ├── 模型: RealESRGANModel (继承 SRGANModel)
    ├── 生成器损失: L1 + 感知损失 + GAN损失
    ├── 判别器: UNetDiscriminatorSN
    ├── 学习率: 1e-4 (MultiStepLR)
    ├── 数据: 同样在线合成退化
    └── 目的: 提升视觉质量，产生更锐利、更自然的细节
```

---

## 8. API 接口详解

### 8.1 核心推理 API

```python
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# 1. 创建模型架构
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4)

# 2. 创建推理引擎
upsampler = RealESRGANer(
    scale=4,
    model_path='weights/RealESRGAN_x4plus.pth',
    model=model,
    tile=0,          # 0=不分块, >0=分块处理大图
    tile_pad=10,     # 分块填充
    pre_pad=10,      # 边界填充
    half=True,       # FP16加速
    gpu_id=0
)

# 3. 图像增强
import cv2
img = cv2.imread('input.jpg', cv2.IMREAD_UNCHANGED)
output, img_mode = upsampler.enhance(img, outscale=4)
cv2.imwrite('output.png', output)
```

### 8.2 带人脸增强的 API

```python
from gfpgan import GFPGANer

# 创建人脸增强器 (内部使用 RealESRGANer 作为背景上采样器)
face_enhancer = GFPGANer(
    model_path='weights/GFPGANv1.4.pth',
    upscale=4,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=upsampler  # Real-ESRGAN 作为背景上采样器
)

# 增强 (同时处理人脸和背景)
_, _, output = face_enhancer.enhance(
    img, has_aligned=False, only_center_face=False, paste_back=True
)
```

### 8.3 深度网络插值 (DNI) API

```python
# 混合两个模型以控制降噪强度
upsampler = RealESRGANer(
    scale=4,
    model_path=['model_denoise.pth', 'model_standard.pth'],
    dni_weight=[0.5, 0.5],  # 各50%权重混合
    model=model
)
```

### 8.4 训练 API

```bash
# 启动训练
python -m realesrgan.train -opt options/train_realesrgan_x4plus.yml

# 恢复训练
python -m realesrgan.train -opt options/train_realesrgan_x4plus.yml \
    --auto_resume
```

---

## 9. C++ API 详解 (NCNN-Vulkan)

### 9.1 概述与架构

Real-ESRGAN 官方提供了基于 [NCNN](https://github.com/Tencent/ncnn) 框架的 C++ 实现，位于独立仓库 [Real-ESRGAN-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan)。该实现使用 Vulkan GPU 加速，支持 Windows、Linux、macOS 三大平台，**无需安装 CUDA 或 PyTorch 环境**。

**C++ 实现的技术栈**:

```
┌──────────────────────────────────┐
│     应用层: main.cpp              │
│     (命令行工具, 多线程任务调度)     │
├──────────────────────────────────┤
│     核心层: realesrgan.h/cpp      │
│     (模型加载, 分块推理, 前后处理)   │
├──────────────────────────────────┤
│     框架层: NCNN                  │
│     (神经网络推理, Vulkan GPU加速)  │
├──────────────────────────────────┤
│     图像I/O: stb_image / WIC      │
│     (跨平台图像编解码)              │
└──────────────────────────────────┘
```

**与 Python 版本的对应关系**:

| Python (本仓库) | C++ (ncnn-vulkan) |
|-----------------|-------------------|
| `RealESRGANer` 类 | `RealESRGAN` 类 |
| `RealESRGANer.enhance()` | `RealESRGAN::process()` |
| `RealESRGANer.tile_process()` | 内置于 `process()` 中 |
| `.pth` 模型权重 | `.param` + `.bin` NCNN模型 |
| PyTorch + CUDA | NCNN + Vulkan |

### 9.2 C++ 类接口

核心 C++ 类定义于 `src/realesrgan.h`：

```cpp
#include "net.h"    // ncnn
#include "gpu.h"    // ncnn
#include "layer.h"  // ncnn

class RealESRGAN
{
public:
    // 构造函数
    // gpuid: GPU设备ID (-1表示CPU)
    // tta_mode: 是否启用TTA (Test-Time Augmentation) 模式
    RealESRGAN(int gpuid, bool tta_mode = false);
    ~RealESRGAN();

    // 加载NCNN模型
    // parampath: .param 模型结构文件路径
    // modelpath: .bin 模型权重文件路径
    // 返回: 0=成功, 非0=失败
    int load(const std::string& parampath, const std::string& modelpath);

    // 图像超分辨率处理
    // inimage: 输入图像 (ncnn::Mat, 像素格式 RGB, 类型 PIXEL)
    // outimage: 输出图像 (ncnn::Mat, 自动分配)
    // 返回: 0=成功, 非0=失败
    int process(const ncnn::Mat& inimage, ncnn::Mat& outimage) const;

public:
    int scale;       // 上采样倍数 (2/3/4)
    int tilesize;    // 分块大小 (0=自动)
    int prepadding;  // 预填充像素数
};
```

**关键公共属性**:

| 属性 | 类型 | 说明 |
|------|------|------|
| `scale` | int | 上采样倍数，支持 2/3/4 |
| `tilesize` | int | 分块大小，0 表示自动选择，减小此值可降低显存占用 |
| `prepadding` | int | 预填充像素，避免边界伪影 |

### 9.3 C++ 图像推理完整示例

以下是使用 Real-ESRGAN C++ API 进行图像超分辨率的完整示例：

```cpp
#include <stdio.h>
#include "gpu.h"
#include "realesrgan.h"

// 使用 stb_image 进行图像读写
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int argc, char** argv)
{
    const char* input_path = "input.jpg";
    const char* output_path = "output.png";
    const char* param_path = "models/realesrgan-x4plus.param";
    const char* model_path = "models/realesrgan-x4plus.bin";
    int scale = 4;

    // =============================================
    // 第1步: 初始化 Vulkan GPU 环境
    // =============================================
    ncnn::create_gpu_instance();

    // =============================================
    // 第2步: 创建 RealESRGAN 实例
    // =============================================
    int gpuid = ncnn::get_default_gpu_index();
    bool tta_mode = false;
    RealESRGAN realesrgan(gpuid, tta_mode);

    // 设置参数
    realesrgan.scale = scale;
    realesrgan.tilesize = 0;     // 0=自动, 设为具体值可控制显存
    realesrgan.prepadding = 10;  // 预填充像素

    // =============================================
    // 第3步: 加载 NCNN 模型
    // =============================================
    int ret = realesrgan.load(param_path, model_path);
    if (ret != 0) {
        fprintf(stderr, "模型加载失败! 返回码: %d\n", ret);
        ncnn::destroy_gpu_instance();
        return -1;
    }

    // =============================================
    // 第4步: 读取输入图像
    // =============================================
    int width, height, channels;
    unsigned char* pixeldata = stbi_load(input_path, &width, &height, &channels, 3);
    if (!pixeldata) {
        fprintf(stderr, "无法读取图像: %s\n", input_path);
        ncnn::destroy_gpu_instance();
        return -1;
    }

    // 创建 ncnn::Mat (从像素数据, RGB格式)
    ncnn::Mat inimage = ncnn::Mat::from_pixels(pixeldata, ncnn::Mat::PIXEL_RGB,
                                                width, height);

    // =============================================
    // 第5步: 执行超分辨率推理
    // =============================================
    ncnn::Mat outimage;
    ret = realesrgan.process(inimage, outimage);
    if (ret != 0) {
        fprintf(stderr, "推理失败! 返回码: %d\n", ret);
        stbi_image_free(pixeldata);
        ncnn::destroy_gpu_instance();
        return -1;
    }

    // =============================================
    // 第6步: 保存输出图像
    // =============================================
    int out_w = outimage.w;
    int out_h = outimage.h;

    // 将 ncnn::Mat 转为像素数据
    unsigned char* outpixeldata = new unsigned char[out_w * out_h * 3];
    outimage.to_pixels(outpixeldata, ncnn::Mat::PIXEL_RGB);

    // 写入 PNG 文件
    stbi_write_png(output_path, out_w, out_h, 3, outpixeldata, 0);
    printf("超分辨率完成: %dx%d → %dx%d\n", width, height, out_w, out_h);
    printf("输出已保存至: %s\n", output_path);

    // =============================================
    // 第7步: 清理资源
    // =============================================
    delete[] outpixeldata;
    stbi_image_free(pixeldata);
    ncnn::destroy_gpu_instance();

    return 0;
}
```

### 9.4 编译与构建

**前置依赖**: CMake ≥ 3.9, Vulkan SDK, C++17 编译器

```bash
# 克隆仓库 (含NCNN子模块)
git clone https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan.git
cd Real-ESRGAN-ncnn-vulkan
git submodule update --init --recursive

# 构建
mkdir build && cd build
cmake ../src
cmake --build . -j$(nproc)
```

**模型转换** (PyTorch → NCNN):

```bash
# 第1步: PyTorch → ONNX
python scripts/pytorch2onnx.py --input RealESRGAN_x4plus.pth --output realesrgan-x4.onnx

# 第2步: ONNX → NCNN
onnx2ncnn realesrgan-x4.onnx realesrgan-x4-raw.param realesrgan-x4-raw.bin

# 第3步: 优化 NCNN 模型 (FP16)
ncnnoptimize realesrgan-x4-raw.param realesrgan-x4-raw.bin \
    realesrgan-x4.param realesrgan-x4.bin 1
```

### 9.5 命令行工具使用

编译完成后可直接使用命令行工具：

```bash
# 基本用法: 4x 超分辨率
./realesrgan-ncnn-vulkan -i input.jpg -o output.png -n realesrgan-x4plus

# 指定上采样倍数
./realesrgan-ncnn-vulkan -i input.jpg -o output.png -n realesr-animevideov3 -s 2

# 批量处理整个目录
./realesrgan-ncnn-vulkan -i input_folder/ -o output_folder/ -n realesrgan-x4plus

# 使用特定GPU, 自定义分块大小和线程数
./realesrgan-ncnn-vulkan -i input.jpg -o output.png \
    -g 0 -t 128 -j 2:4:2 -n realesrgan-x4plus

# 多GPU并行处理
./realesrgan-ncnn-vulkan -i input.jpg -o output.png \
    -g 0,1,2 -t 128,128,128 -j 2:4,4,4:2

# 启用 TTA 模式 (更高质量, 8x 推理时间)
./realesrgan-ncnn-vulkan -i input.jpg -o output.png -x -n realesrgan-x4plus
```

**命令行参数详解**:

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-i` | 输入图像路径或目录 | 必填 |
| `-o` | 输出图像路径或目录 | 必填 |
| `-s` | 上采样倍数 (2/3/4) | 4 |
| `-t` | 分块大小 (≥32, 0=自动) | 0 |
| `-m` | 模型文件夹路径 | `models` |
| `-n` | 模型名称 | `realesr-animevideov3` |
| `-g` | GPU设备ID (支持多GPU) | auto |
| `-j` | 线程数 load:proc:save | `1:2:2` |
| `-x` | 启用TTA模式 | 关闭 |
| `-f` | 输出格式 (jpg/png/webp) | ext/png |

**支持的模型**:

| 模型名称 | 说明 |
|----------|------|
| `realesrgan-x4plus` | 通用图像 4x 超分 (默认) |
| `realesrnet-x4plus` | 通用图像 4x (无GAN) |
| `realesrgan-x4plus-anime` | 动漫图像优化 |
| `realesr-animevideov3` | 动漫视频优化 (轻量快速) |

---

## 10. Java/Android API 详解 (NCNN-Android)

### 10.1 概述与架构

社区项目 [RealSR-NCNN-Android](https://github.com/tumuyan/RealSR-NCNN-Android) 将 Real-ESRGAN 的 NCNN C++ 实现封装为 Android 应用，通过 JNI（Java Native Interface）桥接 Java/Kotlin 层与 C++ 推理层。

**Android 实现的技术架构**:

```
┌────────────────────────────────────────────┐
│     Java/Kotlin 层 (Android UI)             │
│     ├── Activity / Fragment                 │
│     ├── Bitmap 图像对象                      │
│     └── 调用 JNI native 方法                 │
├────────────────────────────────────────────┤
│     JNI 桥接层 (C/C++)                      │
│     ├── Java_..._RealESRGAN_process()       │
│     ├── Bitmap ↔ ncnn::Mat 转换             │
│     └── 生命周期管理 (init/process/destroy)  │
├────────────────────────────────────────────┤
│     NCNN C++ 推理层                         │
│     ├── RealESRGAN 类                       │
│     ├── 分块推理 + Vulkan GPU 加速           │
│     └── .param + .bin 模型文件              │
├────────────────────────────────────────────┤
│     Android 平台层                          │
│     ├── Vulkan GPU 驱动                     │
│     └── NDK / CMake 构建系统                 │
└────────────────────────────────────────────┘
```

### 10.2 JNI 桥接层

Android 端通过 JNI 将 C++ 的 `RealESRGAN` 类暴露给 Java 层。典型的 JNI 接口设计如下：

```java
public class RealESRGAN {
    // 加载 native 库
    static {
        System.loadLibrary("realesrgan-ncnn-vulkan");
    }

    // Native 方法声明
    // 初始化模型
    // modelDir: 模型文件目录 (包含 .param 和 .bin 文件)
    // modelName: 模型名称 (如 "realesrgan-x4plus")
    // gpuId: GPU设备ID (-1=CPU)
    // scale: 上采样倍数
    // tileSize: 分块大小 (0=自动)
    public native boolean init(String modelDir, String modelName,
                               int gpuId, int scale, int tileSize);

    // 执行超分辨率处理
    // inputBitmap: 输入 Android Bitmap
    // 返回: 超分辨率后的 Bitmap
    public native Bitmap process(Bitmap inputBitmap);

    // 释放资源
    public native void destroy();
}
```

**对应的 C++ JNI 实现** (`realesrgan_jni.cpp`):

```cpp
#include <jni.h>
#include <android/bitmap.h>
#include "realesrgan.h"
#include "gpu.h"

static RealESRGAN* realesrgan = nullptr;  // 注意: 简化示例, 生产环境应使用线程安全的实例管理

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_realesrgan_RealESRGAN_init(
    JNIEnv* env, jobject thiz,
    jstring modelDir, jstring modelName,
    jint gpuId, jint scale, jint tileSize)
{
    const char* model_dir = env->GetStringUTFChars(modelDir, nullptr);
    const char* model_name = env->GetStringUTFChars(modelName, nullptr);

    // 初始化 Vulkan GPU
    ncnn::create_gpu_instance();

    // 创建 RealESRGAN 实例
    realesrgan = new RealESRGAN(gpuId, false);
    realesrgan->scale = scale;
    realesrgan->tilesize = tileSize;
    realesrgan->prepadding = 10;

    // 构建模型路径并加载
    std::string param_path = std::string(model_dir) + "/" + model_name + ".param";
    std::string bin_path = std::string(model_dir) + "/" + model_name + ".bin";
    int ret = realesrgan->load(param_path, bin_path);

    env->ReleaseStringUTFChars(modelDir, model_dir);
    env->ReleaseStringUTFChars(modelName, model_name);

    return ret == 0;
}

extern "C"
JNIEXPORT jobject JNICALL
Java_com_example_realesrgan_RealESRGAN_process(
    JNIEnv* env, jobject thiz, jobject inputBitmap)
{
    // 获取输入 Bitmap 信息
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, inputBitmap, &info);

    // 锁定像素数据
    void* pixels;
    AndroidBitmap_lockPixels(env, inputBitmap, &pixels);

    // 转换为 ncnn::Mat (RGBA → RGB)
    ncnn::Mat inimage = ncnn::Mat::from_pixels(
        (const unsigned char*)pixels,
        ncnn::Mat::PIXEL_RGBA2RGB,
        info.width, info.height);

    AndroidBitmap_unlockPixels(env, inputBitmap);

    // 执行超分辨率推理
    ncnn::Mat outimage;
    realesrgan->process(inimage, outimage);

    // 创建输出 Bitmap
    jclass bitmapClass = env->FindClass("android/graphics/Bitmap");
    jmethodID createBitmap = env->GetStaticMethodID(bitmapClass, "createBitmap",
        "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");
    jclass configClass = env->FindClass("android/graphics/Bitmap$Config");
    jfieldID argb8888 = env->GetStaticFieldID(configClass, "ARGB_8888",
        "Landroid/graphics/Bitmap$Config;");
    jobject config = env->GetStaticObjectField(configClass, argb8888);

    jobject outputBitmap = env->CallStaticObjectMethod(bitmapClass, createBitmap,
        outimage.w, outimage.h, config);

    // 写入像素数据
    void* outPixels;
    AndroidBitmap_lockPixels(env, outputBitmap, &outPixels);
    outimage.to_pixels((unsigned char*)outPixels, ncnn::Mat::PIXEL_RGB2RGBA);
    AndroidBitmap_unlockPixels(env, outputBitmap);

    return outputBitmap;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_realesrgan_RealESRGAN_destroy(JNIEnv* env, jobject thiz)
{
    delete realesrgan;
    realesrgan = nullptr;
    ncnn::destroy_gpu_instance();
}
```

### 10.3 Java/Android 图像推理完整示例

以下是在 Android Activity 中使用 Real-ESRGAN 进行图像超分辨率的完整示例：

```java
import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;

public class SuperResolutionActivity extends Activity {

    private RealESRGAN realesrgan;
    private ImageView imageView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView = findViewById(R.id.imageView);

        // =============================================
        // 第1步: 初始化 RealESRGAN
        // =============================================
        realesrgan = new RealESRGAN();

        // 将模型文件从 assets 复制到应用内部存储
        String modelDir = getFilesDir().getAbsolutePath() + "/models";
        copyAssetsToDir("models", modelDir);

        // 初始化模型
        boolean success = realesrgan.init(
            modelDir,                      // 模型目录
            "realesrgan-x4plus",           // 模型名称
            0,                             // GPU ID (0=默认GPU)
            4,                             // 上采样倍数
            0                              // 分块大小 (0=自动)
        );

        if (!success) {
            Log.e("RealESRGAN", "模型初始化失败!");
            return;
        }

        // =============================================
        // 第2步: 加载输入图像
        // =============================================
        Bitmap inputBitmap = BitmapFactory.decodeResource(
            getResources(), R.drawable.input_image);

        // =============================================
        // 第3步: 在后台线程执行超分辨率 (避免阻塞UI)
        // =============================================
        new AsyncTask<Bitmap, Void, Bitmap>() {
            @Override
            protected Bitmap doInBackground(Bitmap... bitmaps) {
                // 执行超分辨率推理
                return realesrgan.process(bitmaps[0]);
            }

            @Override
            protected void onPostExecute(Bitmap result) {
                // 在UI线程显示结果
                if (result != null) {
                    imageView.setImageBitmap(result);
                    Log.i("RealESRGAN", String.format(
                        "超分完成: %dx%d → %dx%d",
                        inputBitmap.getWidth(), inputBitmap.getHeight(),
                        result.getWidth(), result.getHeight()));
                }
            }
        }.execute(inputBitmap);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // =============================================
        // 第4步: 释放资源
        // =============================================
        if (realesrgan != null) {
            realesrgan.destroy();
        }
    }
}
```

### 10.4 Android 项目集成指南

**1. Gradle 构建配置** (`app/build.gradle`):

```groovy
android {
    defaultConfig {
        ndk {
            abiFilters 'arm64-v8a', 'armeabi-v7a'  // 支持的CPU架构
        }
        externalNativeBuild {
            cmake {
                arguments '-DNCNN_VULKAN=ON'         // 启用Vulkan加速
            }
        }
    }
    externalNativeBuild {
        cmake {
            path 'src/main/jni/CMakeLists.txt'
        }
    }
}
```

**2. CMakeLists.txt** (`app/src/main/jni/CMakeLists.txt`):

```cmake
cmake_minimum_required(VERSION 3.10)
project(realesrgan-ncnn-vulkan)

# NCNN 预编译库
set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

# 编译 Real-ESRGAN JNI 库
add_library(realesrgan-ncnn-vulkan SHARED
    realesrgan_jni.cpp
    realesrgan.cpp
)

target_link_libraries(realesrgan-ncnn-vulkan
    ncnn
    android
    jnigraphics    # Android Bitmap JNI
    vulkan
    log
)
```

**3. 模型文件放置**:

将 NCNN 模型文件放入 `app/src/main/assets/models/` 目录：
```
assets/
└── models/
    ├── realesrgan-x4plus.param     # 模型结构
    └── realesrgan-x4plus.bin       # 模型权重
```

---

## 11. ONNX 导出与跨平台部署

Real-ESRGAN 通过 ONNX 格式实现从 Python/PyTorch 到 C++/移动端的模型迁移：

**转换流程**:

```
PyTorch (.pth)  →  ONNX (.onnx)  →  NCNN (.param + .bin)
                                  →  TensorRT (.engine)
                                  →  CoreML (.mlmodel)
                                  →  其他推理框架
```

**ONNX 导出脚本** (`scripts/pytorch2onnx.py`):

```python
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet

# 创建模型并加载权重
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4)
model.load_state_dict(torch.load('RealESRGAN_x4plus.pth')['params_ema'])
model.eval()

# 导出 ONNX
x = torch.rand(1, 3, 64, 64)
with torch.no_grad():
    torch.onnx.export(model, x, 'realesrgan-x4.onnx',
                      opset_version=11, export_params=True)
```

**完整的跨平台部署路径**:

| 目标平台 | 推理框架 | 模型格式 | 仓库/工具 |
|----------|----------|----------|-----------|
| Windows/Linux/macOS (GPU) | NCNN + Vulkan | `.param` + `.bin` | [Real-ESRGAN-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan) |
| Android (GPU) | NCNN + Vulkan | `.param` + `.bin` | [RealSR-NCNN-Android](https://github.com/tumuyan/RealSR-NCNN-Android) |
| 通用 (GPU) | ONNX Runtime | `.onnx` | `scripts/pytorch2onnx.py` |
| NVIDIA GPU | TensorRT | `.engine` | ONNX → TensorRT |
| Apple 设备 | CoreML | `.mlmodel` | ONNX → CoreML |

---

## 12. 调用流程图

### 12.1 推理调用流程

```
用户命令: python inference_realesrgan.py -i input.jpg -s 4
    │
    ├── 解析命令行参数
    ├── 根据 model_name 确定网络架构和模型路径
    │   ├── 'RealESRGAN_x4plus' → RRDBNet(64, 23, scale=4)
    │   ├── 'realesr-animevideov3' → SRVGGNetCompact(64, 16, scale=4)
    │   └── ...
    │
    ├── 自动下载模型权重 (如果本地不存在)
    │   └── load_file_from_url() → ~/.cache/realesrgan/
    │
    ├── 创建 RealESRGANer 实例
    │   └── __init__():
    │       ├── 加载模型权重 (支持 params / params_ema 键)
    │       ├── 模型 → GPU
    │       └── 可选: 半精度转换
    │
    ├── [可选] 创建 GFPGANer 人脸增强器
    │
    ├── 遍历输入图像:
    │   ├── cv2.imread(path, IMREAD_UNCHANGED)
    │   │
    │   ├── upsampler.enhance(img, outscale=4):
    │   │   ├── 检测图像格式 (灰度/RGB/RGBA)
    │   │   ├── pre_process():
    │   │   │   ├── numpy → tensor
    │   │   │   ├── 预填充 (pre_pad)
    │   │   │   └── 模数填充 (确保尺寸是 scale 的倍数)
    │   │   │
    │   │   ├── process() 或 tile_process():
    │   │   │   └── output = model(input)  # PyTorch 前向传播
    │   │   │
    │   │   ├── post_process():
    │   │   │   ├── 移除填充
    │   │   │   └── tensor → numpy, clamp [0, 1]
    │   │   │
    │   │   ├── 处理 Alpha 通道
    │   │   └── 缩放到目标尺寸 (outscale)
    │   │
    │   └── cv2.imwrite(output_path, output)
    │
    └── 完成
```

### 12.2 训练调用流程

```
用户命令: python -m realesrgan.train -opt options/train_realesrgan_x4plus.yml
    │
    ├── realesrgan/train.py
    │   ├── import realesrgan.archs   → 注册: SRVGGNetCompact, UNetDiscriminatorSN
    │   ├── import realesrgan.data    → 注册: RealESRGANDataset, RealESRGANPairedDataset
    │   ├── import realesrgan.models  → 注册: RealESRGANModel, RealESRNetModel
    │   └── train_pipeline(root_path) → 调用 BasicSR 训练管线
    │
    ├── BasicSR train_pipeline():
    │   ├── 解析 YAML 配置
    │   ├── 创建数据集 (根据配置中的 type 从 DATASET_REGISTRY 获取)
    │   │   └── RealESRGANDataset / RealESRGANPairedDataset
    │   ├── 创建模型 (根据配置中的 type 从 MODEL_REGISTRY 获取)
    │   │   └── RealESRGANModel / RealESRNetModel
    │   ├── 加载预训练权重
    │   │
    │   └── 训练循环:
    │       for iter in range(total_iters):
    │           │
    │           ├── data = dataset[idx]
    │           │   └── RealESRGANDataset.__getitem__():
    │           │       ├── 加载 GT 图像
    │           │       ├── 数据增强 (翻转, 旋转)
    │           │       └── 生成退化核 (kernel1, kernel2, sinc_kernel)
    │           │
    │           ├── model.feed_data(data)
    │           │   └── 在线合成退化:
    │           │       ├── GT → USM锐化
    │           │       ├── 第一阶段: 模糊 → 缩放 → 噪声 → JPEG
    │           │       ├── 第二阶段: 模糊 → 缩放 → 噪声 → sinc/JPEG
    │           │       ├── 裁切为 GT-LQ 对
    │           │       └── 放入/取出训练队列
    │           │
    │           ├── model.optimize_parameters(iter)
    │           │   ├── 生成器优化:
    │           │   │   ├── SR = generator(LQ)
    │           │   │   ├── L_pixel = L1(SR, GT)
    │           │   │   ├── L_percep = VGG_Loss(SR, GT)
    │           │   │   ├── L_gan = GAN_Loss(D(SR), True)
    │           │   │   └── 反向传播 + 梯度更新
    │           │   └── 判别器优化:
    │           │       ├── L_real = GAN_Loss(D(GT), True)
    │           │       ├── L_fake = GAN_Loss(D(SR.detach()), False)
    │           │       └── 反向传播 + 梯度更新
    │           │
    │           ├── 更新 EMA (指数移动平均)
    │           ├── 更新学习率
    │           └── 定期: 验证 / 保存检查点 / 日志记录
```

---

## 13. 关键设计模式与工程决策

### 13.1 注册机制 (Registry Pattern)

通过 BasicSR 的注册机制实现组件的松耦合：

```python
# 定义时注册
@ARCH_REGISTRY.register()
class MyNewArch(nn.Module): ...

# 使用时通过配置文件引用
# config.yml:
#   network_g:
#     type: MyNewArch
```

**优势**: 添加新组件无需修改任何现有代码，只需创建文件并注册。

### 13.2 在线退化合成 (Online Degradation Synthesis)

退化图像在训练时在线生成，而非预先准备：

**优势**:
- 无需存储大量退化图像
- 每次迭代产生不同退化组合
- 退化参数可随时调整
- 队列机制进一步增加多样性

### 13.3 分块推理 (Tile-based Inference)

对于大图像，分块处理避免 GPU 显存溢出：

```
┌──────────────────────┐
│  tile  │  tile  │    │
│   1    │   2    │    │
├────────┼────────┤    │
│  tile  │  tile  │    │  ← 每块加 tile_pad 像素重叠
│   3    │   4    │    │
├────────┼────────┤    │
│        │        │    │
└──────────────────────┘
```

每块独立处理后加权融合边界区域，消除接缝。

### 13.4 深度网络插值 (Deep Network Interpolation, DNI)

通过混合两个模型的权重实现连续可调的效果：

```python
# 权重空间线性插值
for k in net_a.keys():
    net_interp[k] = (1 - α) × net_a[k] + α × net_b[k]
```

应用场景: `realesr-general-x4v3` 模型的降噪强度控制。

### 13.5 生产者-消费者模式 (Producer-Consumer Pattern)

视频推理中使用多线程流水线：

```
PrefetchReader (生产者线程)
    ↓ 队列
RealESRGANer.enhance() (主线程处理)
    ↓ 队列
IOConsumer (消费者线程)
```

三个阶段并行执行，最大化吞吐量。

### 13.6 两阶段训练策略

```
阶段一 (PSNR导向):
    RealESRNet + L1 损失 → 高 PSNR, 但可能模糊

阶段二 (感知质量导向):
    RealESRGAN + L1 + 感知 + GAN 损失 → 视觉质量高, 锐利细节
```

这种策略先学习正确的映射关系，再提升视觉质量，比从头 GAN 训练更稳定。

---

## 14. 预训练模型一览

| 模型名称 | 架构 | 倍数 | 参数量 | 用途 |
|----------|------|------|--------|------|
| RealESRGAN_x4plus | RRDBNet(64, 23) | 4x | ~16.7M | 通用图像超分 |
| RealESRGAN_x2plus | RRDBNet(64, 23) | 2x | ~16.7M | 通用图像 2x 超分 |
| RealESRNet_x4plus | RRDBNet(64, 23) | 4x | ~16.7M | 通用图像 (无GAN) |
| RealESRGAN_x4plus_anime_6B | RRDBNet(64, 6) | 4x | ~4.2M | 动漫图像优化 |
| realesr-animevideov3 | SRVGGNetCompact(64, 16) | 4x | ~0.4M | 动漫视频 (轻量快速) |
| realesr-general-x4v3 | SRVGGNetCompact(64, 16) | 4x | ~0.4M | 通用 (支持DNI降噪调节) |

---

## 15. 测试体系

项目包含 4 个测试文件，覆盖核心功能：

### 15.1 test_dataset.py

- **`test_realesrgan_dataset()`**: 测试在线退化数据集
  - 验证磁盘和 LMDB 后端
  - 检查输出张量形状 (3, 400, 400)
  - 验证 6 种核类型的初始化
  - 测试 sinc_prob=0 边界条件

- **`test_realesrgan_paired_dataset()`**: 测试配对数据集
  - 验证 GT/LQ 配对加载
  - 检查归一化参数
  - 验证输出形状 (gt: 3×128×128, lq: 3×32×32)

### 15.2 test_discriminator_arch.py

- **`test_unetdiscriminatorsn()`**: 测试判别器
  - CPU 和 GPU 前向传播
  - 输入 (1,3,32,32) → 输出 (1,1,32,32)

### 15.3 test_model.py

- **`test_realesrnet_model()`**: 测试回归模型
  - 验证组件类型 (RRDBNet, L1Loss, Adam)
  - 测试 feed_data() 退化流水线
  - 测试 nondist_validation

- **`test_realesrgan_model()`**: 测试 GAN 模型
  - 验证 3 种损失类型
  - 测试 optimize_parameters() 输出

### 15.4 test_utils.py

- **`test_realesrganer()`**: 测试推理引擎
  - 测试 pre_process, tile_process, enhance
  - 多格式支持: RGB, 16-bit, 灰度, RGBA
  - 验证 outscale 参数

---

## 16. 总结

### 架构优势

1. **模块化设计**: 通过 BasicSR 注册机制实现高度模块化，各组件松耦合
2. **在线退化合成**: 训练时动态生成退化图像，无需预存储，退化多样性极高
3. **二阶退化模型**: 比单阶段退化更贴近真实世界的复杂退化
4. **灵活部署**: 支持从命令行工具到云端 API 的多种部署方式
5. **工程完善**: 分块推理、多GPU并行、半精度加速等工程优化

### 核心创新

1. **高阶退化建模**: 两阶段退化 + sinc 滤波 + 随机退化顺序
2. **USM 锐化策略**: GT 锐化 + 可配置的损失函数 USM 开关
3. **训练对队列**: 打破批次退化一致性，提升训练稳定性
4. **DNI 降噪控制**: 权重空间插值实现连续可调的降噪效果

### 技术栈总结

```
前端/应用: argparse CLI, Cog API
核心引擎: PyTorch, BasicSR
图像处理: OpenCV, Pillow, numpy
视频处理: ffmpeg (通过子进程)
人脸增强: GFPGAN, facexlib
部署方案: PyTorch 推理, ONNX 导出, NCNN 移动端, Cog 云端
```

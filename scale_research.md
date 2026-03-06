# libheif Scale 功能深度研究报告

## 目录

1. [项目概述](#1-项目概述)
2. [libheif 整体架构](#2-libheif-整体架构)
3. [Scale 功能的当前实现](#3-scale-功能的当前实现)
4. [Scale API 分析](#4-scale-api-分析)
5. [Scale 在项目中的使用现状](#5-scale-在项目中的使用现状)
6. [heif-enc 工具当前的处理流程](#6-heif-enc-工具当前的处理流程)
7. [将 Scale 功能集成到 heif-enc 的方案设计](#7-将-scale-功能集成到-heif-enc-的方案设计)
8. [实现细节与代码示例](#8-实现细节与代码示例)
9. [潜在问题与改进建议](#9-潜在问题与改进建议)
10. [总结](#10-总结)

---

## 1. 项目概述

libheif 是一个符合 ISO/IEC 23008-12 标准的 HEIF（High Efficiency Image File Format）和 AVIF（AV1 Image File Format）文件格式的编解码器库。它支持多种编解码格式，包括 HEVC (H.265)、AV1、VVC、AVC、JPEG、JPEG-2000 和 ISO/IEC 23001-17 等。

项目地址：https://github.com/strukturag/libheif

### 1.1 核心能力

- 支持 HEIC、AVIF、VVC、AVC、JPEG-in-HEIF、JPEG2000、uncompressed 编解码
- Alpha 通道、深度图、缩略图、辅助图像
- 多图像文件支持
- HEIF 图像序列和 MP4 视频
- 分块图像（tiled images）
- HDR 图像与色彩配置
- 图像变换（裁剪、镜像、旋转）
- 插件接口支持动态加载编解码器

### 1.2 项目目录结构

```
libheif/
├── CMakeLists.txt              # 顶层构建配置
├── libheif/                    # 核心库代码
│   ├── api/libheif/            # 公共 C API 头文件和实现
│   │   ├── heif_image.h        # 图像操作 API（含 scale）
│   │   ├── heif_image.cc       # 图像操作 API 实现
│   │   ├── heif_aux_images.h   # 辅助图像 API（含 thumbnail）
│   │   ├── heif_aux_images.cc  # 辅助图像 API 实现
│   │   ├── heif_cxx.h          # C++ 头文件封装
│   │   └── ...
│   ├── pixelimage.h            # 内部像素图像类声明
│   ├── pixelimage.cc           # 内部像素图像类实现（含 scale_nearest_neighbor）
│   ├── context.h               # HeifContext 上下文声明
│   ├── context.cc              # HeifContext 实现（含 encode_thumbnail）
│   ├── image-items/            # 图像项处理
│   │   └── image_item.cc       # 图像项实现（含 alpha 缩放）
│   ├── codecs/                 # 编解码器相关
│   ├── color-conversion/       # 颜色空间转换
│   └── plugins/                # 编解码器插件
├── examples/                   # 示例工具程序
│   ├── heif_enc.cc             # 编码工具（JPEG/PNG → HEIF/AVIF）
│   ├── heif_dec.cc             # 解码工具（HEIF/AVIF → JPEG/PNG）
│   ├── heif_info.cc            # 文件信息查看工具
│   ├── heif_thumbnailer.cc     # 缩略图生成工具（使用了 scale）
│   ├── heif_view.cc            # 图像查看工具
│   └── CMakeLists.txt          # 示例程序构建配置
├── heifio/                     # I/O 相关的编解码辅助代码
├── tests/                      # 测试代码
└── third-party/                # 第三方依赖
```

---

## 2. libheif 整体架构

### 2.1 分层架构

libheif 采用了清晰的分层架构设计：

```
┌─────────────────────────────────────────────────┐
│                 应用层 (examples/)                │
│   heif_enc / heif_dec / heif_thumbnailer / ...   │
├─────────────────────────────────────────────────┤
│              公共 C API 层 (api/libheif/)         │
│  heif_image.h  heif_aux_images.h  heif_cxx.h    │
├─────────────────────────────────────────────────┤
│               核心逻辑层 (libheif/)               │
│  HeifContext / HeifPixelImage / ImageItem / ...   │
├─────────────────────────────────────────────────┤
│              编解码器插件层 (plugins/)             │
│  libde265 / x265 / libaom / dav1d / vvenc / ...  │
└─────────────────────────────────────────────────┘
```

### 2.2 关键类和数据结构

| 类/结构体 | 文件位置 | 职责 |
|-----------|----------|------|
| `heif_image` | api/libheif/heif_image.h | 公共 C API 的图像结构，封装 `HeifPixelImage` |
| `HeifPixelImage` | pixelimage.h/cc | 内部像素图像表示，包含所有通道数据和图像操作方法 |
| `HeifContext` | context.h/cc | HEIF 文件的上下文管理器，负责编解码流程 |
| `ImageItem` | image-items/image_item.h/cc | 图像项抽象，管理单个图像及其属性 |
| `heif_scaling_options` | api/libheif/heif_image.h | 缩放选项结构（当前未定义） |

### 2.3 图像处理流程

**解码流程：**
```
HEIF/AVIF 文件
   → heif_context_read_from_file()
   → heif_context_get_primary_image_handle()
   → heif_decode_image()
   → heif_image（解码后的像素数据）
```

**编码流程：**
```
heif_image（源像素数据）
   → heif_context_encode_image()
   → 编码器插件处理
   → heif_context_write_to_file()
   → HEIF/AVIF 文件
```

---

## 3. Scale 功能的当前实现

### 3.1 核心缩放算法：`HeifPixelImage::scale_nearest_neighbor()`

**文件位置：** `libheif/pixelimage.cc`（约第 1760 行起）

这是 libheif 中唯一的图像缩放实现，采用**最近邻插值（Nearest Neighbor）** 算法。

#### 函数签名

```cpp
// libheif/pixelimage.h（第 514 行）
Error scale_nearest_neighbor(
    std::shared_ptr<HeifPixelImage>& output,
    uint32_t width,
    uint32_t height,
    const heif_security_limits* limits
) const;
```

#### 实现原理

该函数的处理步骤如下：

**步骤 1：创建输出图像并分配通道平面**

根据输入图像的色彩空间（colorspace）和色度格式（chroma），为输出图像创建对应的通道平面：

- **交错格式（interleaved）**：创建单一的 `heif_channel_interleaved` 平面
- **RGB 平面格式**：分别创建 R、G、B 三个平面
- **单色格式（monochrome）**：创建 Y 平面
- **YCbCr 格式**：创建 Y、Cb、Cr 三个平面（Cb 和 Cr 根据子采样率计算尺寸）
- **Alpha 通道**：如果存在，额外创建 Alpha 平面

```cpp
// YCbCr 格式的通道创建示例
if (get_colorspace() == heif_colorspace_YCbCr) {
    uint32_t cw, ch;
    get_subsampled_size(width, height, heif_channel_Cb, get_chroma_format(), &cw, &ch);
    out_img->add_plane(heif_channel_Y, width, height, get_bits_per_pixel(heif_channel_Y), limits);
    out_img->add_plane(heif_channel_Cb, cw, ch, get_bits_per_pixel(heif_channel_Cb), limits);
    out_img->add_plane(heif_channel_Cr, cw, ch, get_bits_per_pixel(heif_channel_Cr), limits);
}
```

**步骤 2：对每个通道执行最近邻缩放**

对交错格式和平面格式分别处理，同时区分 SDR（8bit 及以下）和 HDR（超过 8bit）两种情况：

```cpp
// SDR 平面格式的缩放核心逻辑
for (uint32_t y = 0; y < out_h; y++) {
    uint32_t iy = y * m_height / height;  // 计算源像素 Y 坐标
    for (uint32_t x = 0; x < out_w; x++) {
        uint32_t ix = x * m_width / width;  // 计算源像素 X 坐标
        out_data[y * out_stride + x] = in_data[iy * in_stride + ix];
    }
}

// SDR 交错格式的缩放核心逻辑（需要处理每像素多个分量）
for (uint32_t y = 0; y < out_h; y++) {
    uint32_t iy = y * m_height / height;
    for (uint32_t x = 0; x < out_w; x++) {
        uint32_t ix = x * m_width / width;
        for (int c = 0; c < nInterleaved; c++) {
            out_data[y * out_stride + x * nInterleaved + c] =
                in_data[iy * in_stride + ix * nInterleaved + c];
        }
    }
}
```

### 3.2 支持的色彩空间与位深

| 色彩空间 | 色度格式 | 位深 | 支持状态 |
|----------|----------|------|----------|
| RGB | 交错 (RGB/RGBA) | 8-bit | ✅ 已支持 |
| RGB | 交错 (RRGGBB/RRGGBBAA) | 10-16bit | ✅ 已支持（标注为未测试） |
| RGB | 平面 (R/G/B) | 8-bit | ✅ 已支持 |
| RGB | 平面 (R/G/B) | 10-16bit | ✅ 已支持 |
| YCbCr | 4:4:4 / 4:2:2 / 4:2:0 | 8-bit | ✅ 已支持 |
| YCbCr | 4:4:4 / 4:2:2 / 4:2:0 | 10-16bit | ✅ 已支持 |
| Monochrome | 单通道 | 8-bit | ✅ 已支持 |
| Monochrome | 单通道 | 10-16bit | ✅ 已支持 |
| Filter Array | - | - | ❌ 不支持 |

### 3.3 算法局限性

1. **仅支持最近邻插值**：不支持双线性插值、双三次插值、Lanczos 等高质量缩放算法
2. **HDR 交错格式标记为未测试**：代码中有 `// TODO: untested` 注释
3. **`heif_scaling_options` 结构体未定义**：API 中预留了缩放选项参数，但当前实现始终忽略该参数
4. **无抗锯齿处理**：缩小图像时可能出现锯齿和摩尔纹

---

## 4. Scale API 分析

### 4.1 公共 C API

**文件位置：** `libheif/api/libheif/heif_image.h`

```c
typedef struct heif_scaling_options heif_scaling_options;

// Currently, heif_scaling_options is not defined yet. Pass a NULL pointer.
LIBHEIF_API
heif_error heif_image_scale_image(const heif_image* input,
                                  heif_image** output,
                                  int width, int height,
                                  const heif_scaling_options* options);
```

**关键特征：**
- `heif_scaling_options` 结构体已声明但**未定义**，API 注释明确说明"当前传入 NULL 指针"
- 函数接受输入图像指针（`const`，不修改输入）
- 输出为新分配的 `heif_image*`，调用者需要通过 `heif_image_release()` 释放
- 返回 `heif_error` 表示成功或失败

### 4.2 API 实现层

**文件位置：** `libheif/api/libheif/heif_image.cc`

```cpp
heif_error heif_image_scale_image(const heif_image* input,
                                  heif_image** output,
                                  int width, int height,
                                  const heif_scaling_options* options)
{
    std::shared_ptr<HeifPixelImage> out_img;

    Error err = input->image->scale_nearest_neighbor(out_img, width, height, nullptr);
    if (err) {
        return err.error_struct(input->image.get());
    }

    *output = new heif_image;
    (*output)->image = std::move(out_img);

    return Error::Ok.error_struct(input->image.get());
}
```

**关键观察：**
- `options` 参数完全被忽略
- 直接调用 `HeifPixelImage::scale_nearest_neighbor()`
- `security_limits` 传入 `nullptr`（没有内存安全限制）

### 4.3 C++ 封装 API

**文件位置：** `libheif/api/libheif/heif_cxx.h`

```cpp
class Image {
public:
    // ...
    Image scale_image(int width, int height,
                      const ScalingOptions& options = ScalingOptions()) const;
    // ...
};

// 实现
inline Image Image::scale_image(int width, int height,
                                const ScalingOptions&) const
{
    heif_image* img;
    Error err = Error(heif_image_scale_image(m_image.get(), &img, width, height,
                                             nullptr)); // TODO: scaling options not defined yet
    if (err) {
        throw err;
    }
    return Image(img);
}
```

### 4.4 相关辅助 API

**缩略图编码 API：**

```c
// libheif/api/libheif/heif_aux_images.h
LIBHEIF_API
heif_error heif_context_encode_thumbnail(heif_context*,
                                         const heif_image* image,
                                         const heif_image_handle* master_image_handle,
                                         heif_encoder* encoder,
                                         const heif_encoding_options* options,
                                         int bbox_size,
                                         heif_image_handle** out_thumb_image_handle);
```

此 API 内部自动完成缩放+编码，但缩放逻辑被封装在内部实现中，不暴露给用户。

---

## 5. Scale 在项目中的使用现状

### 5.1 使用场景一：heif_thumbnailer（示例程序）

**文件：** `examples/heif_thumbnailer.cc`

这是 libheif 中**唯一直接使用** `heif_image_scale_image()` 公共 API 的示例程序。

**流程：**
```
1. 读取 HEIF 文件
2. 获取主图像 handle
3. 尝试获取内嵌缩略图
4. 解码图像 → heif_image*
5. 计算缩放尺寸（保持宽高比，限制在 bbox_size 内）
6. 调用 heif_image_scale_image() 缩放
7. 将缩放后的图像编码为 PNG 输出
```

**关键代码：**
```cpp
if (input_width > size || input_height > size) {
    int thumbnail_width, thumbnail_height;

    if (input_width > input_height) {
        thumbnail_height = input_height * size / input_width;
        thumbnail_width = size;
    } else if (input_height > 0) {
        thumbnail_width = input_width * size / input_height;
        thumbnail_height = size;
    } else {
        thumbnail_width = thumbnail_height = 0;
    }

    struct heif_image* scaled_image = NULL;
    err = heif_image_scale_image(image, &scaled_image,
                                 thumbnail_width, thumbnail_height, NULL);
    if (err.code) {
        std::cerr << "Could not scale image : " << err.message << "\n";
        return 1;
    }

    heif_image_release(image);
    image = scaled_image;
}
```

### 5.2 使用场景二：内部缩略图编码（HeifContext::encode_thumbnail）

**文件：** `libheif/context.cc`（第 1637 行）

```cpp
Result<std::shared_ptr<ImageItem>> HeifContext::encode_thumbnail(
    const std::shared_ptr<HeifPixelImage>& image,
    heif_encoder* encoder,
    const heif_encoding_options& options,
    int bbox_size)
{
    int orig_width = image->get_width();
    int orig_height = image->get_height();
    int thumb_width, thumb_height;

    // 计算缩放尺寸（保持宽高比）
    if (orig_width <= bbox_size && orig_height <= bbox_size) {
        return Error::Ok; // 原图已经足够小，不需要缩略图
    } else if (orig_width > orig_height) {
        thumb_height = orig_height * bbox_size / orig_width;
        thumb_width = bbox_size;
    } else {
        thumb_width = orig_width * bbox_size / orig_height;
        thumb_height = bbox_size;
    }

    // 尺寸对齐到偶数（编码器要求）
    thumb_width &= ~1;
    thumb_height &= ~1;

    // 执行缩放
    std::shared_ptr<HeifPixelImage> thumbnail_image;
    Error error = image->scale_nearest_neighbor(
        thumbnail_image, thumb_width, thumb_height, get_security_limits());

    // 编码缩略图
    auto encodingResult = encode_image(thumbnail_image, encoder, options,
                                       heif_image_input_class_thumbnail);
    return *encodingResult;
}
```

### 5.3 使用场景三：Alpha 通道尺寸匹配（ImageItem）

**文件：** `libheif/image-items/image_item.cc`（第 857 行附近）

当 Alpha 通道的尺寸与主图像不一致时，使用 `scale_nearest_neighbor()` 进行匹配：

```cpp
// TODO: we should include a decoding option to control whether libheif
// should automatically scale the alpha channel, and if so, which scaling
// filter (enum: Off, NN, Bilinear, ...).
if ((alpha_image->get_width() != img->get_width()) ||
    (alpha_image->get_height() != img->get_height())) {
    std::shared_ptr<HeifPixelImage> scaled_alpha;
    Error err = alpha->scale_nearest_neighbor(
        scaled_alpha, img->get_width(), img->get_height(),
        m_heif_context->get_security_limits());
    alpha = std::move(scaled_alpha);
}
```

**注意：** 代码中的 TODO 注释表明开发者已经意识到需要支持多种缩放算法。

### 5.4 使用场景四：heif_enc 中的缩略图（间接使用）

**文件：** `examples/heif_enc.cc`（第 2254 行）

`heif_enc` 工具通过 `-t` 参数生成缩略图，但它**不直接调用** `heif_image_scale_image()`，而是使用了封装好的 `heif_context_encode_thumbnail()` API：

```cpp
if (thumbnail_bbox_size > 0) {
    struct heif_image_handle* thumbnail_handle;
    options->save_alpha_channel = master_alpha && thumb_alpha;

    error = heif_context_encode_thumbnail(context,
                                          image.get(),
                                          handle,
                                          encoder,
                                          options,
                                          thumbnail_bbox_size,
                                          &thumbnail_handle);
}
```

---

## 6. heif-enc 工具当前的处理流程

### 6.1 整体流程

```
main()
 ├── 解析命令行参数（getopt_long）
 ├── 初始化 libheif（LibHeifInitializer）
 ├── 获取编码器（heif_context_get_encoder_for_format）
 ├── 根据模式分发：
 │   ├── do_encode_images()   ← 静态图像编码
 │   └── do_encode_sequence() ← 序列/视频编码
 ├── 添加 MIME 项（可选）
 └── 写入输出文件（heif_context_write_to_file）
```

### 6.2 `do_encode_images()` 详细流程

```
do_encode_images(context, encoder, options, args)
 │
 ├── for each input_filename in args:
 │   │
 │   ├── 检测是否为分块 TIFF（自动处理）
 │   ├── load_image(input_filename, output_bit_depth)
 │   │   → 返回 InputImage { image, exif, xmp, orientation }
 │   │
 │   ├── 处理分块/切片选项
 │   │
 │   ├── create_output_nclx_profile_and_configure_encoder()
 │   │
 │   ├── 编码主图像：
 │   │   ├── 分块模式 → encode_tiled()
 │   │   └── 普通模式 → heif_context_encode_image()
 │   │
 │   ├── 设置 CLLI/PASP/OMAF 属性
 │   │
 │   ├── 写入 EXIF 元数据
 │   ├── 写入 XMP 元数据
 │   │
 │   └── 生成缩略图（如果 -t 参数启用）：
 │       └── heif_context_encode_thumbnail()
 │
 └── 返回结果
```

### 6.3 当前命令行参数（与图像变换相关）

```
-q, --quality #          设置编码质量
-L, --lossless           无损编码
-t, --thumb #            生成最大尺寸为 # 的缩略图
    --no-thumb-alpha      缩略图不保存 Alpha 通道
-b, --bit-depth #        输入 HDR 图像位深（9-16）
-C, --chroma-downsampling 色度下采样算法
    --crop LEFT,RIGHT,TOP,BOTTOM  裁剪参数（尚未在命令行暴露）
```

**关键发现：heif-enc 当前没有独立的图像缩放功能。** 它只在生成缩略图时间接使用了缩放，但主图像的编码路径中不存在缩放步骤。

---

## 7. 将 Scale 功能集成到 heif-enc 的方案设计

### 7.1 需求分析

**核心需求：** 允许 heif-enc 在读取输入图像后、编码前，对图像执行缩放操作。

**使用场景：**
1. 将大尺寸图像缩小后编码为 HEIF/AVIF（减小文件体积）
2. 将小尺寸图像放大后编码
3. 指定精确输出尺寸
4. 按比例缩放（如 50%、200%）
5. 仅指定宽度或高度，自动计算另一维度（保持宽高比）

### 7.2 命令行参数设计

建议新增以下命令行选项：

```
--scale WxH              缩放到指定宽高（如 --scale 1920x1080）
--scale-width W          缩放到指定宽度，高度按比例计算
--scale-height H         缩放到指定高度，宽度按比例计算
--scale-factor F         按比例缩放（如 --scale-factor 0.5 缩小一半）
```

**长选项注册（在 `long_options[]` 数组中添加）：**

```cpp
// 新增的选项枚举值
#define OPTION_SCALE_SIZE          1001  // 使用尚未被占用的值
#define OPTION_SCALE_WIDTH         1002
#define OPTION_SCALE_HEIGHT        1003
#define OPTION_SCALE_FACTOR        1004

// 在 long_options[] 数组中添加
{(char* const) "scale",            required_argument, 0, OPTION_SCALE_SIZE},
{(char* const) "scale-width",     required_argument, 0, OPTION_SCALE_WIDTH},
{(char* const) "scale-height",    required_argument, 0, OPTION_SCALE_HEIGHT},
{(char* const) "scale-factor",    required_argument, 0, OPTION_SCALE_FACTOR},
```

### 7.3 处理流程设计

修改后的 `do_encode_images()` 流程：

```
do_encode_images(context, encoder, options, args)
 │
 ├── for each input_filename in args:
 │   │
 │   ├── load_image(input_filename, output_bit_depth)
 │   │   → InputImage { image, exif, xmp, orientation }
 │   │
 │   ├── 【新增】Scale 处理步骤：
 │   │   ├── 根据 --scale/--scale-width/--scale-height/--scale-factor 计算目标尺寸
 │   │   ├── 调用 heif_image_scale_image() 进行缩放
 │   │   ├── 释放原始 image，使用缩放后的 scaled_image
 │   │   └── 如果目标尺寸与原始相同，跳过缩放
 │   │
 │   ├── create_output_nclx_profile_and_configure_encoder()
 │   │
 │   ├── heif_context_encode_image() ← 使用（可能已缩放的）图像
 │   │
 │   └── ... (EXIF, XMP, 缩略图等后续处理不变)
```

### 7.4 数据流图

```
   ┌──────────────┐
   │  输入图像文件  │  (JPEG, PNG, TIFF, Y4M)
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │  load_image() │  解码为 heif_image
   └──────┬───────┘
          │
          ▼
   ┌──────────────────────┐
   │  是否需要缩放？        │
   │  (检查 scale 参数)     │
   └──────┬───────────┬───┘
          │ 是         │ 否
          ▼            │
   ┌──────────────┐   │
   │ 计算目标尺寸   │   │
   │ (保持宽高比)   │   │
   └──────┬───────┘   │
          │            │
          ▼            │
   ┌────────────────────┐
   │ heif_image_scale_  │ │
   │ image()            │ │
   │ (最近邻缩放)        │ │
   └──────┬─────────────┘ │
          │               │
          ▼               │
   ┌──────────────┐      │
   │ 释放原始图像   │      │
   │ 更新 image 指针│      │
   └──────┬───────┘      │
          │               │
          ◄───────────────┘
          │
          ▼
   ┌──────────────────────┐
   │ heif_context_encode_ │
   │ image()              │
   │ (HEIF/AVIF 编码)     │
   └──────┬───────────────┘
          │
          ▼
   ┌──────────────┐
   │  输出 HEIF 文件 │
   └──────────────┘
```

---

## 8. 实现细节与代码示例

### 8.1 新增全局变量

在 `heif_enc.cc` 文件头部的全局变量区域添加：

```cpp
// Scale 相关参数
int scale_target_width = 0;   // --scale WxH 或 --scale-width W
int scale_target_height = 0;  // --scale WxH 或 --scale-height H
float scale_factor = 0.0f;    // --scale-factor F
```

### 8.2 命令行参数解析

在 `main()` 函数的 `switch` 语句中添加：

```cpp
case OPTION_SCALE_SIZE: {
    // 解析 WxH 格式
    if (sscanf(optarg, "%dx%d", &scale_target_width, &scale_target_height) != 2) {
        std::cerr << "Invalid scale size format. Use WxH (e.g., 1920x1080).\n";
        return 5;
    }
    if (scale_target_width <= 0 || scale_target_height <= 0) {
        std::cerr << "Scale dimensions must be positive.\n";
        return 5;
    }
    break;
}
case OPTION_SCALE_WIDTH:
    scale_target_width = atoi(optarg);
    if (scale_target_width <= 0) {
        std::cerr << "Scale width must be positive.\n";
        return 5;
    }
    break;
case OPTION_SCALE_HEIGHT:
    scale_target_height = atoi(optarg);
    if (scale_target_height <= 0) {
        std::cerr << "Scale height must be positive.\n";
        return 5;
    }
    break;
case OPTION_SCALE_FACTOR:
    scale_factor = atof(optarg);
    if (scale_factor <= 0.0f) {
        std::cerr << "Scale factor must be positive.\n";
        return 5;
    }
    break;
```

### 8.3 缩放逻辑实现

在 `do_encode_images()` 函数中，在 `load_image()` 之后、`heif_context_encode_image()` 之前插入：

```cpp
// --- Scale image if requested ---
if (scale_target_width > 0 || scale_target_height > 0 || scale_factor > 0.0f) {
    int orig_width = heif_image_get_primary_width(image.get());
    int orig_height = heif_image_get_primary_height(image.get());
    int new_width = 0;
    int new_height = 0;

    if (scale_factor > 0.0f) {
        // 按比例缩放
        new_width = static_cast<int>(orig_width * scale_factor + 0.5f);
        new_height = static_cast<int>(orig_height * scale_factor + 0.5f);
    }
    else if (scale_target_width > 0 && scale_target_height > 0) {
        // 指定精确尺寸
        new_width = scale_target_width;
        new_height = scale_target_height;
    }
    else if (scale_target_width > 0) {
        // 仅指定宽度，按比例计算高度
        new_width = scale_target_width;
        new_height = orig_height * scale_target_width / orig_width;
    }
    else if (scale_target_height > 0) {
        // 仅指定高度，按比例计算宽度
        new_width = orig_width * scale_target_height / orig_height;
        new_height = scale_target_height;
    }

    // 确保尺寸至少为 1
    if (new_width < 1) new_width = 1;
    if (new_height < 1) new_height = 1;

    // 对齐到偶数（某些编码器要求）
    if (new_width > 1) new_width &= ~1;
    if (new_height > 1) new_height &= ~1;

    if (new_width != orig_width || new_height != orig_height) {
        heif_image* scaled_image = nullptr;
        heif_error err = heif_image_scale_image(image.get(), &scaled_image,
                                                 new_width, new_height, nullptr);
        if (err.code) {
            std::cerr << "Could not scale image: " << err.message << "\n";
            return 1;
        }

        if (logging_level > 0) {
            std::cout << "Scaled image from " << orig_width << "x" << orig_height
                      << " to " << new_width << "x" << new_height << "\n";
        }

        // 替换为缩放后的图像
        image = std::shared_ptr<heif_image>(scaled_image,
                                            [](heif_image* img) { heif_image_release(img); });
    }
}
```

### 8.4 帮助文档更新

在 `show_help()` 函数中添加说明：

```cpp
<< "\n"
<< "Scaling options:\n"
<< "      --scale WxH              scale image to specified dimensions (e.g., 1920x1080)\n"
<< "      --scale-width W          scale to specified width, height computed to keep aspect ratio\n"
<< "      --scale-height H         scale to specified height, width computed to keep aspect ratio\n"
<< "      --scale-factor F         scale by factor (e.g., 0.5 for half, 2.0 for double)\n"
```

### 8.5 使用示例

```bash
# 将图像缩放到 1920x1080 后编码为 HEIF
heif-enc --scale 1920x1080 input.jpg -o output.heic

# 将图像缩放到宽度 800（保持宽高比）后编码为 AVIF
heif-enc --scale-width 800 -A input.png -o output.avif

# 将图像缩小一半后编码
heif-enc --scale-factor 0.5 input.jpg -o output.heic

# 结合其他选项
heif-enc --scale 1280x720 -q 80 -t 320 input.jpg -o output.heic
```

---

## 9. 潜在问题与改进建议

### 9.1 当前 Scale 实现的不足

#### 9.1.1 算法质量

**问题：** 最近邻插值是最简单但质量最差的插值算法。缩小图像时会产生锯齿，放大图像时会产生明显的像素化块效应。

**建议：** 未来应在 `heif_scaling_options` 中支持多种插值算法：

```c
typedef enum heif_scaling_algorithm {
    heif_scaling_nearest_neighbor = 0,  // 最近邻（快速，低质量）
    heif_scaling_bilinear = 1,          // 双线性插值（中等质量）
    heif_scaling_bicubic = 2,           // 双三次插值（高质量）
    heif_scaling_lanczos = 3,           // Lanczos 重采样（最高质量，用于缩小）
} heif_scaling_algorithm;

struct heif_scaling_options {
    uint8_t version;
    heif_scaling_algorithm algorithm;
    // 未来可扩展：
    // float sharpness;           // 锐化参数
    // int preserve_aspect_ratio; // 是否保持宽高比
};
```

#### 9.1.2 安全性

**问题：** 公共 API `heif_image_scale_image()` 中，`security_limits` 传入 `nullptr`，不受内存分配限制的保护。

**建议：** 
- 在公共 API 中添加安全限制参数，或
- 在 API 实现中使用默认的全局安全限制

#### 9.1.3 元数据处理

**问题：** 缩放后的图像可能包含不准确的元数据（如 EXIF 中的分辨率信息）。

**建议：** 缩放后应更新 EXIF 中的 `PixelXDimension` 和 `PixelYDimension` 标签。

#### 9.1.4 色彩空间考虑

**问题：** 对于 YCbCr 4:2:0/4:2:2 格式的图像，色度通道的缩放比例与亮度通道不同，但当前实现使用统一的最近邻算法，可能导致色彩偏差。

**建议：** 色度通道应使用至少双线性插值以减少色彩偏差。

### 9.2 heif-enc 集成时的注意事项

#### 9.2.1 缩放与分块编码的交互

当使用分块编码（`--tiling`）时，缩放应该在分块之前完成。需要确保缩放后的图像尺寸能被分块大小整除，否则可能需要添加填充。

#### 9.2.2 缩放与缩略图的交互

如果同时指定了 `--scale` 和 `-t`（缩略图），缩略图应基于缩放后的图像生成。当前的流程设计已经自然支持这一点，因为缩放在编码之前执行。

#### 9.2.3 序列/视频编码支持

`do_encode_sequence()` 也应该支持缩放功能，对序列中的每一帧应用相同的缩放参数。

#### 9.2.4 宽高对齐

某些编码器（如 HEVC）要求宽和高为偶数。缩放后应进行对齐：

```cpp
// 对齐到偶数
new_width &= ~1;
new_height &= ~1;
```

### 9.3 长期改进路线图

1. **Phase 1（基础集成）**：将 `heif_image_scale_image()` 通过命令行参数暴露到 `heif-enc`
2. **Phase 2（算法增强）**：定义 `heif_scaling_options` 结构体，实现双线性/双三次插值
3. **Phase 3（质量优化）**：添加 Lanczos 重采样、抗锯齿预滤波
4. **Phase 4（功能扩展）**：支持区域裁剪+缩放的组合操作、支持序列批量缩放

---

## 10. 总结

### 10.1 核心发现

1. **Scale API 已经存在**：`heif_image_scale_image()` 是一个完整的公共 C API，具备跨色彩空间、跨位深的缩放能力。

2. **仅使用最近邻算法**：当前唯一的缩放实现 `scale_nearest_neighbor()` 使用了最简单的最近邻插值算法，质量有限但性能高效。

3. **`heif_scaling_options` 预留但未实现**：API 设计中已经预留了缩放选项的扩展点，但当前这个结构体未定义，所有调用都传入 `nullptr`。

4. **heif-enc 未暴露独立缩放功能**：`heif-enc` 工具目前只在生成缩略图时间接使用了缩放（通过 `heif_context_encode_thumbnail()`），没有给用户提供独立的图像缩放能力。

5. **heif-thumbnailer 是参考实现**：`heif_thumbnailer.cc` 提供了完整的"解码 → 缩放 → 编码"流程参考，可以直接借鉴其实现模式。

### 10.2 集成可行性评估

| 维度 | 评估 | 说明 |
|------|------|------|
| API 可用性 | ✅ 高 | `heif_image_scale_image()` 已经是稳定的公共 API |
| 实现复杂度 | ✅ 低 | 只需在 `do_encode_images()` 中添加约 50 行代码 |
| 侵入性 | ✅ 低 | 不需要修改核心库，仅修改示例程序 |
| 参考实现 | ✅ 有 | `heif_thumbnailer.cc` 提供了完整的参考 |
| 算法质量 | ⚠️ 中 | 最近邻算法质量有限，但对于初始版本可接受 |
| 向后兼容 | ✅ 完全 | 新增命令行选项，不影响现有功能 |

### 10.3 建议的实施优先级

1. **P0（必须）**：添加 `--scale WxH` 命令行选项，支持精确尺寸缩放
2. **P1（推荐）**：添加 `--scale-width` 和 `--scale-height`，支持保持宽高比的单维度缩放
3. **P1（推荐）**：添加 `--scale-factor`，支持按比例缩放
4. **P2（可选）**：将缩放功能扩展到 `do_encode_sequence()` 支持序列/视频批量缩放
5. **P3（未来）**：定义 `heif_scaling_options`，实现高质量缩放算法

---

*报告生成时间：2026-03-06*
*分析基于 libheif 仓库最新主分支（commit: baac699f9e52e8920873fe2782d74b6d2e8379a5）*

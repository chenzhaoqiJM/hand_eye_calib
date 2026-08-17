# AprilTag 36h11 标定板 PDF 生成器

本目录用于生成具有确定物理尺寸的单张 AprilTag 36h11 标定板 PDF。OpenCV 负责生成标签编码图像，ReportLab 负责把标签按毫米尺寸放到 PDF 页面中。

生成的标定板适用于相邻目录 `eye_to_hand_rgb_v4l2/` 的眼在手外标定流程。

## 安装依赖

```bash
python -m pip install -r requirements.txt
```

必须安装包含 `aruco` 模块的 `opencv-contrib-python`，不能只安装基础版 `opencv-python`。

## 生成 PDF

生成 AprilTag 36h11、ID 0、黑色正方形边长 50 mm 的 A4 PDF：

```bash
python generate_tag_pdf.py \
  --id 0 \
  --size-mm 50 \
  --output apriltag_36h11_id0_50mm.pdf
```

默认参数：

- 标签家族：`AprilTag 36h11`
- 标签 ID：`0`，可选范围为 `0-586`
- 黑色标签最外框边长：`50 mm`
- 标签外围白色静区：每侧 `15 mm`
- 页面：`A4`
- PDF 内部标签位图：`1200 x 1200` 像素

其他示例：

```bash
# 生成 ID 12、黑色外框边长 80 mm 的标签
python generate_tag_pdf.py --id 12 --size-mm 80

# 使用 Letter 页面
python generate_tag_pdf.py --id 0 --size-mm 50 --page-size LETTER

# 增加外围白色静区
python generate_tag_pdf.py --id 0 --size-mm 50 --margin-mm 25
```

## 打印流程

1. 用 PDF 阅读器打开生成的文件。
2. 打印比例选择 `100%` 或“实际大小”。
3. 禁止选择“适合页面”“缩小超大页面”或任何自动缩放选项。
4. 建议关闭打印机驱动中的无边距扩展、页面适配和自动裁剪。
5. 打印后用卡尺测量标签最外侧黑色正方形的水平和垂直边长。
6. 如果两个方向的尺寸偏差明显，检查打印机的缩放或纸张走纸误差。
7. 将打印件平整粘贴到硬质平板上，避免翘曲、褶皱和反光覆盖膜。

程序页面下方的说明文字不属于标签。标签尺寸只指最外侧黑色正方形，不包含外围白色静区。

## 用于手眼标定

打印并测量后，将实测的黑色正方形边长传给眼在手外标定程序。例如实测为 `49.82 mm`：

```bash
cd ../eye_to_hand_rgb_v4l2

python calibrate.py \
  --device /dev/video0 \
  --intrinsics intrinsics.json \
  --tag-size-mm 49.82
```

不要因为生成参数写的是 `50 mm` 就忽略打印后的实测尺寸。PnP 平移尺度直接依赖 `--tag-size-mm`，尺寸输入误差会按比例影响相机外参的平移结果。

## 安装与拍摄建议

- 标签应刚性固定在机械臂法兰或与法兰刚性连接的平板上。
- 黑色外框和外围白色静区必须完整可见。
- 尽量使用哑光纸或哑光覆膜，避免灯光反射破坏黑白边界。
- 标签平面必须保持平整；普通纸直接悬空容易弯曲，不适合精确标定。
- 拍摄距离应保证标签每条边有足够像素，避免严重模糊和过曝。
- 整个数据采集过程中必须使用同一个标签 ID 和同一块安装板。

## 参数说明

```text
--id          AprilTag 36h11 标签 ID，范围 0-586
--size-mm     最外侧黑色正方形的目标边长，单位 mm
--margin-mm   黑色正方形外围的白色静区宽度，单位 mm
--pixels      PDF 内部标签位图分辨率，不改变打印物理尺寸
--page-size   A4 或 LETTER
--output      输出 PDF 路径
```

# 眼在手外 RealSense D4xx 相机标定

本目录支持使用 Intel RealSense D4xx 系列相机进行眼在手外 RGB 手眼标定。相机固定在工作空间中，AprilTag 或棋盘格刚性固定在机器人法兰上。

输出结果为：

```text
T_base_camera：RealSense 彩色相机坐标系 -> 机器人基座坐标系
```

## 支持范围

- D405、D415、D435、D435i、D455 等 D4xx 系列设备；
- 使用 RealSense **color** 流采集 BGR 图像；
- 自动读取当前设备、当前分辨率对应的出厂彩色相机内参；
- 支持 RealSense `brown_conrady`、`inverse_brown_conrady` 和无畸变模型；
- 深度流不是本标定流程的必要条件，因此默认不启动深度流。

## 安装依赖

先安装通用依赖：

```bash
python -m pip install -r requirements.txt
```

再安装 RealSense 后端依赖：

```bash
python -m pip install -r requirements-realsense.txt
```

也可以直接安装两份依赖：

```bash
python -m pip install -r requirements.txt -r requirements-realsense.txt
```

安装后可以用下面的命令确认 Python 能加载 SDK：

```bash
python -c "import pyrealsense2 as rs; print(rs.__version__)"
```

如果系统尚未安装 librealsense 驱动、udev 规则或 RealSense Viewer，请先按照 Intel RealSense SDK 的系统安装说明完成安装，并确认设备可以被系统识别。

## 检查 RealSense 设备

建议先运行：

```bash
realsense-viewer
```

确认设备能够正常显示彩色图像，并确认目标分辨率和帧率可用。也可以使用项目自带的相机后端做最小采集测试：

```bash
python camera_realsense.py
```

该命令会启动彩色流并打印设备信息、实际流配置、彩色内参和畸变模型，同时保存一张 `captured_realsense_color.jpg`。

如果连接了多个 RealSense 设备，先列出序列号，再使用 `--serial-number` 选择目标设备：

```bash
rs-enumerate-devices
```

## 采集标定数据

### 使用 AprilTag

默认采集 `1280×720 @ 30 FPS` 彩色图像，内参由设备自动读取，不需要提供 `--intrinsics`：

```bash
python calibrate.py \
  --camera-backend realsense \
  --target-type apriltag \
  --tag-size-mm 50 \
  --min-samples 20
```

多个设备时：

```bash
python calibrate.py \
  --camera-backend realsense \
  --serial-number 123456789012 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --target-type apriltag \
  --tag-size-mm 50 \
  --min-samples 20
```

### 使用棋盘格

例如采集 `7×6` 内角点、方格边长 `40 mm` 的棋盘格：

```bash
python calibrate.py \
  --camera-backend realsense \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --target-type chessboard \
  --chessboard-columns 7 \
  --chessboard-rows 6 \
  --square-size-mm 40 \
  --min-board-coverage 0.015 \
  --max-reprojection-px 2.0 \
  --min-samples 20
```

脚本启动后会打印浏览器预览地址。每组样本按以下步骤操作：

1. 将机器人移动到新的法兰位置和姿态；
2. 停止机器人，确保标定板清晰、完整且没有反光遮挡；
3. 确认预览中的目标检测状态为有效；
4. 在终端按 Enter 保存彩色图像；
5. 输入对应的 `T_base_flange`：`x y z rx ry rz`；
6. 重复采集至少 15～20 组，建议 20～30 组。

默认输入单位为 `mm, deg`，也可以使用：

```bash
python calibrate.py \
  --camera-backend realsense \
  --position-unit m \
  --angle-unit rad
```

只采集不立即求解：

```bash
python calibrate.py --camera-backend realsense --no-solve ...
```

默认情况下数据保存到：

```text
eye_to_hand_rgb/data/YYYY-mm-dd_HHMMSS/
```

## 设备自动保存的内参

每次采集会把当前 RealSense 彩色流的内参保存到数据目录：

```text
intrinsics.json
```

文件中包含：

- `fx`, `fy`, `cx`, `cy`：彩色相机内参；
- `width`, `height`：内参对应的彩色流分辨率；
- `coeffs`：RealSense 原始畸变系数；
- `distortion_model`：RealSense 畸变模型；
- `session.json` 中的 `camera_device`：设备名称、序列号、固件版本和 USB 类型；
- `session.json` 中的 `camera_stream`：实际彩色流配置。

不要把某个分辨率的内参用于另一个分辨率的图像。RealSense 后端获取的是当前启动配置对应的内参，因此比手工复制内参更安全。

## 离线求解和验证

采集目录中包含 `intrinsics.json`、图像和手动记录的机器人位姿后，可以重复求解：

```bash
python calibrate_from_data.py data/2026-01-01_120000
```

`calibrate_from_data.py` 会优先从该目录的 `session.json` 读取目标类型、AprilTag 尺寸或棋盘格参数，因此通常不需要再次填写这些参数。

指定算法：

```bash
python calibrate_from_data.py \
  data/2026-01-01_120000 \
  --method PARK
```

比较多种 OpenCV 手眼算法：

```bash
python validate_eye_to_hand.py data/2026-01-01_120000
```

如果位姿还在 CSV 中，可以先导入：

```bash
python calibrate_from_data.py data/my_session \
  --poses-csv data/my_session/poses.csv \
  --position-unit mm \
  --angle-unit deg
```

## 坐标系和深度数据说明

眼在手外的约束为：

```text
T_flange_target = inv(T_base_flange) @ T_base_camera @ T_camera_target
```

这里的 `T_camera_target` 是由 RealSense **彩色相机**图像计算得到的。因此最终结果 `T_base_camera` 表示彩色相机坐标系到机器人基座坐标系的变换。

本流程默认不开启深度流。如果后续需要将深度点转换到基座坐标系，需要：

1. 使用 RealSense 深度流获得深度相机坐标系中的点；
2. 使用设备提供的 color/depth 外参，在深度相机坐标系和彩色相机坐标系之间转换；
3. 使用本流程生成的 `T_base_camera` 转换到机器人基座坐标系。

不要把深度相机坐标系和 RGB 彩色相机坐标系混用。

## 常见问题

### 找不到设备

- 确认 USB 线和端口满足设备要求；
- 尽量使用稳定的 USB 3.x 连接；
- 使用 `realsense-viewer` 或 `rs-enumerate-devices` 检查设备；
- Linux 下确认当前用户具有访问 `/dev/video*` 和 RealSense USB 设备的权限；
- 多设备时使用正确的 `--serial-number`。

### 不支持指定分辨率或帧率

D4xx 不同型号支持的彩色分辨率、帧率和格式可能不同。先用 `realsense-viewer` 确认可用组合，再调整 `--width`、`--height` 和 `--fps`。

### 棋盘格或 AprilTag 无法检测

- 确认目标实际尺寸与参数一致；
- 确保目标完整出现在彩色图像中；
- 避免运动模糊和强反光；
- 采集时覆盖多个法兰位置和旋转方向；
- 确认使用的是彩色图像，而不是深度图像。

### 畸变模型错误

RealSense 的畸变模型不能随意改成 `plumb_bob`。离线求解会根据 `intrinsics.json` 中的 `distortion_model` 处理 `brown_conrady` 和 `inverse_brown_conrady`。如果删除或手工修改该字段，可能导致 PnP 结果不正确。

## 与现有 V4L2 流程的关系

现有 V4L2 命令和 `README.md` 保持不变。V4L2 仍然是默认后端：

```bash
python calibrate.py \
  --camera-backend v4l2 \
  --device /dev/video0 \
  --intrinsics intrinsics.json
```

RealSense 使用新增的 `--camera-backend realsense`，两种后端共用目标检测、数据格式、手眼求解和验证逻辑。

# 单棋盘格像素到二维平面映射

本目录提供单张图片和实时 Web 两种标定方式。两种方式默认先使用相机内参对图像去畸变，再检测角点和计算映射。输出矩阵 `H_pixel_to_plane` 将去畸变图像的像素坐标映射到棋盘格平面坐标：

```text
[X, Y, W] = H_pixel_to_plane @ [u, v, 1]
X_plane = X / W
Y_plane = Y / W
```

平面坐标约定：原点是 OpenCV 检测角点序列中的第一个内角点，X 沿棋盘格列方向，Y 沿行方向。`square-size` 的单位会原样成为输出坐标单位，例如输入毫米则输出毫米。角点参数和坐标原点都基于“内角点”，不是棋盘外边缘。

## 安装

```bash
python -m pip install -r requirements.txt
```

棋盘格参数是“内角点数量”，不是方格数量。比如棋盘有 10 列、7 行方格时，参数是 `9x6`。参数格式为 `列x行`，默认值为 `9x6`。

## 运行前准备

两种模式默认读取 `../monocular_rgb_calibration/intrinsics.json`，也可通过 `--intrinsics` 指定其他内参文件。该文件必须包含以下字段：

```json
{
  "fx": 700.0,
  "fy": 700.0,
  "cx": 640.0,
  "cy": 360.0,
  "width": 1280,
  "height": 720,
  "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
  "distortion_model": "plumb_bob"
}
```

`width` 和 `height` 必须与输入图像或摄像头实际返回的分辨率完全一致；脚本不会自动缩放内参。建议先使用 `monocular_rgb_calibration` 完成内参标定，再将生成的文件通过 `--intrinsics` 传入。去畸变后，PnP 使用相同的相机内参并传入全零畸变参数。

去畸变控制项为 `--undistort/--no-undistort`，默认值为 `--undistort`。如果输入图像或视频已经去畸变，可使用 `--no-undistort` 关闭此步骤。关闭后，实时模式会将内参文件中的原始畸变系数传给 PnP；单张图片模式只计算单应性，不执行 PnP。

PnP 畸变参数可通过 `--zero-distortion` 置零，默认不启用。开启后，无论是否使用 `--undistort`，传给 `solvePnP` 的畸变参数都会是全零；该选项只影响 PnP，不改变图像是否去畸变。去畸变模式本身仍默认使用全零参数；在 `--no-undistort` 下可用该选项强制置零。可使用 `--no-zero-distortion` 显式恢复默认行为。

## 1. 从图像计算

```bash
python calculate_homography.py image.jpg \
  --pattern 7x4 \
  --square-size 24 \
  --intrinsics ../monocular_rgb_calibration/intrinsics.json \
  --output pixel_to_plane_homography.json \
  --visualization corners.jpg
```

脚本会在终端打印矩阵，并将矩阵、坐标约定、图像尺寸、内点数量和重投影误差保存到 JSON。单张棋盘图能够得到数学上的单应性，但要获得稳定结果，应让棋盘尽量覆盖工作区域，保证图像清晰且角点完整。

如需使用原始、未去畸变的图片进行计算：

```bash
python calculate_homography.py image.jpg \
  --pattern 7x4 --square-size 24 \
  --no-undistort --output pixel_to_plane_homography.json
```

无论是否去畸变，都需要确保图片中的棋盘格与 `--pattern` 完全匹配。

## 2. 打开视频流和 Web 页面

```bash
python live_homography_web.py \
  --device /dev/video4 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --pattern 7x4 \
  --square-size 24 \
  --intrinsics ../monocular_rgb_calibration/intrinsics.json \
  --output pixel_to_plane_homography.json \
  --port 8080
```

浏览器打开 `http://127.0.0.1:8080`。页面会显示视频和棋盘格检测结果；只有点击 **Calculate mapping matrix** 后，脚本才会计算并保存矩阵。若需要局域网其他设备访问，将 `--host` 保持为默认的 `0.0.0.0`，然后访问运行设备的实际 IP。

如果从其他目录启动脚本，建议使用脚本所在目录作为当前目录，或者把 `--intrinsics` 和 `--output` 写成明确路径。例如：

```bash
cd /path/to/hand_eye_calib/planar_homography
python live_homography_web.py --device /dev/video4 \
  --width 1280 --height 720 --fps 30 \
  --pattern 7x4 --square-size 25 \
  --intrinsics ../monocular_rgb_calibration/intrinsics.json \
  --output pixel_to_plane_homography.json --port 8080
```

实时模式默认对每一帧调用 `cv2.undistort`。若摄像头输出已经去畸变，可添加 `--no-undistort`：

```bash
python live_homography_web.py --device /dev/video4 \
  --width 1280 --height 720 --fps 30 \
  --pattern 7x4 --square-size 25 --no-undistort
```

如果需要在未去畸变的图像上使用零畸变参数执行 PnP：

```bash
python live_homography_web.py --device /dev/video4 \
  --width 1280 --height 720 --pattern 7x4 --square-size 25 \
  --no-undistort --zero-distortion
```

程序启动时会验证摄像头能否打开、实际分辨率是否等于 `--width`/`--height`，以及内参文件是否匹配。任一项失败都会直接退出；这不是棋盘检测失败。使用 V4L2 设备时，确认当前用户有权限访问 `/dev/video4`，并先用 `v4l2-ctl --list-formats-ext -d /dev/video4` 查看支持的分辨率。

计算完成后，页面会冻结并显示本次用于标定的图像，以红色十字标出实际坐标原点。此时可使用：

- **Measure point**：点击图像，查看像素坐标及其对应的实际二维坐标。
- **Measure distance**：依次点击 A、B 两点，查看两点实际坐标及平面距离。
- **Clear marks**：清除当前测量标记。

测量结果的单位与 `--square-size` 一致。浏览器中图像即使被缩放，点击位置也会换算回相机原始像素坐标。再次点击 **Calculate mapping matrix** 可用当前视频帧重新标定。

实时页面中显示的相机坐标由 `matrix_camera_plane` 计算，表示棋盘平面坐标点在相机坐标系中的位置。`--zero-distortion` 开启时，该位姿使用零畸变参数计算。该位姿仅用于测量显示；像素到平面的主要结果是 `matrix_pixel_to_plane`。

## 输出 JSON

常用字段如下：

- `matrix_pixel_to_plane`：3x3 像素到棋盘平面单应性矩阵。
- `plane_coordinate_unit`、`square_size`：平面坐标单位及方格尺寸。
- `plane_origin`、`plane_axes`：坐标原点和轴方向约定。
- `image_size`、`pattern_inner_corners`：矩阵适用的图像尺寸和棋盘参数。
- `inlier_count`、`corner_count`、`mean_reprojection_error`、`max_reprojection_error`：RANSAC 内点和重投影误差。

实时模式还会写入 `plane_origin_pixel`、`matrix_camera_plane`、`camera_coordinate_unit`、`pnp_reprojection_rms` 和 `undistorted`。应用读取矩阵时，应使用输出中的图像尺寸；更换摄像头分辨率、镜头位置或工作平面后必须重新标定。

## 相机畸变

实时模式和单张图片模式默认先去畸变。如果镜头畸变明显，建议保持默认设置，并使用与输入分辨率匹配的内参。需要强制 PnP 忽略畸变时使用 `--zero-distortion`；不要把原始畸变系数传给已经去畸变的像素，否则会重复校正。

## 注意

该矩阵只描述当前相机视角下的一个平面。棋盘所在平面改变，或者相机/工作台位置改变，都需要重新计算。映射结果不包含 Z，高度必须由应用场景单独确定。重投影误差应结合应用精度要求判断，不能只看矩阵是否成功生成。

# 单棋盘格像素到二维平面映射

本目录提供两个脚本。输出矩阵 `H_pixel_to_plane` 将图像像素坐标映射到棋盘格平面坐标：

```text
[X, Y, W] = H_pixel_to_plane @ [u, v, 1]
X_plane = X / W
Y_plane = Y / W
```

平面坐标约定：原点是检测角点序列中的第一个内角点，X 沿棋盘格列方向，Y 沿行方向。`square-size` 的单位会原样成为输出坐标单位，例如输入毫米则输出毫米。

## 安装

```bash
python -m pip install -r requirements.txt
```

棋盘格参数是“内角点数量”，不是方格数量。比如棋盘有 10 列、7 行方格时，参数是 `9x6`。

## 1. 从图像计算

```bash
python calculate_homography.py image.jpg \
  --pattern 9x6 \
  --square-size 25 \
  --output pixel_to_plane_homography.json \
  --visualization corners.jpg
```

脚本会在终端打印矩阵，并将矩阵、坐标约定、内点数量和重投影误差保存到 JSON。单张棋盘图能够得到数学上的单应性，但要获得稳定结果，应让棋盘尽量覆盖工作区域，保证图像清晰且角点完整。

## 2. 打开视频流和 Web 页面

```bash
python live_homography_web.py \
  --device /dev/video4 \
  --width 1280 --height 720 --fps 30 \
  --pattern 7x4 \
  --square-size 25 \
  --output pixel_to_plane_homography.json \
  --port 8080
```

浏览器打开 `http://127.0.0.1:8080`。页面会显示视频和棋盘格检测结果；只有点击 **Calculate mapping matrix** 后，脚本才会计算并保存矩阵。若需要局域网其他设备访问，将 `--host` 保持为默认的 `0.0.0.0`，然后访问运行设备的实际 IP。

## 相机畸变

如果镜头畸变明显，建议先完成相机内参标定并在计算前去畸变。目前脚本没有强制要求内参，因为许多工业相机已经提供了近似无畸变图像；若使用广角镜头，应扩展 `homography_common.py` 的输入流程，先调用 `cv2.undistort` 再检测和求解。

## 注意

该矩阵只描述当前相机视角下的一个平面。棋盘所在平面改变，或者相机/工作台位置改变，都需要重新计算。映射结果不包含 Z，高度必须由应用场景单独确定。

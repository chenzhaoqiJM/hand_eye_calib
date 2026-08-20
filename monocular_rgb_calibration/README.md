# 单目 USB RGB 相机标定（V4L2 + 浏览器预览）

使用 OpenCV 直接读取 Linux V4L2 相机（如 `/dev/video0`），实时检测棋盘格并通过 MJPEG 推送到浏览器。页面会显示当前画面是否有效、清晰度、棋盘覆盖率，并提供采集和计算按钮。

## 标定板

默认使用 **9×6 个内角点**的棋盘格；`--columns`、`--rows` 填的是内角点数量，不是方格数量。请准确测量相邻角点之间的距离，默认 `25 mm`。

建议：

- 打印后粘贴到平整硬板，不能翘曲；
- 至少采集 15 张，建议 20–30 张；
- 棋盘应覆盖中心、四角和边缘，并具有不同距离、倾角；
- 浏览器显示绿色 `VALID` 时才可采集，仍需避免重复采集几乎相同的姿态；
- 标定后 RMS 重投影误差通常越小越好；若明显超过 1 px，应检查角点尺寸、运动模糊、对焦和样本多样性。

### 生成可打印棋盘格 PDF

默认生成与标定程序一致的 9×6 内角点、25 mm 方格、A4 横向 PDF：

```bash
python generate_chessboard_pdf.py
```

指定尺寸和输出路径：

```bash
python generate_chessboard_pdf.py \
  --columns 7 --rows 4 \
  --square-size-mm 25 \
  --page-size A4 --orientation landscape \
  --output chessboard_7x4_25mm.pdf
```

PDF 使用矢量方格，并附带 100 mm 校验尺。打印时必须选择 **100% / 实际大小**，关闭“适合页面”“缩小超大页面”和其他缩放选项。打印后用直尺测量多个方格及 100 mm 校验尺；测量值不准确时不能用于标定。

## 安装

```bash
cd /media/chenzhaoqi/data/tmp/hand_eye_calib/monocular_rgb_calibration
python -m pip install -r requirements.txt
```

## 运行

```bash
python calibrate_camera.py \
  --device /dev/video4 \
  --width 1280 --height 720 --fps 30 \
  --pixel-format MJPG \
  --columns 7 --rows 4 \
  --square-size-mm 24 \
  --min-samples 15 \
  --output ./intrinsics.json
```

程序会打印浏览器地址，通常为 `http://127.0.0.1:8080`，同一局域网设备也可通过打印出的主机 IP 打开。预览服务默认监听所有网卡且没有认证，只应在可信网络使用。

浏览器操作顺序：

1. 调整棋盘位置，等待状态变成绿色 `VALID: ready to capture`；
2. 点击“采集当前有效图像”；
3. 改变距离和姿态，重复采集；
4. 达到最少样本数后，点击“计算并保存标定结果”；
5. 终端会打印 RMS、相机矩阵、畸变系数以及完整 JSON；按 `Ctrl+C` 退出。

## 输出

- `--output`：输出与 `eye_to_hand_rgb_v4l2/intrinsics.example.json` 相同结构的 JSON；
- `captures/<时间>/frame_*.png`：实际采用的原始图片；
- `captures/<时间>/calibration_report.json`：包含 RMS、逐图误差、相机配置和内参。

`coeffs` 为 OpenCV 顺序：

```text
[k1, k2, p1, p2, k3, ...]
```

必须使用与实际业务完全一致的分辨率、相机焦距/变焦、对焦及成像模式。分辨率变化后不能直接沿用原内参。

## 常用调节

设备不支持 MJPG 时可尝试 `--pixel-format YUYV`。如果画面清晰但一直因阈值无效，可适当降低：

```bash
python calibrate_camera.py ... --min-sharpness 50 --min-coverage 0.03
```

不要为了通过检测过度降低阈值；模糊或棋盘过小的样本会降低标定质量。

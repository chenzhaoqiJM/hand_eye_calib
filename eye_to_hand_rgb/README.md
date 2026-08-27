# 眼在手外 RGB 相机标定（V4L2）

本目录用于固定 RGB 相机的眼在手外标定，默认使用刚性固定在法兰上的**棋盘格**。结果为：

```text
T_base_camera：相机坐标系 -> 机器人基座坐标系
```

## 准备工作

- RGB 相机固定在机器人外部，棋盘格固定在法兰上，采集期间二者安装关系不能改变。
- 准备与采集分辨率一致的 RGB 相机内参，格式参考 `intrinsics.example.json`。
- `coeffs` 使用 OpenCV 顺序：`[k1, k2, p1, p2, k3]`。
- 建议采集 20～30 组不同位置和姿态，至少 8 组。
- 每次输入对应的 `T_base_flange`，默认单位为 `mm, deg`。

## 检查相机

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

如设备不支持 `MJPG`，将参数改为 `--pixel-format YUYV`。

## 棋盘格采集

安装依赖：

```bash
python -m pip install -r requirements.txt
```

采集一个 `7×6` 内角点、方格边长 `40 mm` 的棋盘格：

```bash
python calibrate.py \
  --device /dev/video0 \
  --intrinsics intrinsics.json \
  --target-type chessboard \
  --chessboard-columns 7 \
  --chessboard-rows 6 \
  --square-size-mm 40 \
  --min-board-coverage 0.015 \
  --max-reprojection-px 2.0 \
  --min-samples 20
```

启动后终端会打印浏览器预览地址。打开该地址可以看到实时画面、棋盘格角点，以及以下状态：

- `VALID`：当前画面是否满足保存条件；
- `RMS`：PnP 重投影 RMS 误差，单位为像素；
- `COVERAGE`：棋盘格角点包围区域占图像的比例。

程序会在保存前再次检测棋盘格并检查这些质量指标。机器人停止后按 Enter 拍摄，再输入对应的：

```text
x y z rx ry rz
```

也可使用米和弧度：`--position-unit m --angle-unit rad`。

如果预览中显示 `NOT FOUND`，请确认棋盘格内角点参数与实物一致，并确保棋盘格完整、清晰地出现在画面中。`--min-board-coverage` 和 `--max-reprojection-px` 分别控制最小覆盖率和最大允许重投影误差。

## 离线求解与验证

仅采集：

```bash
python calibrate.py ... --no-solve
```

离线求解。棋盘格参数也可以从数据目录的 `session.json` 自动读取：

```bash
python calibrate_from_data.py data/2026-01-01_120000
```

比较五种 OpenCV 手眼算法：

```bash
python validate_eye_to_hand.py data/2026-01-01_120000
```

核心约束为：

```text
T_flange_target = inv(T_base_flange) @ T_base_camera @ T_camera_target
```

结果保存在 `T_base_camera.json`。投入使用前，应使用独立姿态和已知空间点验证。

## AprilTag（可选）

仍支持单个 `DICT_APRILTAG_36h11` 标签：

```bash
python calibrate.py ... --target-type apriltag --tag-size-mm 50
python calibrate_from_data.py data/my_session --target-type apriltag --tag-size-mm 50
```

## 从 CSV 导入机器人位姿

```csv
index,x,y,z,rx,ry,rz
0,412.3,-105.7,638.2,178.1,2.4,-89.6
```

```bash
python calibrate_from_data.py data/my_session \
  --poses-csv data/my_session/poses.csv \
  --position-unit mm --angle-unit deg
```

## 单应性平面抓取

平面抓取中，Z 不是由单应性额外计算出来的，而是由工作平面的几何约束确定。

因此抓取位置可以写成：

```bash
pixel = np.array([u, v, 1.0])
mapped = H_pixel_to_base_plane @ pixel

x_base = mapped[0] / mapped[2]
y_base = mapped[1] / mapped[2]
z_base = table_z
```

这里的 table_z 是桌面在机器人基座坐标系中的实际高度，可以通过机器人末端探测桌面上的多个点得到。
# 眼在手外 RGB 相机标定（V4L2）

本目录是独立实现，不修改 `eye_in_hand/`。支持 Linux 标准 V4L2 RGB 相机节点（例如 `/dev/video0`），标定结果为：

```text
T_base_camera：相机坐标系 -> 机器人基座坐标系
```

## 坐标系和安装要求

- RGB 相机必须固定在机器人外部，采集期间不能移动。
- 单个 AprilTag 36h11 必须刚性固定在法兰上，采集期间不能改变安装关系。
- 每次输入机器人控制器给出的 `T_base_flange`：法兰在基座坐标系下的位姿。
- 位姿旋转严格使用 SciPy `Rotation.from_euler("xyz", ...)` 的小写 `xyz` 外旋约定。必须确认控制器的欧拉角顺序、内旋/外旋定义和角度正方向一致；不一致时先转换。
- 至少采集 15 组，建议 20-30 组，位置和三个旋转轴都要有明显变化。

计算约束为：

```text
T_flange_target = inv(T_base_flange) @ T_base_camera @ T_camera_target
```

其中 `T_flange_target` 在所有样本中应保持不变，这也是验证程序采用的一致性指标。

## 1. 准备 RGB 相机内参

V4L2 只负责视频采集，不提供可靠内参。必须先用棋盘格等方法标定 RGB 相机，并按 `intrinsics.example.json` 创建内参文件。内参的分辨率必须与采集分辨率完全一致。

`coeffs` 使用 OpenCV顺序：

```text
[k1, k2, p1, p2, k3]（也可包含后续 OpenCV 畸变参数）
```

`intrinsics.example.json` 里的数字只是格式示例，不能直接用于实际标定。

## 2. 确认 V4L2 节点和格式

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

确保当前用户对节点有读写权限。若设备不支持 `MJPG`，可通过 `--pixel-format YUYV` 指定实际 FourCC。

## 3. 安装依赖并采集

```bash
python -m pip install -r requirements.txt

python calibrate.py \
  --device /dev/video0 \
  --intrinsics intrinsics.json \
  --width 1280 --height 720 --fps 30 \
  --pixel-format MJPG \
  --tag-size-mm 50 \
  --min-samples 20
```

启动后终端会打印浏览器预览地址。局域网中的电脑或手机可打开该地址确认标签是否完整清晰；拍摄和位姿输入仍在终端完成。预览默认监听 `0.0.0.0:8080`，可通过 `--preview-port` 修改，或使用 `--no-preview` 关闭。预览无认证，只应在可信局域网使用。

默认输入为 `x y z rx ry rz`，单位 `mm, deg`。例如：

```text
412.3 -105.7 638.2 178.1 2.4 -89.6
```

也可以改用米和弧度：

```bash
python calibrate.py ... --position-unit m --angle-unit rad
```

采集时每次先让机器人完全停止，按 Enter 捕获图像，再立即输入与该图像对应的法兰位姿。图像中只能出现一个相同 ID 的 AprilTag。

## 4. 离线求解和验证

仅采集、不立即计算：

```bash
python calibrate.py ... --no-solve
```

之后离线求解：

```bash
python calibrate_from_data.py data/2026-01-01_120000
```

比较 OpenCV 的五种手眼方法：

```bash
python validate_eye_to_hand.py data/2026-01-01_120000
```

主要输出为 `T_base_camera.json`。其中 `matrix_4x4` 左乘相机坐标齐次点即可得到基座坐标：

```text
p_base = T_base_camera @ p_camera
```

`validated` 默认保持为 `false`。应使用未参与标定的独立姿态和已知空间点进行现场验证后，再投入生产。

## 从 CSV 导入机器人位姿

CSV 表头：

```csv
index,x,y,z,rx,ry,rz
0,412.3,-105.7,638.2,178.1,2.4,-89.6
```

导入并计算：

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
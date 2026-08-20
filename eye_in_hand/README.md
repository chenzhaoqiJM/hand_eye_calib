# 手动手眼标定

本仓库现在只保留一条流程：

```text
RealSense D405 图像 + 手动记录的机械臂法兰位姿 -> T_flange_camera
```

也适用于其他 RealSense D4xx 系列相机，验证时使用的是 D405

脚本不会连接机械臂。你需要手动移动机械臂，从示教器或控制器读取法兰位姿，然后把该位姿输入脚本或整理成 CSV 导入。

## 安装依赖

```bash
python -m pip install -r requirements.txt
```

## 采集并计算

默认输入格式为 `x y z rx ry rz`，单位是 `mm, deg`，适合多数示教器显示格式。

```bash
python calibrate.py --tag-size-mm 77.6 --min-samples 20
```

启动后，终端会列出浏览器预览地址，例如：

```text
Browser preview:
  http://192.168.1.20:8080
  http://127.0.0.1:8080
```

同一局域网内的手机或电脑访问 `http://192.168.1.20:8080` 即可实时预览。请使用终端列出的实际局域网地址，不要在其他设备上使用 `127.0.0.1`。首次启动时，需要允许 Python 通过 Windows 防火墙的“专用网络”。浏览器页面只负责预览，拍摄和位姿输入仍在运行脚本的终端完成。

每组样本按下面步骤操作：

1. 手动移动机械臂到一个新姿态，等待机械臂完全静止。
2. 确保图像中只出现一个 AprilTag 36h11 标签。
3. 按 Enter 保存当前图像为 `frame_NNN.png`。
4. 输入当前法兰位姿：`x y z rx ry rz`。

脚本会把输入位姿转换为米和弧度，保存为 `pose_NNN.json`，然后使用 OpenCV PARK 方法计算 `T_flange_camera`。

常用参数：

```bash
python calibrate.py --position-unit m --angle-unit rad
python calibrate.py --no-solve
python calibrate.py --output-dir data/my_session
python calibrate.py --preview-port 8081
python calibrate.py --no-preview
```

预览默认监听所有网络接口的 TCP `8080` 端口。可用 `--preview-port` 修改端口，或用 `--no-preview` 完全关闭预览。该预览服务不提供登录认证，请只在可信的局域网中使用，不要把端口映射到公网。

## 从已有手动数据计算

如果你已经采集好了图像并手动记录了位姿，把它们放到同一个目录：

```text
data/my_session/
  intrinsics.json
  frame_000.png
  frame_001.png
  ...
  pose_000.json
  pose_001.json
  ...
```

每个 `pose_NNN.json` 的格式如下：

```json
{
  "x": 0.123,
  "y": 0.456,
  "z": 0.789,
  "rx": 0.0,
  "ry": 0.0,
  "rz": 0.0,
  "position_unit": "m",
  "angle_unit": "rad"
}
```

然后运行：

```bash
python calibrate_from_data.py data/my_session --tag-size-mm 50
```

也可以先从 CSV 导入手动记录的位姿：

```csv
index,x,y,z,rx,ry,rz
0,120.1,35.2,410.0,179.8,0.5,-91.2
1, ...
```

```bash
python calibrate_from_data.py data/my_session \
  --poses-csv data/my_session/poses.csv \
  --position-unit mm \
  --angle-unit deg \
  --tag-size-mm 50
```

## 输出结果

主要结果文件为：

```text
data/<session>/T_flange_camera.json
```

其中包含：

- `matrix_4x4`：从相机坐标系转换到法兰坐标系的 4x4 矩阵
- `xyz_m`：相机原点在法兰坐标系下的位置，单位为米
- `rpy_deg`：XYZ 欧拉角，单位为度
- `valid_indices`：实际参与计算的有效样本编号
- `validated: false`：表示该结果还没有通过独立现场验证

在用独立姿态验证之前，请保持 `validated: false`。

可选的一致性检查：

```bash
python validate_hand_eye.py data/my_session --tag-size-mm 50
```

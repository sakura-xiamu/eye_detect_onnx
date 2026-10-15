# 视线出屏检测

本地页面用摄像头实时判断画面里最大的一张脸是否在看屏幕外。人脸用 YuNet，视线用 MobileOne-S0（L2CS 分箱）。推理走 ONNX Runtime，在 Windows 上优先 DirectML，没有 DirectML 时用 CPU。

## 判定

角度单位是度。正俯仰朝画面上方，正偏航朝画面右。摄像头先镜像再检测，和预览一致。

默认窗口：

- 俯仰 `[-30, 5]`
- 偏航 `[-22, 22]`

窗口内是屏幕内，窗口外是屏幕外。页面上可以改，下一帧按新窗口判定。

摄像头对俯仰和偏航取最近 5 次成功检测的中值。离开窗口 3° 才变成屏幕外，回到窗口内侧 3° 才变回屏幕内。方向只在屏幕外时出现，按窗口本身判断上、下、左、右。连续 10 次没找到脸才清空；中间漏检仍显示上一次结果。

上传图片逐帧直出，不进入摄像头的平滑状态。

## 环境

面向 Windows，conda 环境名 `eye-gaze`，Python 3.11。

```powershell
conda env create -f environment.yml
conda activate eye-gaze
```

## 模型

- `weights/face_detection_yunet_2023mar.onnx`：OpenCV Zoo 的 YuNet，可放进仓库
- `weights/mobileone_s0_gaze.onnx`：启动时自动下载。权重来自 Gaze360，仅限研究、非商业使用，不能再分发，已写入 `.gitignore`

视线模型的正俯仰与页面一致（朝上）。模型的正偏航朝画面左，程序里取反，使页面上的正偏航朝右。

## 启动

```powershell
conda activate eye-gaze
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

浏览器打开 http://127.0.0.1:8001 ，点「打开摄像头」。预览按摄像头帧率播放，检测在后台进行，画面上画最近一次人脸框、视线箭头和屏幕内/外。

## 测试

```powershell
conda activate eye-gaze
python -m unittest discover -s tests -v
```

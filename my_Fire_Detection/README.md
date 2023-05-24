# my-Fire-Detection
modified from https://github.com/pedbrgs/Fire-Detection

my_detect.py是从detect.py修改而来，变为专门用于处理单张图片。

interface.py中写了传入原始图片得到带检测框图片和标注数据的函数接口。原仓库中的时间序列分析技术（例如适用于室内环境的temporal persistence技术，分析连续多帧中有多少帧是检出火焰的，达到某个阈值才判断为真，用于减少false positive）是专门针对视频的，单张图片无法使用该技术，因此在my_detect.py中删除了相关代码。可以在接口外部增加该技术。该技术代码在detect.py中体现。

yolov5s.pt是原仓库提供的训练好的模型，可直接用于火焰及烟雾检测

fire_000082.jpg是我加入的一张示例火灾图片

其他文件从原仓库照搬的

## 使用方法
如：用自带摄像头，采用temporal persistence技术
```python detect.py --source 0 --imgsz 640 --weights yolov5s.pt  --temporal persistence```

我个人实现的接口：interface.py中的start_fire_detect()函数。暂时较为混乱

目录
=================
- [入门指南](#入门指南)
  * [准备工作](#准备工作)
    + [使用pip安装依赖](#使用pip安装依赖)
    + [设置 Twilio token 和 SID](#设置-twilio-token-和-sid)
  * [运行方式](#运行方式)
    + [HTTP 方式](#http-方式)
    + [HTTPS 方式](#https-方式)
    + [访问已部署的应用](#访问已部署的应用)
  * [开发方式](#开发方式)
    + [开发UI界面](#开发ui界面)
      - [细节](#细节)
    + [开发报警逻辑](#开发报警逻辑)
    + [AI模型](#ai模型)
      - [使用AI模型接口](#使用ai模型接口)
  * [贡献您的代码](#贡献您的代码)

# 入门指南
> **注意**：如果您决定稍后要贡献代码，请先fork此项目。具体说明请见：[贡献您的代码](#贡献您的代码)
## 准备工作
> 注意: 要求使用 Python == 3.9.x
### 使用pip安装依赖
```pip install -r requirements.txt```

如果提示Killed，可以尝试：  
```pip install --no-cache-dir -r requirements.txt```

### 设置 Twilio token 和 SID
1. 在 [Twilio](https://twilio.com/) 注册一个免费账号
2. 从 [Twilio 控制台](https://www.twilio.com/console) 复制 auth token 和 account SID
3. 将 auth token 和 account SID 添加到环境变量中
    - ```export TWILIO_ACCOUNT_SID=<your account SID>```
    - ```export TWILIO_AUTH_TOKEN=<your auth token>```
    
## 运行方式
### HTTP 方式
1. 运行脚本 ```streamlit run home.py```
2. 打开浏览器并访问 ```http://localhost:8501```

### HTTPS 方式
1. 在**项目根目录**运行脚本 ```. script/start.sh```
2. 打开浏览器并访问 ```https://localhost:8000```

        注意：在开发模式下，浏览器会提示不安全连接，这是正常的

### 访问已部署的应用
- [fasmad](https://fasmad.streamlit.app)

## 开发方式

### 开发UI界面
> 请确保您已经大致了解了以下提供资源的内容：
- [Streamlit 入门](https://docs.streamlit.io/en/stable/getting_started.html)
- [Streamlit 组件](https://docs.streamlit.io/en/stable/api.html)

与UI界面相关的代码在 ```layout``` 以及 ```pages``` 目录下。```layout``` 目录下的代码用于定义页面当中某一模块的布局，```pages``` 目录下的代码用于定义整体页面的内容。

例如在 ```layout/fire_detection.py``` 中定义了火灾检测模块布局：
```python
class FireDetectionLayout(BaseLayout):
    @override
    def __init__(self):
        super().__init__()
        self.key = "fire-detection"

    @override
    def mount(self):
        st.title("火灾识别监控")

        def callback(frame):
            ...

        self.video_frame_callback = callback

        self.webrtc_ctx = create_webrtc_streamer(self.webrtc_streamer_attributes)

    @override
    def update(self):
        ...
```
以上代码是一个常见的样板代码(boilerplate)。

可以看到，```FireDetectionLayout``` 继承了 ```BaseLayout``` 类，```BaseLayout``` 类提供了关于视频检测布局的基础属性，诸如 ```webrtc_streamer_attributes``` 与 ```streaming``` 等。

而 ```mount``` 是一个关键的方法，它用于与 ```page``` 当中的页面进行"挂钩"，使得页面能渲染此布局模块。

您可以使用 ```update``` 方法执行布局模块的更新逻辑。详细使用方式请参考 ```pages``` 目录下任一页中的代码。~~**注意 ```update``` 方法的内部实现应该写在 ```update_impl``` 方法中**，这是为了保证 ```update``` 更新布局状态时会考虑是否正在进行视频串流。~~   
  > **更新**：现在请直接将更新逻辑写在 ```update``` 方法中。

#### 细节

- 如果您想对每一个视频帧进行处理，例如使用AI模型进行检测。请定义 ```self.video_frame_callback``` 。它是一个回调函数，它会在视频帧更新时被调用。

    > 虽然 ```self.video_frame_callback``` 与 ```update``` 同属于更新函数，但它们运行在不同的线程中，所以在使用同一资源时**务必考虑线程安全**。

- ```self.metadata_queue_ref``` 是一个队列，它用于存储视频帧的元数据。例如视频帧的检测结果。您可以使用 ```self.metadata_queue_ref.get()``` 获取队列中的元数据。这在前后端交互信息时非常有用。
    > 注意 ```self.metadata_queue_ref``` 属于服务型队列。意味着**它是共享而不是单独属于任何一个布局模块。** 布局模块只是保存了它的引用。

### 开发报警逻辑

- 在 ```module/alarm_agent.py``` 中实现报警逻辑。

### AI模型

#### 使用AI模型接口
- [火灾及烟雾识别模型接口说明](https://github.com/sweetbrulee/fasmad-streamlit/blob/master/service/kernel/firedetection/README.md)

### 贡献您的代码
> 请参考以下教程：
[教程](https://github.com/firstcontributions/first-contributions/blob/main/translations/README.zh-cn.md)

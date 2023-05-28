# 入门指南
## 准备工作
> 注意: 要求使用 Python == 3.9.x
### 使用pip安装依赖
```pip install -r requirements.txt```

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
    def __init__(self):
        super().__init__()
        self.key = "fire-detection"

    def mount(self):
        st.title("火灾识别监控")

        def callback(frame):
            ...

        self.video_frame_callback = callback

        self.webrtc_ctx = create_webrtc_streamer(self.webrtc_streamer_attributes)

    def update(self):
        ...
```
以上代码是一个常见的样板代码(boilerplate)。

可以看到，```FireDetectionLayout``` 继承了 ```BaseLayout``` 类，```BaseLayout``` 类提供了关于视频检测布局的基础属性，诸如 ```webrtc_streamer_attributes``` 与 ```streaming``` 等。

而 ```mount``` 是一个关键的方法，它用于与 ```page``` 当中的页面进行"挂钩"，使得页面能渲染此布局模块。

```update``` 方法用于定义布局模块的循环式更新逻辑。详细使用方式请参考 ```pages``` 目录下任一页中的代码。

> **请注意：```update``` 方法的内部实现应该写在 ```update_impl``` 方法中**，这是为了保证 ```update``` 更新布局状态时会考虑是否正在进行视频串流。


### 开发后端逻辑

### 开发AI模型
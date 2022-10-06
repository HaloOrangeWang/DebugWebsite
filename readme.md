## 运行方法

我编写这个程序用的环境是 `Python 3.8`

#### 1. 安装依赖包

`pip install -r requirements.txt`

另外，这个程序还需要安装 `mongodb` 才能正常运行。

#### 2. 生成知识库

```
cd DebugTrain
python main.py
```

#### 3. 启动词典程序（网页）

```
cd DebugServer
python manage.py runserver 0.0.0.0:8000
```

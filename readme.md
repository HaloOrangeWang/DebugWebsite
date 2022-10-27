# 将Python中各类错误的解决方法做成词典

---

实现原理介绍：<https://www.bilibili.com/av646977726>

测试版服务器：<http://118.195.242.218:8000/>

>
> 如果觉得这个程序做得不错，还请star+分享支持一下哦！
>

---

## 简介

这个工程的目标是，当你在编程时遇到错误时，能够像查阅词典一样，一步到位地找到解决方案，而不用经历繁琐的搜索、阅读博客的过程。

为此，程序使用了BILSTM、CRF等方法，自动识别博客中的错误信息和解决方法，通过自动分析大量文章，构建了一个知识库，这样用户在输入错误信息时，程序就能从知识库中找到可能的解决方法了。

---

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
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

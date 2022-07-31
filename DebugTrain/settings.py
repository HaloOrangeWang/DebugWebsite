# 运行模式：Train表示进行只根据部分文章生成相关信息并验证准确率，Generate表示根据所有文章生成相关信息
MODE = "Generate"
# 是否使用ML
ML = False

# 输出数据的配置
DB_SERVER = "mongodb://localhost:27017"
DB_NAME = "DebugWebsite"
DB_COLLECTION = "PyArticle"

# ArticleID的设置规则
AID_BASE = {'cnblogs': 0, 'jianshu': 100000, 'oschina': 200000}

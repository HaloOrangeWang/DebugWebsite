# 输入数据的配置
DB_SERVER = "mongodb://localhost:27017"
DB_NAME = "DebugWebsite"
DB_COLLECTION = "PyArticle"

# ArticleID的设置规则
AID_BASE = {'cnblogs': 0, 'jianshu': 100000, 'oschina': 200000, '51cto': 300000, 'jb51': 400000, 'csdn': 500000}

# 文章的检索
COMMON_SSTR_MIN_LEN = 4  # 两个解决信息的公共子串的最小长度（一个英文单词算作2个长度）
SSTR_PAGE_LIMIT = 5  # 最多显示多少个公共子串

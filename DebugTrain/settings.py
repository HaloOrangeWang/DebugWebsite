# 运行模式：Train表示进行只根据部分文章生成相关信息并验证准确率，Generate表示根据所有文章生成相关信息
MODE = "Train"
# 是否使用ML
ML = True
# 训练词向量，还是直接使用文件中训练好的结果
WV_LOAD = True
WV_DIR = "../TrainData/WV_Output"

# 输出数据的配置
DB_SERVER = "mongodb://localhost:27017"
DB_NAME = "DebugWebsite"
DB_COLLECTION = "PyArticle"

# ArticleID的设置规则
AID_BASE = {'cnblogs': 0, 'jianshu': 100000, 'oschina': 200000, '51cto': 300000, 'jb51': 400000, 'csdn': 500000}

# 字符编码的一些常量
BOS_IDX = 0  # 表示一行开始
EOS_IDX = 1  # 表示一行结束
PAD_IDX = 2  # 一行结束后，用于填充向量的字符内容
B_IDX = 3  # 判断为“开始是报错/解决问题信息”的标志位
I_IDX = 4  # 判断为“正在是报错/解决问题信息”的标志位
E_IDX = 5  # 判断为“报错/解决问题信息结束”的标志位
O_IDX = 6  # 判断为“不是报错/解决问题信息”的标志位
OUTPUT_DIC_SIZE = (O_IDX + 1)
# 哪些数据用于训练，哪些数据用于测试（例如，如果start=0，ratio=0.2，那么代表样本中aid最小的20%用于测试，其余的用于训练）
TEST_DATA_START = 0.8
TEST_DATA_RATIO = 0.2
# # 训练时，每次输入长度为多少的向量
# TRAIN_TEXT_LEN = 50

TRUE_IDX = 3  # 训练解决方案的起始/终止区间时，需要对段落进行分类。此数值表明是“解决方案的起始/终止”的信号
FALSE_IDX = 4  # 此变量表明不是“解决方案的起始/终止”的信号

# 特殊词汇的编码
SPC_CHN_WRD = 0  # 中文低频词
SPC_ENG_WRD = 1  # 英文低频词
DIGIT_WRD = 2  # 数字
ERR_ENG_WRD = 3  # 偏向于报错信息的英文词
ERC_ENG_WRD = 4  # 80%偏向于报错信息，20%偏向于代码段的英文词
SEN_ENG_WRD = 5  # 偏向于场景信息的英文词
COS_ENG_WRD = 6  # 80%偏向于代码段，20%偏向于场景信息的英文词
COD_ENG_WRD = 7  # 偏向于代码段的英文词
LNK_ENG_WRD = 8  # 偏向于地址信息的英文词
CHN_PUNC_WRD = 9  # 全角标点符号
ENG_PUNC_WRD = 10  # 半角标点符号
NEW_PARA_WRD = 11  # 段落切换的标志
NUM_SPC_WORDS = NEW_PARA_WRD + 1

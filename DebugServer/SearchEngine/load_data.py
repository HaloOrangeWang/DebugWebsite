from configs import *
import pandas as pd
import pymongo
import os


class Data1Article:

    def __init__(self):
        self.title = ""
        self.content = ""
        self.err_msg = ""
        self.scenes = []
        self.solves = []
        self.link = ""


def load_base_data():
    base_data = dict()
    # 1.连接数据库
    conn = pymongo.MongoClient(DB_SERVER)
    db = conn[DB_NAME]
    db_col = db[DB_COLLECTION]
    # 2.找到想要的信息
    db_data = db_col.find({})
    for raw_data in db_data:
        aid = raw_data['_id']
        data = Data1Article()
        data.err_msg = raw_data['err_msg']
        data.scenes = raw_data['scene']
        data.solves = raw_data['solve']
        base_data[aid] = data
    # 3.返回数据
    conn.close()
    return base_data


def load_articles(base_data):
    """读取文章的标题和正文"""
    for site in ['cnblogs', 'jianshu', 'oschina']:
        input_path = '../SpiderData/%s/Clean' % site
        f = open(os.path.join(input_path, 'titles.txt'), 'r', encoding='utf8')
        for line in f.readlines():
            aid_tmp = int(line[:line.find('  ')])
            aid = aid_tmp + AID_BASE[site]
            article_title = line[line.find('  ') + 2:]
            if aid in base_data:
                base_data[aid].title = article_title
                text_filename = os.path.join(os.path.join(input_path, '%05d.txt' % aid_tmp))
                f2 = open(text_filename, 'r', encoding='utf8')
                text = f2.read()
                f2.close()
                base_data[aid].content = text
        f.close()
    return base_data


def load_link(base_data):
    """获取每篇文章aid对应的链接"""
    for site in ['cnblogs', 'jianshu', 'oschina']:
        filename = '../SpiderData/%s/file_list.xlsx' % site
        article_df = pd.read_excel(filename, index_col=0)
        for index in article_df.index:
            aid = int(index) + AID_BASE[site]
            url = article_df.loc[index, 'href']
            if aid in base_data:
                base_data[aid].link = url
    return base_data

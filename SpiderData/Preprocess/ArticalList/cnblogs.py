from settings import *
import pandas as pd
from funcs import GetData, is_exp_article
import os


# 文章列表
file_list_dir = '../../SpiderData/cnblogs'
# 文章列表的文件路径
file_list_marks = [
    '201801-201802',
    '201803-201804',
    '201805-201806',
    '201807-201808',
    '201809-201810',
    '201811-201812',
    '201901-201902',
    '201903-201904',
    '201905-201906',
    '201907-201908',
    '201909-201910',
    '201911-201912',
    '202001-202002',
    '202003-202004',
    '202005-202006',
    '202007-202008',
    '202009-202010',
    '202011-202012',
    '202101-202102',
    '202103-202104',
    '202105-202106',
    '202107-202108',
    '202109-202110',
    '202111-202112',
    '202201-202202',
    '202203-202204',
]


class GetDataCnblogs(GetData):

    def __init__(self):
        super(GetDataCnblogs, self).__init__()
        self.output_dir = file_list_dir

    def parse_list(self):
        # 1.获取文章列表
        article_list = []
        for mark in file_list_marks:
            f = os.path.join(file_list_dir, 'Origin', '[%s] 搜索结果提示 - 博客园找找看.xlsx' % mark)
            article_list.append(pd.read_excel(f))
        all_articles = pd.concat(article_list, ignore_index=True)
        # 2.筛选出符合要求的文章列表
        all_articles['is_exp'] = all_articles['标题'].apply(is_exp_article)
        all_articles['title'] = all_articles['标题']
        all_articles['href'] = all_articles['标题链接']
        all_articles['yr'] = all_articles['信息'].apply(lambda x: int(x[:4]))
        exp_articles = all_articles[all_articles['is_exp'] == True]
        self.article_df = exp_articles[['href', 'yr', 'title']]


def main():
    download_obj = GetDataCnblogs()
    download_obj.parse_list()
    download_obj.after_parse_list()
    download_obj.download()


if __name__ == '__main__':
    main()

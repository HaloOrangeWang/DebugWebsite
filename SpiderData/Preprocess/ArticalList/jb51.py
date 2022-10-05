from settings import *
import pandas as pd
from funcs import GetData, is_exp_article
import os


# 文章列表
file_list_dir = '../../jb51'
# 文章列表的文件路径
file_list_marks = [
    'python报错_站内搜索',
    'python出错_站内搜索',
    'python错误_站内搜索',
]


def get_yr(s):
    ed_dx = s.rindex("-", 0, s.rindex("-"))
    return int(s[ed_dx - 4: ed_dx])


class GetDataJB51(GetData):

    def __init__(self):
        super(GetDataJB51, self).__init__()
        self.output_dir = file_list_dir

    def parse_list(self):
        # 1.获取文章列表
        article_list = []
        for mark in file_list_marks:
            f = os.path.join(file_list_dir, '%s.xlsx' % mark)
            article_list.append(pd.read_excel(f))
        all_articles = pd.concat(article_list, ignore_index=True)
        # 2.筛选出符合要求的文章列表
        all_articles['is_exp'] = all_articles['标题'].apply(is_exp_article)
        all_articles['title'] = all_articles['标题']
        all_articles['href'] = all_articles['标题链接']
        all_articles['yr'] = all_articles['cshowurl'].apply(get_yr)
        exp_articles = all_articles[all_articles['is_exp'] == True]
        self.article_df = exp_articles[['href', 'yr', 'title']]


def main():
    download_obj = GetDataJB51()
    download_obj.parse_list()
    download_obj.after_parse_list()
    download_obj.download()


if __name__ == '__main__':
    main()

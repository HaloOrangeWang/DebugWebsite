from settings import *
import pandas as pd
from funcs import GetData, is_exp_article
import os


# 文章列表
file_list_dir = '../../SpiderData/jianshu'
# 文章列表的文件路径
file_lists = [
    '搜索 - 简书（近三月）.xlsx',
    '搜索 - 简书（全部）.xlsx'
]


def parse_time(time_str):
    """由于简书的修改时间没有具体到年份，因此近似处理"""
    if '年' in time_str:
        return 2022 - int(time_str[:time_str.find('年')])
    else:
        return 2022


class GetDataJianshu(GetData):

    def __init__(self):
        super(GetDataJianshu, self).__init__()
        self.output_dir = file_list_dir

    def parse_list(self):
        # 1.获取文章列表
        article_list = []
        for f0 in file_lists:
            f = os.path.join(file_list_dir, f0)
            article_list.append(pd.read_excel(f))
        all_articles = pd.concat(article_list, ignore_index=True)
        # 2.筛选出符合要求的文章列表
        all_articles['is_exp'] = all_articles['标题'].apply(is_exp_article)
        all_articles['title'] = all_articles['标题']
        all_articles['href'] = all_articles['标题链接']
        all_articles['yr'] = all_articles['时间'].apply(parse_time)
        exp_articles = all_articles[all_articles['is_exp'] == True]
        self.article_df = exp_articles[['href', 'yr', 'title']]


def main():
    download_obj = GetDataJianshu()
    download_obj.parse_list()
    download_obj.after_parse_list()
    download_obj.download()


if __name__ == '__main__':
    main()

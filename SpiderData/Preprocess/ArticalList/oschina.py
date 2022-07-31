from settings import *
import pandas as pd
from funcs import GetData, is_exp_article
import os


# 文章列表路径
file_list_dir = '../../SpiderData/oschina'


def del_comment(url):
    """去除oschina文件列表末尾的 #comments """
    if '#comments' in url:
        return url[:url.rfind('#')]
    return url


class GetDataOschina(GetData):

    def __init__(self):
        super(GetDataOschina, self).__init__()
        self.output_dir = file_list_dir

    def parse_list(self):
        # 1.获取文章列表
        f = os.path.join(file_list_dir, 'python报错 - 开源搜索 - OSCHINA - 中文开源技术交流社区.xlsx')
        all_articles = pd.read_excel(f)
        # 2.筛选出符合要求的文章列表
        all_articles['is_exp'] = all_articles['header'].apply(is_exp_article)
        all_articles['title'] = all_articles['header']
        all_articles['href'] = all_articles['正文链接'].apply(del_comment)
        all_articles['yr'] = all_articles['信息1'].apply(lambda x: int(x[:4]))
        exp_articles = all_articles[all_articles['is_exp'] == True]
        self.article_df = exp_articles[['href', 'yr', 'title']]


def main():
    download_obj = GetDataOschina()
    download_obj.parse_list()
    download_obj.after_parse_list()
    download_obj.download()


if __name__ == '__main__':
    main()

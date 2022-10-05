from settings import *
import pandas as pd
from funcs import GetData, is_exp_article
import os


# 文章列表
file_list_dir = '../../csdn'
# 文章列表的文件路径
file_list_marks = [
    'https___so.csdn.net_so_search_spm=1001.2100.3001.4498_q=python出错_t=blog_u=_s=new',
    'https___so.csdn.net_so_search_spm=1001.2100.3001.4501_q=python报错_t=blog_u=_s=hot',
    'https___so.csdn.net_so_search_spm=1001.2100.3001.4501_q=python报错_t=blog_u=_s=0',
    'https___so.csdn.net_so_search_spm=1001.2100.3001.4501_q=python报错_t=blog_u=_s=new',
    'https___so.csdn.net_so_search_spm=1001.2100.3001.4498_q=python出错_t=blog_u=',
    'https___so.csdn.net_so_search_q=python错误_t=blog_u=_s=new_urw=',
    'https___so.csdn.net_so_search_q=python错误_t=blog_u=_s=0_urw=',
]


class GetDataCSDN(GetData):

    def __init__(self):
        super(GetDataCSDN, self).__init__()
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
        all_articles['yr'] = all_articles['时间'].apply(lambda x: int(x[:4]))
        exp_articles = all_articles[all_articles['is_exp'] == True]
        exp_articles = exp_articles.drop_duplicates(['href'])  # 去除重复的文章
        self.article_df = exp_articles[['href', 'yr', 'title']]


def main():
    download_obj = GetDataCSDN()
    download_obj.parse_list()
    download_obj.after_parse_list()
    download_obj.download()


if __name__ == '__main__':
    main()

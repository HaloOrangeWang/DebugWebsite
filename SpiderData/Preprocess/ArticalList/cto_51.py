from settings import *
import pandas as pd
from funcs import GetData, is_exp_article
import os


# 文章列表路径
file_list_dir = '../../51cto'


class GetData51CTO(GetData):

    def __init__(self):
        super(GetData51CTO, self).__init__()
        self.output_dir = file_list_dir

    def parse_list(self):
        # 1.获取文章列表
        f = os.path.join(file_list_dir, '51CTO搜索-51CTO.COM.xlsx')
        all_articles = pd.read_excel(f)
        # 2.筛选出符合要求的文章列表
        all_articles['is_exp'] = all_articles['标题1'].apply(is_exp_article)
        all_articles['title'] = all_articles['标题1']
        all_articles['href'] = all_articles['描述_链接']
        all_articles['yr'] = all_articles['标题'].apply(lambda x: int(x[:4]))
        exp_articles = all_articles[all_articles['is_exp'] == True]
        self.article_df = exp_articles[['href', 'yr', 'title']]


def main():
    download_obj = GetData51CTO()
    download_obj.parse_list()
    download_obj.after_parse_list()
    download_obj.download()


if __name__ == '__main__':
    main()

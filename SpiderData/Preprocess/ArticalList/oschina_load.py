from settings import *
import pandas as pd
from funcs import GetData, is_exp_article
import os


# 文章列表
file_list_dir = '../../SpiderData/oschina'


class GetDataCnblogsLoad(GetData):

    def __init__(self):
        super(GetDataCnblogsLoad, self).__init__()
        self.output_dir = file_list_dir

    def parse_list(self):
        # 1.获取文章列表
        f = os.path.join(self.output_dir, 'file_list_tmp.xlsx')
        self.article_df = pd.read_excel(f, index_col=0)

    def after_parse_list(self):
        pass


def main():
    download_obj = GetDataCnblogsLoad()
    download_obj.parse_list()
    download_obj.after_parse_list()
    download_obj.download()


if __name__ == '__main__':
    main()

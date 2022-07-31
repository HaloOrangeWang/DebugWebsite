from settings import *
import pandas as pd
import urllib.request
import traceback
import time
import os


class GetData:

    def __init__(self):
        self.article_df = pd.DataFrame(columns=['title', 'href', 'yr'])
        self.output_dir = str()

    def parse_list(self):
        """解析文章列表文件，获取以解决报错问题的文章列表"""
        pass

    def after_parse_list(self):
        """解析文件列表完成之后，保存文件列表，并创建HTML文件所在的文件夹"""
        self.article_df.to_excel(os.path.join(self.output_dir, 'file_list.xlsx'))
        if not os.path.exists(os.path.join(self.output_dir, 'Main')):
            os.makedirs(os.path.join(self.output_dir, 'Main'))

    def download(self):
        header = "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36"
        for row in self.article_df.index:
            try:
                url = self.article_df.loc[row, 'href']
                output_file = os.path.join(self.output_dir, 'Main', '%05d.html' % row)
                # 获取博客数据
                req = urllib.request.Request(url, headers={'User-Agent': header})
                rsp = urllib.request.urlopen(req)
                output_data = rsp.read()
                rsp.close()
                # 输出博客数据
                f = open(output_file, 'wb')
                f.write(output_data)
                f.close()
            except Exception:
                print(traceback.format_exc())
                time.sleep(5)
            time.sleep(10)


def is_exp_article(title):
    """判断一个文章是不是解决报错问题的文章"""
    title2 = title.lower()
    for t in Keywords:
        if t in title2:
            return True
    return False

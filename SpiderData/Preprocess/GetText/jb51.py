import bs4.element
from bs4 import BeautifulSoup
from funcs import HTMLParser, trim_n


class JB51Parser(HTMLParser):

    def __init__(self):
        super(JB51Parser, self).__init__()
        self.input_dir = '../../../SpiderData/jb51/Main'
        self.output_dir = '../../../SpiderData/jb51/Clean'

    def can_output(self, p):
        """
        判断一个html标签是否可以输出。判断标准为：
        1. 内容不能为空
        2. 如果是div标签，行数必须在3行以内
        3. 如果有两个空行，则分别输出
        如果可以输出的话，就返回修剪好的字符串
        """
        p_text_input = p.text.split('\n\n')
        p_text_output = [None for t in range(len(p_text_input))]
        for text_it in range(len(p_text_input)):
            p_text = trim_n(p_text_input[text_it])
            if not p_text:
                break
            # if p.name == 'code':
            #     if p.parent.name != 'pre':
            #         break
            p_text_output[text_it] = p_text
        return p_text_output

    def parse_1html(self, input_file, output_file):
        f2 = open(output_file, 'w', encoding='utf8')
        # 从文章中解析出html文本
        f = open(input_file, 'r', encoding='utf8')
        f1 = f.read()
        html_data = BeautifulSoup(f1, 'html.parser')
        f.close()
        # class为article-detail的div区块为正文。把里面标签为p、h1-6、div的内容挑出来
        html_text = html_data.select('#content')
        p_list = []
        if len(html_text) == 1:
            for item in html_text[0].children:
                if type(item) is bs4.element.Tag and (not ("id" in item.attrs and (item["id"] == "navCategory" or item["id"] == "xingquarticle"))) and (not ("class" in item.attrs and "art_xg" in item["class"])):
                    if item.name in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'pre']:
                        p_list.append(item)
                    p_list.extend([t for t in item.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'pre'])])
            # p_list = html_text[0].find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'code'])
            for p in p_list:
                if p.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:  # 标题行与前面段落的末尾之间空2行
                    f2.write('\n')
                for br_tag in p.findAll('br'):
                    br_tag.replace_with('\n')
                p_text_list = self.can_output(p)
                for text in p_text_list:
                    if text:
                        f2.write(text)
                        f2.write('\n\n')
        else:
            print('html text error for ', input_file)
        f2.close()

    def parse_1title(self, input_file):
        # 从文章中解析出html文本
        f = open(input_file, 'r', encoding='utf8')
        f1 = f.read()
        html_data = BeautifulSoup(f1, 'html.parser')
        # a2 = BeautifulSoup(f1, 'lxml')
        f.close()
        header_tag = html_data.select('.title')
        header_tag_final = None
        for t in header_tag:
            if t.name[0] == 'h':
                header_tag_final = t
        if header_tag_final is None:
            header_tag = html_data.select('#title')
            for t in header_tag:
                if t.name[0] == 'h':
                    header_tag_final = t
        if header_tag_final is not None:
            text = header_tag_final.text
            if '\n' in text:
                return str()
            else:
                return text
        else:
            print('html title error for ', input_file)
            return str()


def main():
    parser_obj = JB51Parser()
    parser_obj.parse_all_data()
    parser_obj.parse_all_title()


if __name__ == '__main__':
    main()

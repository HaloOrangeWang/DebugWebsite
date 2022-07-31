from bs4 import BeautifulSoup
from funcs import HTMLParser, is_not_empty, trim_n


class CnblogsParser(HTMLParser):

    def __init__(self):
        super(CnblogsParser, self).__init__()
        self.input_dir = '../../../SpiderData/cnblogs/Main'
        self.output_dir = '../../../SpiderData/cnblogs/Clean'

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
            if p.name == 'div':
                if p_text.count('\n') >= 3:
                    break
            p_text_output[text_it] = p_text
        return p_text_output

    def parse_1html(self, input_file, output_file):
        f2 = open(output_file, 'w', encoding='utf8')
        # 从文章中解析出html文本
        f = open(input_file, 'r', encoding='utf8')
        f1 = f.read()
        html_data = BeautifulSoup(f1, 'html.parser')
        # a2 = BeautifulSoup(f1, 'lxml')
        f.close()
        # id为cnblogs_post_body的div区块为正文。获取其中所有标签为p的内容
        html_text = html_data.select('#cnblogs_post_body')
        if len(html_text) == 1:
            p_list = html_text[0].find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div'])
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
        header_tag = html_data.select('span[role="heading"]')
        if len(header_tag) == 1:
            return header_tag[0].text
        else:
            print('html title error for ', input_file)
            return str()


def main():
    parser_obj = CnblogsParser()
    parser_obj.parse_all_data()
    parser_obj.parse_all_title()


if __name__ == '__main__':
    main()

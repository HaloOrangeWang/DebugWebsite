import os


def is_not_empty(input_str):
    s = input_str.replace('\t', '').replace(' ', '').replace('\r', '').replace('\n', '').replace(chr(0xa0), '')
    return bool(s)


def trim_n(input_str):
    """去除一个字符串开头结尾的\r, \n, \t, 空格等字符。并把中间的\n\n变成\n"""
    start_pos = -1
    end_pos = -1
    for t in range(len(input_str)):
        if input_str[t] not in ['\t', '\r', '\n', ' ', chr(0xa0)]:
            start_pos = t
            break
    for t in range(len(input_str) - 1, -1, -1):
        if input_str[t] not in ['\t', '\r', '\n', ' ', chr(0xa0)]:
            end_pos = t + 1
            break
    if start_pos == -1 or end_pos == -1:
        return str()
    return input_str[start_pos: end_pos]


class HTMLParser:

    def __init__(self):
        self.input_dir = str()
        self.output_dir = str()

    def parse_1html(self, input_file, output_file):
        """解析一个HTML文件"""
        pass

    def parse_1title(self, input_file):
        """从html文件中解析出文件标题"""
        pass

    def parse_all_title(self):
        # 创建要输出的文件
        output_file = os.path.join(self.output_dir, 'titles.txt')
        fs = open(output_file, 'w', encoding='utf8')
        for f in os.listdir(self.input_dir):
            input_file = os.path.join(self.input_dir, f)
            title = self.parse_1title(input_file)
            fs.write('%s  %s\n' % (f.replace('.html', ''), title))
        fs.close()

    def parse_all_data(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        for f in os.listdir(self.input_dir):
            input_file = os.path.join(self.input_dir, f)
            output_file = os.path.join(self.output_dir, f.replace('html', 'txt'))
            # 解析这个html文件，将正文解析出来
            self.parse_1html(input_file, output_file)

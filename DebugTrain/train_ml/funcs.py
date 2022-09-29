def has_chn_chr(s):
    """判断一个字符串是否包含中文"""
    for ch in s:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


def has_eng_chr(s):
    """判断一个字符串是否包含英文"""
    for ch in s:
        if u'\u0041' <= ch <= u'\u005a' or u'\u0061' <= ch <= u'\u007a':
            return True
    return False


def has_chn_or_eng_chr(s):
    """判断一个字符串是否包含中文或英文或数字"""
    for ch in s:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
        if u'\u0041' <= ch <= u'\u005a' or u'\u0061' <= ch <= u'\u007a':
            return True
    return False


def has_chn_or_eng_or_digit_chr(s):
    """判断一个字符串是否包含中文、英文或数字"""
    for ch in s:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
        if u'\u0041' <= ch <= u'\u005a' or u'\u0061' <= ch <= u'\u007a':
            return True
        if u'\u0030' <= ch <= u'\u0039':
            return True
    return False


def is_chn_eng_number(ch):
    """判断一个字符是否为中文、英文或数字"""
    if u'\u4e00' <= ch <= u'\u9fff':
        return True
    if u'\u0041' <= ch <= u'\u005a' or u'\u0061' <= ch <= u'\u007a':
        return True
    if u'\u0030' <= ch <= u'\u0039':
        return True
    return False


def is_eng_number(ch):
    """判断一个字符是否为英文或者数字"""
    if u'\u0041' <= ch <= u'\u005a' or u'\u0061' <= ch <= u'\u007a':
        return True
    if u'\u0030' <= ch <= u'\u0039':
        return True
    return False


def get_last_line_by_para(text):
    """
    获取一段文字中，每个段落最后一行的行标
    :param text:
    :return:
    """
    # 1.将这段文字拆分成若干个段落。大段落的分隔处要标识出来
    line_dx_list = []
    paragraphs_large = text.split('\n\n\n')
    for t in range(len(paragraphs_large)):
        paragraphs = paragraphs_large[t].split('\n\n')
        last_line = (1 + paragraphs[0].count('\n') if t == 0 else (line_dx_list[-1] + 3))
        for t0 in range(len(paragraphs)):
            if t0 != 0:
                last_line += (2 + paragraphs[t0].count('\n'))
            line_dx_list.append(last_line)
    return line_dx_list


def get_all_line_no(articles):
    """获取所有文章中，每个段落对应的行标的列表"""
    line_no_by_para = {aid: [] for aid in articles}
    for aid in articles:
        text = articles[aid].text.rstrip("\n")
        line_no_by_para[aid] = get_last_line_by_para(text)
    return line_no_by_para


class WordVec1Para:
    """一段话的词向量"""

    def __init__(self):
        self.vec = None  # 词向量的内容。结构为"构成词向量的词在wv中的索引值。英文词和标点符号单独标注"
        self.vec_to_text_dx_list = []  # 词向量中的每一个词，对应原字符串的索引值。
        self.vec_len_list = []  # 词向量中的每一个词的长度
        self.is_first_para = False  # 这一段话是否为一个大段的第一段话
        self.paragraph_dx = -1  # 这一段话为原文章中的第几段


class WordVec1Article:
    """一篇文章的词向量"""

    def __init__(self):
        self.title_c = WordVec1Para()  # 标题的词向量
        self.text_c = []  # 正文的词向量


# def get_str_dx(paragraph, sstr):
#     """
#     找到一个字符串在原始的字符串列表中的位置。这个子串的位置可能是不连续的
#     :param paragraph: 一个段落原始的字符串
#     :param sstr: 要寻找的子串
#     :return: 这段子串在原始字符串中的位置
#     """
#     l2 = [[0 for t0 in range(len(paragraph))] for t in range(len(sstr))]
#     first_dx = 0
#     for t in range(len(sstr)):
#         if t == 0:
#             st_dx = 0
#         else:
#             st_dx = first_dx + 1
#         is_find = False
#         for t0 in range(st_dx, len(paragraph)):
#             if paragraph[t0] == sstr[t]:
#                 l2[t][t0] = 1
#                 if not is_find:
#                     first_dx = t0
#                     is_find = True
#     res = []
#     last_dx = -1
#     for t0 in range(len(paragraph)):
#         if l2[len(sstr) - 1][t0] == 1:
#             last_dx = t0 - 1
#             res.append(t0)
#             break
#     assert last_dx != -1
#     for t in range(len(sstr) - 2, -1, -1):
#         is_find = False
#         for t0 in range(last_dx, -1, -1):
#             if l2[t][t0] == 1:
#                 is_find = True
#                 res.append(t0)
#                 last_dx = t0 - 1
#                 break
#         assert is_find is True
#     return reversed(res)

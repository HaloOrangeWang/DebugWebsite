def is_chn_chrs(ch):
    """判断一个字符是否为中文"""
    if (u'\u4e00' <= ch <= u'\u9fff') or (ch in "（）【】{}“”：；，。？"):
        return True
    return False

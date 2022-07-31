def judge_empty_content(s):
    """判断一个字符串是否只有空格，以及换行符等"""
    for t in s:
        if t not in [' ', chr(0xa0), '\t', '\r', '\n']:
            return False
    return True


def judge_negative_phase(msgs, negative_words):
    is_negative_phase = False
    for t in range(len(msgs)):
        if msgs[t].word.lower().startswith('un'):
            is_negative_phase = True
        for word2 in negative_words:
            if word2 in msgs[t].word.lower():
                is_negative_phase = True
                break
    return is_negative_phase


def has_chn_chrs(s):
    """判断一个字符串是否包含中文"""
    for ch in s:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


def all_chn_chrs(s):
    """判断一个字符串是否全部为中文"""
    for ch in s:
        if (not u'\u4e00' <= ch <= u'\u9fff') and (ch not in "（）【】{}“”：；，。？"):
            return False
    return True


def calc_eng_chr_ratio(s):
    """计算一段字符串内英文和数字字符的占比"""
    eng_cnt = 0
    chn_cnt = 0
    for ch in s:
        if u'\u4e00' <= ch <= u'\u9fff' or ch in "（）【】{}“”：；，。？":
            chn_cnt += 1
        elif ch not in [' ', chr(0xa0), '\t', '\r', '\n']:
            eng_cnt += 1
    return eng_cnt / (eng_cnt + chn_cnt)


def get_split_str(text, split_chars):
    """将字符串按照标点符号拆分成若干个句子，分隔符有逗号、句号等，但如果标点符号在括号里，那么就不算数"""
    # 获取句子分隔符的位置
    split_dx = list()
    bracket_dx = -1
    for t in range(len(text)):
        if text[t] in split_chars and bracket_dx == -1:
            split_dx.append(t)
        if text[t] == '（':
            bracket_dx = t
        if text[t] == '）':
            bracket_dx = -1
    # 将字符串拆分成句子
    if len(split_dx) == 0:
        return [text]
    sentences = list()
    for t in range(len(split_dx) + 1):
        if t == 0:
            sentence = text[0: split_dx[t]]
        elif t <= len(split_dx) - 1:
            sentence = text[split_dx[t - 1] + 1: split_dx[t]]
        else:
            sentence = text[split_dx[t - 1] + 1:]
        if len(sentence) != 0:
            sentences.append(sentence)
    return sentences

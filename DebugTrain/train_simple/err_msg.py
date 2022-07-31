import jieba.posseg as pseg


ErrorWord = ['报错', '故障', '无效', '异常', '错误', '出错', '无法', '问题', '提示', '解决']  # 可以认定为报错内容的关键词。这个列表中的词汇有优先级，排在前面的词优先考虑
NegativeWords = ['not', 'no', 'null', 'error', 'warning', 'errno', 'except', 'traceback', 'fail', 'failed', '失败', '没有', '不到', '坏', '乱码']
SpecialVerbs = ['安装', '找', '不到']  # 认定当出现动词之后，报错信息就结束了。但如果出现这几种特殊的动词，则可以认定报错信息还没有结束


def get_err_msg_1line(text):
    words = list(pseg.cut(text))
    for err_word in ErrorWord:
        # 1.检查标题中是否有”报错“关键词。
        baocuo_dx = -1
        for word_it in range(len(words)):
            if words[word_it].word == err_word:
                baocuo_dx = word_it
                break
        # 2.如果有这个关键词，则检查这个词到下一个动词之间的词（去掉“报错”一词紧跟着的冒号、破折号、括号、空格和助词、动词）。如果包含否定形式的内容，则认为是报错信息
        # todo: 这么做可能会在"没有"一词上出现误判
        if baocuo_dx >= 0:
            err_msg = []
            for word_it in range(baocuo_dx + 1, len(words)):
                if len(err_msg) == 0 and words[word_it].word in [':', '：', '-', '(', '（', '[', '【', '，']:
                    continue
                elif len(err_msg) == 0 and words[word_it].flag[0] in ['v', 'u', 'm'] and words[word_it].word not in SpecialVerbs:
                    continue
                elif words[word_it].flag[0] == 'v' and words[word_it].word not in SpecialVerbs:
                    break
                else:
                    err_msg.append(words[word_it].word)
            if len(err_msg) >= 1 and err_msg[-1] == '\n':
                err_msg = err_msg[:-1]
            is_negative_phase = False
            for word in err_msg:
                if word.lower().startswith('un'):
                    is_negative_phase = True
                    break
                for word2 in NegativeWords:
                    if word2 in word.lower():
                        is_negative_phase = True
                        break
                if is_negative_phase:
                    break
            if is_negative_phase:
                err_msg_output = ""
                for word in err_msg:
                    err_msg_output += word
                return err_msg_output
        # 3.如果有"报错"这个关键词，且这个关键词后面的内容不能构成一条准确的报错信息。则检查这个词到前一个动词之间的词（去掉“报错”一词紧跟着的冒号、破折号、括号、空格和助词、动词）。如果包含否定形式的内容，则认为是报错信息
        # todo: 可能会因为句子中间出现动词，如“打印”等，或“XXError 安装时报错”而出现误判
        if baocuo_dx >= 1:
            start_dx = 0
            end_dx = baocuo_dx - 1
            for word_it in range(baocuo_dx - 1, -1, -1):
                if words[word_it].word in [':', '：', '-', '(', '（', '[', '【']:
                    continue
                elif words[word_it].flag[0] in ['v', 'u', 'm'] and words[word_it].word not in SpecialVerbs:
                    continue
                else:
                    end_dx = word_it
                    break
            for word_it in range(end_dx):
                if words[word_it].flag[0] == 'v' and words[word_it].word not in SpecialVerbs:
                    if words[word_it + 1].flag[0] == 'u':
                        start_dx = word_it + 2
                    else:
                        start_dx = word_it + 1
            err_msg_output = ""
            is_negative_phase = False
            for word_it in range(start_dx, end_dx + 1):
                if words[word_it].word.lower().startswith('un'):
                    is_negative_phase = True
                for word2 in NegativeWords:
                    if word2 in words[word_it].word.lower():
                        is_negative_phase = True
                        break
                err_msg_output += words[word_it].word
            if is_negative_phase:
                return err_msg_output
    # 5.如果句子中没有“报错”等关键词，但有Error这一词汇，那么从Error一词起，后面的若干个词为报错内容，直到遇到中文动词为止。
    baocuo_dx = -1
    for word_it in range(len(words)):
        if 'error' in words[word_it].word.lower():
            baocuo_dx = word_it
            break
    if baocuo_dx >= 0:
        err_msg = []
        for word_it in range(baocuo_dx, len(words)):
            if words[word_it].flag[0] == 'v':
                break
            else:
                err_msg.append(words[word_it].word)
        if len(err_msg) >= 1 and err_msg[-1] == '\n':
            err_msg = err_msg[:-1]
        if len(err_msg) >= 1:
            err_msg_output = ""
            for word in err_msg:
                err_msg_output += word
            return err_msg_output
    return str()


def is_str_code(str_input):
    """判断一个字符串是否为代码段。判定方法为，字符串不包含中文，且长度需要达到20个字符"""
    for ch in str_input:
        if u'\u4e00' <= ch <= u'\u9fff':
            return False
    if len(str_input) >= 20:
        return True
    else:
        return False


def get_err_msg(title, text):
    # title_vocs = list(jieba.cut(title, cut_all=False))
    # 尝试从标题中获取报错信息
    err_msg = get_err_msg_1line(title)
    if err_msg:
        return err_msg
    # 尝试从正文中获取报错信息。原则上只获取前3段的内容
    paragraphs = text.replace('\n\n\n', '\n\n').split('\n\n')
    for para_it in range(min(5, len(paragraphs))):
        err_msg = get_err_msg_1line(paragraphs[para_it])
        # 如果报错信息为多行代码段，则只保留最后一行的内容
        if '\n' in err_msg and is_str_code(err_msg):
            err_msg = err_msg[err_msg.rfind('\n') + 1:]
        if err_msg:
            return err_msg
    return str()

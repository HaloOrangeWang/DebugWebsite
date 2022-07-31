from .funcs import has_chn_chrs, all_chn_chrs, calc_eng_chr_ratio, get_split_str


ErrorWords = ['报错', '错误', '出错']
SolveWords = ["解决方法", "解决办法", "处理方法", "处理办法", "总结", "正确的", "解决方案"]
SolveInParaWords = ["解决", "处理方法", "处理办法", "总结", "正确的", "改"]
SolveImpliedWords = ["即可", "好了"]
CheckWords = ["校验", "检查", "验证"]
PostscriptWords = ["参考", "地址", "转自", "摘自", "附上", "引用", "原文"]
ExampleWords = ["比如", "例如", "我"]
AttemptKeywords = ['学', '搜', '尝试', '发现', '看', '弄', "了解", "理解", "掌握", "盯", "想"]
ReasonWords = ["因为", "原因", "所以", "但是"]


def judge_solve_begin(sentences):
    """判断本段是否包含解决方法开始的信号"""
    # 1.判断这行文字的开头是否包含“解决方法”等明确的解决信号，且长度较短
    if len(sentences[0]) <= 12:
        for t in SolveWords:
            if t in sentences[0]:
                return True
    # 2.判断这行文字中是否包含“通过.....解决”的标识
    word1_dx = -1
    word2_dx = -1
    for word in ["通过", "使用"]:
        if word in sentences[0]:
            word1_dx = sentences[0].index(word)
            break
    if "解决" in sentences[0]:
        word2_dx = sentences[0].index("解决")
    if word1_dx != -1 and word2_dx != -1 and word1_dx < word2_dx:
        return True
    return False


def judge_new_err_msg(text):
    """判断这段文字是否为新的报错信息的描述"""
    has_err_word = False
    for word in ErrorWords:
        if word in text:
            has_err_word = True
            break
    word1_dx = -1
    word2_dx = -1
    for word in ["出现", "发生"]:
        if word in text:
            word1_dx = text.index(word)
            break
    if "问题" in text:
        word2_dx = text.index("问题")
    if word1_dx != -1 and word2_dx != -1 and word1_dx < word2_dx:
        has_err_word = True
    if has_err_word is True:
        has_neg_chn_word = False
        for word in ["不", "消失"]:
            if word in text:
                has_neg_chn_word = True
                break
        if not has_neg_chn_word:
            return True
    return False


def judge_postscript(sentences, is_in_line):
    """
    判断一段话是否为后记
    :param sentences: 这段话的句子列表
    :param is_in_line: 是否在行内搜索
    """
    if is_in_line:
        return False
    if len(sentences[0]) <= 10 or 0 <= sentences[0].find('http') <= 10:
        for word in PostscriptWords:
            if word in sentences[0]:
                return True
    return False


def judge_subtitle(sentences, is_in_line):
    """
    判断一段话是否为小标题
    :param sentences: 这段话的句子列表
    :param is_in_line: 是否在行内搜索
    """
    if is_in_line:
        return False
    if not sentences[0]:
        return False
    if len(sentences[0]) <= 10:
        if sentences[0][0] in ["一", "二", "三", "四", "五", "六", "七", "八", "九", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "a", "b", "c"]:
            if len(sentences[0]) <= 1 or sentences[0][1] in ["、", "）", ")"]:
                return True
        if sentences[0][0] in ["(", "（"]:
            if len(sentences[0]) >= 3 and sentences[0][1] in ["一", "二", "三", "四", "五", "六", "七", "八", "九", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "a", "b", "c"] and sentences[0][2] in ["）", ")"]:
                return True
    return False


def get_solve_1line(text, lp_has_solve_sign, is_in_line):
    """
    根据“解决方法”之类的关键词，从一个段落中尝试获取解决问题的处理信息
    :param text: 这一段落的文字
    :param lp_has_solve_sign: 上一个段落是否包含解决问题的信息
    :param is_in_line: 是否在行内搜索
    :return: 这一行内容是否为解决问题的信息，以及是否有没处理完的解决信息信号
    """
    # 如果这一行内容为空字符串的话，则它不能作为解决问题的信息
    if len(text) == 0:
        return False, lp_has_solve_sign, False

    sentences = get_split_str(text, ['，', ',', '。', '；', ';', '：', ':', '\n'])
    if len(sentences) == 0:
        return False, lp_has_solve_sign, False

    solve_sign = lp_has_solve_sign
    is_solve_para = lp_has_solve_sign
    new_solve_sign = judge_solve_begin(sentences)  # 这个变量标记本行是否会有新的解决方法信息
    # 1.如果上一个段落推导下来为“没有解决问题的信息”，则需要判断本段后没有解决问题的信息起始的信号
    if not solve_sign and new_solve_sign:
        solve_sign = True
    if new_solve_sign:
        is_solve_para = bool(len(text) > 12)
    if is_solve_para:
        # 2.判断本段内容是否为后记。如果是的话，则不应作为解决问题的信息
        if judge_postscript(sentences, is_in_line):
            return False, False, False
        # 3.判断本段内容是否为补充错误的描述
        if judge_new_err_msg(text):
            return False, bool(new_solve_sign), bool(new_solve_sign)
        # 4.判断本段内容是否为小标题
        if judge_subtitle(sentences, is_in_line):
            return False, True, bool(new_solve_sign)
        # 5.判断本段内容是否为解决问题后的检查方法
        for word in CheckWords:
            if word in text:
                return False, False, False
        # 6.判断本段内容是否为解决方法的举例
        for word in ExampleWords:
            if word in text:
                return False, False, False
        # 7.判断本段内容是否为主观上解决问题过程的描述
        for word in AttemptKeywords:
            if word in text:
                return False, False, False
        # 8.判断本段内容是否为对问题原因的描述
        for word in ReasonWords:
            if word in text:
                return False, True, bool(new_solve_sign)
    return is_solve_para, solve_sign, new_solve_sign


def solve_msg_postprocess(solve_text, solve_lines):
    """
    对解决问题的描述进行后处理。去除过长的代码段，以及“这下没有问题了”之类的无效语句。另外，如果某一段解决方案包含多行的话，把中间的行也收入进来
    :param solve_text: 处理之前的解决问题描述
    :param solve_lines: 处理之前的解决问题信息所在行
    :return: 处理之后的解决问题的描述及其所在行
    """
    # 1.去除结尾处“这下没有问题了”之类的无效语句
    reduce_line_num = 0
    solve_text_2 = solve_text.split('\n\n')
    if len(solve_text_2) >= 2:
        for it in range(len(solve_text_2) - 1, 0, -1):
            if len(solve_text_2[it]) < 10 and all_chn_chrs(solve_text_2[it]):
                reduce_line_num += 1
            else:
                break
        if reduce_line_num >= 1:
            solve_lines = solve_lines[:-reduce_line_num]
            solve_text = '\n\n'.join(solve_text_2[:-reduce_line_num])
    # 2.检查是否有连续5行以上的代码段。如果有的话，则只保留第一行
    solve_text_2 = solve_text.replace('\n\n', '\n').split('\n')
    code_line_cnt = 0
    has_many_codes = False
    for line in solve_text_2:
        if (not has_chn_chrs(line)) or (len(line) >= 10 and calc_eng_chr_ratio(line) >= 0.8):
            code_line_cnt += 1
        else:
            code_line_cnt = 0
        if code_line_cnt >= 5:
            has_many_codes = True
            break
    if has_many_codes:
        solve_text = solve_text_2[0]
        solve_lines = [solve_lines[0]]
    # 3.检查是否某一个段落中包含了多行
    solve_text_2 = solve_text.split('\n\n')
    for t in range(len(solve_text_2)):
        add_line_num = solve_text_2[t].count('\n') - 1 if solve_text_2[t].startswith('\n') else solve_text_2[t].count('\n')
        solve_lines.extend([solve_lines[t] - t0 - 1 for t0 in range(add_line_num)])
    return solve_text, solve_lines


def get_solve_basic(paragraph_list, large_para_dx_list, line_dx_list):
    """
    根据显式给出的“解决方法”“处理方法”等词汇，来找出解决方法的信息
    :param paragraph_list: 段落列表
    :param large_para_dx_list: 哪几个段落是大段落的开始
    :param line_dx_list: 每个段落分别对应原文件的行标
    :return: 哪些文字、以及那几行对应的是解决问题的信息
    """
    solve_lines = []
    solve_text_list = []
    solve_sign = False
    solve_text_2 = str()
    solve_lines_2 = []
    for para_it in range(len(paragraph_list)):
        # 如果到了大段落的分隔处，则解决方法的信息应当重新计算。
        if para_it in large_para_dx_list:
            if solve_lines_2:
                solve_text_2, solve_lines_2 = solve_msg_postprocess(solve_text_2, solve_lines_2)
                solve_text_list.append(solve_text_2)
                solve_lines.extend(solve_lines_2)
                solve_text_2 = str()
                solve_lines_2 = []
                solve_sign = False
        # 检查这一行是否为解决方法的描述。如果是的话，将其插入到解决方法的汇总信息中
        is_solve_msg, solve_sign, new_solve_sign = get_solve_1line(paragraph_list[para_it], solve_sign, False)
        if is_solve_msg and (not new_solve_sign):
            if len(solve_lines_2) == 0:
                solve_text_2 += paragraph_list[para_it]
            else:
                solve_text_2 += ('\n\n' + paragraph_list[para_it])
            solve_lines_2.append(line_dx_list[para_it])
        # 到达文件末尾，或解决方法的信息段结束了，或遇到新的解决问题的信息了的处理
        if para_it == len(paragraph_list) - 1 or solve_sign is False:
            if solve_lines_2:
                solve_text_2, solve_lines_2 = solve_msg_postprocess(solve_text_2, solve_lines_2)
                solve_text_list.append(solve_text_2)
                solve_lines.extend(solve_lines_2)
                solve_text_2 = str()
                solve_lines_2 = []
        if new_solve_sign:
            if solve_lines_2:
                solve_text_2, solve_lines_2 = solve_msg_postprocess(solve_text_2, solve_lines_2)
                solve_text_list.append(solve_text_2)
                solve_lines.extend(solve_lines_2)
                if is_solve_msg:
                    solve_text_2 = paragraph_list[para_it]
                    solve_lines_2 = [line_dx_list[para_it]]
                else:
                    solve_text_2 = str()
                    solve_lines_2 = []
            else:
                if is_solve_msg:
                    solve_text_2 = paragraph_list[para_it]
                    solve_lines_2 = [line_dx_list[para_it]]
    return solve_text_list, solve_lines


def get_solve_in_line(paragraph_list, large_para_dx_list, line_dx_list):
    """
    根据段落内的“解决”词汇，来找出解决方法的信息
    :param paragraph_list: 段落列表
    :param large_para_dx_list: 哪几个段落是大段落的开始
    :param line_dx_list: 每个段落分别对应原文件的行标
    :return: 哪些文字、以及那几行对应的是解决问题的信息
    """
    total_solve_text = list()
    total_solve_lines = list()
    for para_it in range(len(paragraph_list) - 1, -1, -1):
        # 1.从文字中尝试找出“解决”等关键词
        has_solve_keyword = False
        for word in SolveInParaWords:
            if word in paragraph_list[para_it]:
                has_solve_keyword = True
                break
        # 2.尝试从后续的段落中找到解决方法的信息
        if has_solve_keyword:
            find_solve_msg = False
            solve_text = str()
            solve_lines = []
            if not total_solve_lines:
                for para_it2 in range(para_it + 1, len(paragraph_list)):
                    # 如果到了大段落的分隔处，则解决方法的信息就此结束。
                    if para_it2 in large_para_dx_list:
                        if solve_lines:
                            solve_text, solve_lines = solve_msg_postprocess(solve_text, solve_lines)
                            total_solve_text.append(solve_text)
                            total_solve_lines.extend(solve_lines)
                            find_solve_msg = True
                            break
                    # 检查这一行是否为解决方法的描述。如果是的话，将其插入到解决方法的汇总信息中
                    is_solve_msg, solve_sign, __ = get_solve_1line(paragraph_list[para_it2], True, False)
                    if is_solve_msg:
                        if len(solve_lines) == 0:
                            solve_text += paragraph_list[para_it2]
                        else:
                            solve_text += ('\n\n' + paragraph_list[para_it2])
                        solve_lines.append(line_dx_list[para_it2])
                    # 到达文件末尾，或解决方法的信息段结束了的处理。如果已经找到了解决问题的信息，则直接返回。如果没找到，则转入下一个寻找策略。
                    if para_it2 == len(paragraph_list) - 1 or solve_sign is False:
                        if solve_lines:
                            solve_text, solve_lines = solve_msg_postprocess(solve_text, solve_lines)
                            total_solve_text.append(solve_text)
                            total_solve_lines.extend(solve_lines)
                            find_solve_msg = True
                            break
                        else:
                            break
                if find_solve_msg:
                    continue
        # 3.尝试从本段内找到解决方法的信息
        if para_it >= len(paragraph_list) - 5 or (not total_solve_lines):
            sentences = get_split_str(paragraph_list[para_it], ['，', ',', '。', '；', ';', '：', ':', '\n'])
            for sentence_it in range(len(sentences) - 1, -1, -1):
                find_solve_msg = False
                has_solve_keyword_2 = False
                keyword_dx = 0
                for word in SolveInParaWords:
                    if word in sentences[sentence_it]:
                        has_solve_keyword_2 = True
                        keyword_dx = sentences[sentence_it].index(word)
                        break
                if has_solve_keyword_2:
                    solve_text = str()
                    # 看这个词后面的句子
                    for sentence_it2 in range(sentence_it + 1, len(sentences)):
                        # 检查这一句是否为解决方法的描述。如果是的话，将其插入到解决方法的汇总信息中
                        is_solve_msg, solve_sign, __ = get_solve_1line(sentences[sentence_it2], True, True)
                        if is_solve_msg:
                            if len(solve_text) == 0:
                                solve_text += sentences[sentence_it2]
                            else:
                                # 这里做了个简化处理，把标点符号全部改成了逗号
                                solve_text += ('，' + sentences[sentence_it2])
                        # 到达这一行文字的末尾，或解决方法的信息段结束了的处理。如果已经找到了解决问题的信息，则直接返回。如果没找到，则转入下一个寻找策略。
                        if sentence_it2 == len(sentences) - 1 or solve_sign is False:
                            if solve_text:
                                total_solve_text.append(solve_text)
                                total_solve_lines.append(line_dx_list[para_it])
                                find_solve_msg = True
                                break
                            else:
                                break
                    if find_solve_msg:
                        break
                    # 看这个词前面的句子
                    if sentence_it >= 1 or keyword_dx >= 10 and '没有' not in sentences[sentence_it]:
                        for sentence_it2 in range(sentence_it, -1, -1):
                            # 检查这一句是否为解决方法的描述。如果是的话，将其插入到解决方法的汇总信息中
                            is_solve_msg, solve_sign, __ = get_solve_1line(sentences[sentence_it2], True, True)
                            if is_solve_msg:
                                if len(solve_text) == 0:
                                    solve_text += sentences[sentence_it2]
                                else:
                                    # 这里做了个简化处理，把标点符号全部改成了逗号
                                    solve_text = sentences[sentence_it2] + '，' + solve_text
                            # 到达这一行文字的开头，或解决方法的信息段结束了的处理。如果已经找到了解决问题的信息，则直接返回。如果没找到，则转入下一个寻找策略。
                            if sentence_it2 == 0 or solve_sign is False:
                                if solve_text:
                                    total_solve_text.append(solve_text)
                                    total_solve_lines.append(line_dx_list[para_it])
                                    find_solve_msg = True
                                    break
                                else:
                                    break
                        if find_solve_msg:
                            break
    return total_solve_text, total_solve_lines


def get_solve_in_implied(text_1line):
    """
    根据“即可、好了”等词汇，找到隐含的解决问题信息
    :param text_1line: 一行文字内容
    :return: 哪些文字对应的是解决问题的信息
    """
    sentences = get_split_str(text_1line, ['，', ',', '。', '；', ';', '：', ':', '\n'])
    for sentence_it in range(len(sentences) - 1, -1, -1):
        has_solve_keyword_2 = False
        keyword_dx = 0
        for word in SolveImpliedWords:
            if word in sentences[sentence_it]:
                has_solve_keyword_2 = True
                keyword_dx = sentences[sentence_it].index(word)
                break
        if has_solve_keyword_2:
            solve_text = str()
            # 看这个词后面的句子
            for sentence_it2 in range(sentence_it + 1, len(sentences)):
                # 检查这一句是否为解决方法的描述。如果是的话，将其插入到解决方法的汇总信息中
                is_solve_msg, solve_sign, __ = get_solve_1line(sentences[sentence_it2], True, True)
                if is_solve_msg:
                    if len(solve_text) == 0:
                        solve_text += sentences[sentence_it2]
                    else:
                        # 这里做了个简化处理，把标点符号全部改成了逗号
                        solve_text += ('，' + sentences[sentence_it2])
                # 到达这一行文字的末尾，或解决方法的信息段结束了的处理。如果已经找到了解决问题的信息，则直接返回。如果没找到，则转入下一个寻找策略。
                if sentence_it2 == len(sentences) - 1 or solve_sign is False:
                    if solve_text:
                        return solve_text
                    else:
                        break
            # 看这个词前面的句子
            if sentence_it >= 1 or keyword_dx >= 10:
                for sentence_it2 in range(sentence_it, -1, -1):
                    # 检查这一句是否为解决方法的描述。如果是的话，将其插入到解决方法的汇总信息中
                    is_solve_msg, solve_sign, __ = get_solve_1line(sentences[sentence_it2], True, True)
                    if is_solve_msg:
                        if len(solve_text) == 0:
                            solve_text += sentences[sentence_it2]
                        else:
                            # 这里做了个简化处理，把标点符号全部改成了逗号
                            solve_text = sentences[sentence_it2] + '，' + solve_text
                    # 到达这一行文字的开头，或解决方法的信息段结束了的处理。如果已经找到了解决问题的信息，则直接返回。如果没找到，则转入下一个寻找策略。
                    if sentence_it2 == 0 or solve_sign is False:
                        if solve_text:
                            return solve_text
                        else:
                            break
    return str()


def get_solve_msg(text):
    solve_lines = []
    solve_text_list = []
    # 1.将这段文字拆分成若干个段落。大段落的分隔处要标识出来
    paragraph_list = []
    large_para_dx_list = []
    line_dx_list = []
    paragraphs_large = text.split('\n\n\n')
    for t in range(len(paragraphs_large)):
        paragraphs = paragraphs_large[t].split('\n\n')
        paragraph_list.extend(paragraphs)
        large_para_dx_list.append((0 if len(large_para_dx_list) == 0 else large_para_dx_list[-1]) + len(paragraphs))
        # if paragraphs_large[t].startswith('\n'):
        #     last_line = (2 if t == 0 else (line_dx_list[-1] + 4))
        # else:
        #     last_line = (1 if t == 0 else (line_dx_list[-1] + 3))
        last_line = (1 + paragraphs[0].count('\n') if t == 0 else (line_dx_list[-1] + 3))
        for t0 in range(len(paragraphs)):
            if t0 != 0:
                last_line += (2 + paragraphs[t0].count('\n'))
            line_dx_list.append(last_line)
    # 2.根据显式给出的“解决方法”“处理方法”等词汇，来找出解决方法的信息
    solve_text_list_2, solve_lines_2 = get_solve_basic(paragraph_list, large_para_dx_list, line_dx_list)
    solve_text_list.extend(solve_text_list_2)
    solve_lines.extend(solve_lines_2)
    # 3.如果根据上述方法找不到解决方法的信息的话，那么尝试找段落内的解决信息。
    if len(solve_lines) == 0:
        solve_text_list_2, solve_lines_2 = get_solve_in_line(paragraph_list, large_para_dx_list, line_dx_list)
        solve_text_list.extend(solve_text_list_2)
        solve_lines.extend(solve_lines_2)
    # 4.如果仍然不能找到解决方法的信息的话，那么尝试找出隐含的解决信息
    if len(solve_lines) == 0:
        for para_it in range(len(paragraph_list) - 1, -1, -1):
            solve_text = get_solve_in_implied(paragraph_list[para_it])
            if solve_text:
                solve_text_list.append(solve_text)
                solve_lines.append(line_dx_list[para_it])
                break
    return solve_lines, solve_text_list

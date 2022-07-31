from .funcs import judge_empty_content, judge_negative_phase, has_chn_chrs, get_split_str
import jieba.posseg as pseg


# 如果出现ErrorWord中的词汇，且这个词汇前面没有表示否定的词汇，则可以认为这句话为场景信息
# 这里的ErrorWord，和get_err_msg中的ErrorWord中的内容不完全相同。
ErrorWord = ['报错', '出错', '提示', '解决', '遇到', '发生', '出现']
NegativeWords = ['not', 'no', 'null', 'error', 'warning', 'errno', 'except', 'traceback', 'fail', 'failed', '失败', '没有', '不到', '坏', '乱码', '问题', '错误']
OperateKeywords = ['安装', '设置', '配置', '环境', '运行', '版本', '使用', '引入', '模块', '用', '版', '系统', '搭建', '命令', '任务']
AssumeKeywords = ['假如', '假若', '如果', '若是']
AttemptKeywords = ['学', '学习', '搜', '搜索', '检查', '尝试', '发现', '看', '看到', '想', '想着', '感觉', '弄']


def handle_pre_scene_msg(pre_scene_msg):
    """
    去除初步场景信息首尾处的空格、代词、介词、连词、时间词、助词。
    并判断，如果初步的错误信息中只有空格、python、代词、介词、连词、时间词、助词，则不能判定为场景信息。
    :return: 是否为合理的错误信息, 错误信息字符串
    """
    msg_start_dx = -1
    msg_end_dx = -1
    valid_scene_msg = False  # 这个标志位表明是否包含真正的场景信息。
    for t in range(len(pre_scene_msg)):
        if (not judge_empty_content(pre_scene_msg[t].word)) and (pre_scene_msg[t].flag[0] not in ['u', 'c', 'p', 'r', 'f', 't']):
            msg_end_dx = t
            if msg_start_dx == -1:
                msg_start_dx = t
            if pre_scene_msg[t].word not in OperateKeywords and pre_scene_msg[t].word.lower() not in ['python', 'python2', 'python3']:
                valid_scene_msg = True
        if pre_scene_msg[t].word in AttemptKeywords:
            return False, str()
    if valid_scene_msg:
        scene_msg = ''.join([pre_scene_msg[t].word for t in range(msg_start_dx, msg_end_dx + 1)])
        return True, scene_msg
    else:
        return False, str()


def scene_msg_can_in_next_sentence(pre_scene_msg):
    """判断场景信息可不可能在下一句中。判断依据是，如果这一句话中只有动词、标点符号和OperateKeyword，那么场景信息应当在下一句中"""
    for t in range(len(pre_scene_msg)):
        if (pre_scene_msg[t].flag[0] not in ['v', 'x']) and (pre_scene_msg[t].word not in OperateKeywords):
            return False
    return True


def calc_eng_word_ratio(pre_scene_msg):
    """判断一句话中英文和数字的占比"""
    eng_num_cnt = 0
    chn_cnt = 0
    for t in range(len(pre_scene_msg)):
        if pre_scene_msg[t].flag in ['m', 'eng']:
            eng_num_cnt += 1
        elif pre_scene_msg[t].flag not in ['x']:
            chn_cnt += 1
    if eng_num_cnt + chn_cnt == 0:
        return 0
    return eng_num_cnt / (eng_num_cnt + chn_cnt)


def get_scene_1line(sentences, is_title, ls_has_ope_word_v, is_already_scene_msg, err_msg_sentences):
    """
    从一段话中解析出场景信息
    :param sentences: 将正文拆分成句子之后的文字内容
    :param is_title: 是不是标题
    :param ls_has_ope_word_v: 上一句话是否包含某些关键词，从而认定本句有场景信息
    :param is_already_scene_msg: 在处理这段话之前，是否已经有场景信息了。
    :param err_msg_sentences: 本句话中的错误信息
    :return: 场景信息，以及场景信息是否是通过ErrorWord解析出来的
    """

    def is_err_msg_in_scene(scene_msg2):
        for sentence2 in err_msg_sentences:
            if sentence2 in scene_msg2:
                return True
        return False

    scene_msg_output_list = list()
    # 0.判断这个段落是不是一个假设段。如果是的话，那么它就不能作为场景描述。
    for t in AssumeKeywords:
        if sentences[0].startswith(t):
            return scene_msg_output_list, False, ls_has_ope_word_v
    # 1.先将这段话进行拆分，拆分标志为逗号、冒号、分号、句号
    is_scene_from_ew = False
    ls_has_ope_word = False  # 上一句话中是否包含场景描述相关的信息，且被判定为场景描述句
    for sentence_it in range(len(sentences)):
        words = list(pseg.cut(sentences[sentence_it]))
        # 2.根据报错的关键词找到场景信息
        if (not is_already_scene_msg) and (not is_scene_from_ew):
            pre_scene_msg = None
            for word_it in range(len(words) - 1, -1, -1):
                if words[word_it].word in ErrorWord:
                    # 找到错误信息之后，以上一个错误信息-这个错误信息之间的内容，作为初定的场景信息
                    start_dx = 0
                    for t in range(word_it - 1, -1, -1):
                        if words[t].word in ErrorWord:
                            start_dx = t + 1
                            break
                    pre_scene_msg = words[start_dx: word_it]
                    # 判断这段初定的场景信息是否包含否定信息，且是否有意义。如果内容有意义，且不包含否定信息的话，则认定为场景信息
                    if pre_scene_msg is not None:
                        is_scene_msg, scene_msg = handle_pre_scene_msg(pre_scene_msg)
                        if (not judge_negative_phase(pre_scene_msg, NegativeWords)) and is_scene_msg and (not is_err_msg_in_scene(scene_msg)):
                            scene_msg_output_list.append(scene_msg)
                            ls_has_ope_word = False
                            ls_has_ope_word_v = False
                            is_scene_from_ew = True
                            break
            if is_scene_from_ew:
                continue
        # 4.如果上一句因OperateKeyword被判断为场景信息，且本句中英文和数字占比超过25%，则也可以认定为场景描述
        if ls_has_ope_word:
            ls_has_ope_word = False
            ls_has_ope_word_v = False
            end_dx = len(words)
            for t in range(len(words)):
                if words[t].word in ErrorWord:
                    end_dx = t
                    break
            pre_scene_msg = words[0: end_dx]
            eng_ratio = calc_eng_word_ratio(pre_scene_msg)
            if eng_ratio >= 0.25:
                is_scene_msg, scene_msg = handle_pre_scene_msg(pre_scene_msg)
                if (not judge_negative_phase(pre_scene_msg, NegativeWords)) and is_scene_msg and (not is_err_msg_in_scene(scene_msg)):
                    scene_msg_output_list.append(scene_msg)
                    continue
        # 5.如果上一句有OperateKeyword，但未被判断为场景信息。且下一句为全英文内容，则下一句也可以被认定为场景信息
        if ls_has_ope_word_v:
            if (not has_chn_chrs(sentences[sentence_it])) and (not judge_negative_phase(words, NegativeWords)) and (not is_err_msg_in_scene(sentences[sentence_it])):
                scene_msg_output_list.append(sentences[sentence_it])
                continue
            else:
                ls_has_ope_word = False
                ls_has_ope_word_v = False
        # 3.根据关键词判断是否为场景信息
        has_operate_word = False
        for word_it in range(len(words)):
            if words[word_it].word in OperateKeywords:
                has_operate_word = True
                break
        if has_operate_word:
            end_dx = len(words)
            for t in range(len(words)):
                if words[t].word in ErrorWord:
                    end_dx = t
                    break
            pre_scene_msg = words[0: end_dx]
            is_scene_msg, scene_msg = handle_pre_scene_msg(pre_scene_msg)
            is_scene_msg_2 = True
            if is_scene_msg and (not is_err_msg_in_scene(scene_msg)):
                # 如果已经找到一部分场景信息了，那么再次查找场景信息时，要求英文占比超过25%。
                if is_already_scene_msg:
                    eng_ratio = calc_eng_word_ratio(pre_scene_msg)
                    is_scene_msg_2 = bool(eng_ratio >= 0.25)
                else:
                    is_scene_msg_2 = True
                if is_scene_msg_2:
                    scene_msg_output_list.append(scene_msg)
                    ls_has_ope_word = True
                    continue
            if (not (is_scene_msg and is_scene_msg_2)) and scene_msg_can_in_next_sentence(pre_scene_msg):
                ls_has_ope_word_v = True
    return scene_msg_output_list, is_scene_from_ew, ls_has_ope_word_v


def get_scene_msg(title, text, correct_err_msgs):
    # title_vocs = list(jieba.cut(title, cut_all=False))
    scene_msg_list_all = list()
    ls_has_ope_word_v = False
    # 拆分错误信息
    err_msg_sentences = get_split_str(correct_err_msgs, ['，', '。', '；', '：', '\n'])
    # 尝试从标题中获取场景信息
    sentences_title = get_split_str(title, ['，', ',', '。', '；', ';', '：', ':', '\n'])
    if len(sentences_title) != 0:
        scene_msg_list, is_scene_from_ew, ls_has_ope_word_v = get_scene_1line(sentences_title, True, ls_has_ope_word_v, False, err_msg_sentences)
        scene_msg_list_all.extend(scene_msg_list)
    # 尝试从正文中获取场景信息。原则上只获取前5段的内容
    paragraphs = text.replace('\n\n\n', '\n\n').split('\n\n')
    for para_it in range(min(5, len(paragraphs))):
        # 如果某一段落中出现“解决方法”，或上一段落中出现“成功解决”字样，则认为后面的内容是解决问题的方式了
        sentences = get_split_str(paragraphs[para_it], ['，', ',', '。', '；', ';', '：', ':', '\n'])
        if len(sentences) == 0:
            continue
        if sentences[0].startswith('解决方法') or sentences[0].startswith('解决办法') or (para_it != 0 and '成功解决' in paragraphs[para_it - 1]):
            break
        scene_msg_list, is_scene_from_ew, ls_has_ope_word_v = get_scene_1line(sentences, False, ls_has_ope_word_v, bool(scene_msg_list_all), err_msg_sentences)
        scene_msg_list_all.extend(scene_msg_list)
    return scene_msg_list_all

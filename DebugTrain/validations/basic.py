from train_simple.funcs import get_split_str
from train_simple.scene import handle_pre_scene_msg
import jieba.posseg as pseg


def trim_n(s):
    """将字符串开头、末尾处的空格去掉。将连续多个的空格转变为1个"""
    # 现将特殊空格符都转换为' '
    s = s.replace('\t', ' ').replace('\r', '').replace(chr(0xa0), ' ')
    if len(s) == 0:
        return s
    # 将开头、结尾处的空格、中文逗号、中文分号qudiao
    output_str_list = list()
    is_start = False
    for t in range(len(s)):
        if s[t] not in [' ']:
            is_start = True
            output_str_list.append(s[t])
        else:
            if is_start and (s[t - 1] != ' '):
                output_str_list.append(s[t])
    if output_str_list[-1] != ' ':
        output_str = ''.join(output_str_list)
    else:
        output_str = ''.join(output_str_list[:-1])
    # 如果开头、结尾处有中文的逗号、括号的话，将其去掉
    if output_str[0] in ['，', '。', '【', '】', '：', '；']:
        output_str = output_str[1:]
    if output_str[-1] in ['，', '。', '【', '】', '：', '；']:
        output_str = output_str[:-1]
    # 如果开头、结尾都是引号（含中文引号）的话，将其去掉
    if output_str[0] in ['\'', '"', '“', '”'] and output_str[-1] in ['\'', '"', '“', '”']:
        output_str = output_str[1: -1]
    return output_str


def valid_err_msg(all_err_msgs, all_mark_data):
    """
    对比通过算法推算的错误信息，与通过标注确认的错误信息。如果内容一致（允许有python/python2/python3这几个关键词的不一致）则记1分，漏判记0分，内容错误记-1分。
    :param all_err_msgs: 通过算法推算的错误信息
    :param all_mark_data: 通过标注确认的错误、场景、解决方案的信息
    :return: 分数
    """

    def clean_str_func(s):
        return s.lower().replace('python2', '').replace('python3', '').replace('python', '')

    score = 0
    for aid in all_err_msgs:
        correct_err_msg = trim_n(clean_str_func(all_mark_data[aid].err_msg))
        origin_err_msg = trim_n(clean_str_func(all_err_msgs[aid]))
        if origin_err_msg == correct_err_msg:
            score += 1
        elif correct_err_msg != str() and origin_err_msg == str():
            pass
        else:
            score -= 1
    return score


def valid_scene(all_scene, all_mark_data):
    """
    对比通过算法推算的场景信息，与通过标注确认的场景信息。
    :param all_scene: 通过算法推算的错误信息
    :param all_mark_data: 通过标注确认的错误、场景、解决方案的信息
    :return: 分数、总分
    """
    def clean_str_func(s):
        return s.lower().replace('python2', '').replace('python3', '').replace('python', '').replace('的时候', '').replace('时候', '').replace('时', '')

    score = 0  # 程序得分
    total_score = 0  # 总分
    for aid in all_scene:
        if all_mark_data[aid].err_msg != str():
            total_score += 1
            # 先对标注后的场景信息做一些简单处理。按照句子进行拆分
            correct_scene_list = list()
            half_correct_scene_list = list()  # 后面这个变量表示“半解决方案”。对于“半解决方案”，程序生成的结果无论如何都不算错
            for t in range(len(all_mark_data[aid].scenes)):
                sentences = get_split_str(all_mark_data[aid].scenes[t], ['，', ',', '。', '；', ';', '：', ':', '\n'])
                if len(sentences) == 1:
                    if all_mark_data[aid].scene_weight[t] == 1:
                        correct_scene_list.append(sentences[0])
                    elif all_mark_data[aid].scene_weight[t] == 0.5:
                        half_correct_scene_list.append(sentences[0])
                else:
                    for t0 in range(len(sentences)):
                        words = list(pseg.cut(sentences[t0]))
                        is_scene_msg, scene_msg = handle_pre_scene_msg(words)
                        if is_scene_msg:
                            if all_mark_data[aid].scene_weight[t] == 1:
                                correct_scene_list.append(scene_msg)
                            elif all_mark_data[aid].scene_weight[t] == 0.5:
                                half_correct_scene_list.append(scene_msg)
            # 对比标注好的场景信息，与程序判定的场景信息
            in_mark_scene_dx_list = set()  # 对于程序判定的场景列表中，有哪些出现在了正确或半正确的场景判定列表中
            in_prog_scene_dx_list = set()  # 对于正确的场景列表，有哪些出现在了程序判定的场景列表中
            for t in range(len(all_scene[aid])):
                for t0 in range(len(correct_scene_list)):
                    # todo: 这里可能会因为标点符号、python关键词等原因而导致误判
                    correct_scene = trim_n(clean_str_func(correct_scene_list[t0]))
                    origin_scene = trim_n(clean_str_func(all_scene[aid][t]))
                    if correct_scene == origin_scene:
                        # score += 1 / len(correct_scene_list)
                        in_mark_scene_dx_list.add(t)
                        in_prog_scene_dx_list.add(t0)
                for t0 in range(len(half_correct_scene_list)):
                    correct_scene = trim_n(clean_str_func(half_correct_scene_list[t0]))
                    origin_scene = trim_n(clean_str_func(all_scene[aid][t]))
                    if correct_scene == origin_scene:
                        in_mark_scene_dx_list.add(t)
            # 计算得分。
            # 如果标注好的场景信息为空，那么程序判定的场景信息必须全部在正确信息或半正确信息中
            if len(correct_scene_list) == 0:
                if len(in_mark_scene_dx_list) == len(all_scene[aid]):
                    score += 1
                else:
                    score -= 1
            else:
                score += len(in_prog_scene_dx_list) / len(correct_scene_list)
                if len(all_scene[aid]) != 0:
                    score -= (len(all_scene[aid]) - len(in_mark_scene_dx_list)) / len(all_scene[aid])
            # print("%d, %.2f" % (aid, score))
    return score, total_score


def valid_solve(all_solve_lines, all_solve_msgs, all_mark_data):
    """
    对比通过算法推算的解决问题的信息，与通过标注确认的解决问题信息。
    :param all_solve_lines: 通过算法确认的解决问题的行标
    :param all_solve_msgs: 通过算法确认的解决问题的信息
    :param all_mark_data: 通过标注确认的错误、场景、解决方案的信息
    """

    def clean_str_func(s_list: list):
        paras2 = []
        for t in s_list:
            paras2.extend(t.replace('\n\n\n', '\n\n').split('\n\n'))
        for t in range(len(paras2)):
            for punc in ['，', ',', '。', '；', ';', '：', ':', '\n']:
                paras2[t] = paras2[t].replace(punc, '')
        return paras2

    line_score = 0  # 程序在行标判断中的得分
    msg_score = 0  # 程序在信息判断中得分
    total_score = 0  # 总分
    for aid in all_solve_lines:
        if all_mark_data[aid].err_msg != str():
            total_score += 1
            # 计算行标判断中的得分
            solve_lines = set(all_solve_lines[aid])
            c_solve_lines = set()
            half_c_solve_lines = set()
            for t0 in range(len(all_mark_data[aid].solve_lines)):
                if all_mark_data[aid].solve_weight[t0] == 1:
                    c_solve_lines.add(all_mark_data[aid].solve_lines[t0])
                elif all_mark_data[aid].solve_weight[t0] == 0.5:
                    half_c_solve_lines.add(all_mark_data[aid].solve_lines[t0])
            if len(c_solve_lines) == 0:
                if len(solve_lines - half_c_solve_lines) == 0:
                    line_score += 1
                else:
                    line_score -= 1
            else:
                line_score += len(solve_lines & c_solve_lines) / len(c_solve_lines)
                if len(solve_lines) != 0:
                    line_score -= len(solve_lines - c_solve_lines - half_c_solve_lines) / len(solve_lines)
            # 计算报错消息判断中的得分
            c_solve_msgs = []
            half_c_solve_msgs = []
            for t0 in range(len(all_mark_data[aid].solves)):
                if all_mark_data[aid].solve_weight[t0] == 1:
                    c_solve_msgs.extend(clean_str_func([all_mark_data[aid].solves[t0]]))
                elif all_mark_data[aid].solve_weight[t0] == 0.5:
                    half_c_solve_msgs.extend(clean_str_func([all_mark_data[aid].solves[t0]]))
            solve_msgs = clean_str_func(all_solve_msgs[aid])
            in_mark_solve_dx_list = set()  # 对于程序判定的解决方案列表中，有哪些出现在了正确或半正确的解决方案判定列表中
            in_prog_solve_dx_list = set()  # 对于正确的解决方案列表，有哪些出现在了程序判定的解决方案列表中
            for t0 in range(len(solve_msgs)):
                for t1 in range(len(c_solve_msgs)):
                    # todo: 这里可能会因为标点符号、python关键词等原因而导致误判
                    origin_solve_msg = trim_n(solve_msgs[t0])
                    correct_solve_msg = trim_n(c_solve_msgs[t1])
                    if correct_solve_msg == origin_solve_msg:
                        # score += 1 / len(correct_scene_list)
                        in_mark_solve_dx_list.add(t0)
                        in_prog_solve_dx_list.add(t1)
                for t1 in range(len(half_c_solve_msgs)):
                    origin_solve_msg = trim_n(solve_msgs[t0])
                    correct_solve_msg = trim_n(half_c_solve_msgs[t1])
                    if correct_solve_msg == origin_solve_msg:
                        in_mark_solve_dx_list.add(t0)
            # 计算得分。
            # 如果标注好的解决问题信息为空，那么程序判定的场景信息必须全部在正确信息或半正确信息中
            if len(c_solve_msgs) == 0:
                if len(in_mark_solve_dx_list) == len(solve_msgs):
                    msg_score += 1
                else:
                    msg_score -= 1
            else:
                msg_score += len(in_prog_solve_dx_list) / len(c_solve_msgs)
                if len(solve_msgs) != 0:
                    msg_score -= (len(solve_msgs) - len(in_mark_solve_dx_list)) / len(solve_msgs)
            # print("%d, %.2f" % (aid, score))
            # print("%d, %.2f " % (aid, line_score), solve_lines)
    return line_score, msg_score, total_score

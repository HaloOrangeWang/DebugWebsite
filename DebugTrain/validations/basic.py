from data_io.load_data import MarkData
from train_simple.funcs import get_split_str
import jieba.posseg as pseg
from typing import Dict, List
import re


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


def trim_n(s):
    """将字符串开头、末尾处的空格去掉。将连续多个的空格转变为1个"""
    # 现将特殊空格符都转换为' '
    s = s.replace('\t', ' ').replace('\r', '').replace(chr(0xa0), ' ')
    if len(s) == 0:
        return s
    # 将开头、结尾处的标点符号去掉，只保留中文，英文和空格（看一下是否需要将多个连续的空格变为单个）
    st_dx = 0
    ed_dx = 0
    for t in range(len(s)):
        if is_chn_eng_number(s[t]):
            st_dx = t
            break
    for t in range(len(s) - 1, -1, -1):
        if is_chn_eng_number(s[t]):
            ed_dx = t
            break
    s = s[st_dx: ed_dx + 1]
    if ed_dx <= st_dx:
        return str()
    # 将句子中间连续多个空格更改为1个
    assert s[0] != " " and s[-1] != " "
    output_str_list = list()
    for t in range(len(s)):
        # 注：经过了上一轮的判断之后，数组的第一项和最后一项一定不为空格
        if s[t] != " " or s[t - 1] != " ":
            output_str_list.append(s[t])
    output_str = ''.join(output_str_list)
    return output_str


def trim_number(s):
    """将字符串开头、末尾处的数字也去掉。如1. 一、 a)等。避免1.xxx前面的1.影响判断。判断依据是第一个字符为数字，而第二个字符既不是中文也不是英文"""
    if len(s) == 0:
        return str()
    if s[0] in ["一", "二", "三", "四", "五", "六", "七", "八", "九"] or is_eng_number(s[0]):
        if len(s) == 1:
            return str()
        if not is_chn_eng_number(s[1]):
            # 第一个字符为单个数字或汉字数字或英文，但第二个字符是空格或标点符号，说明大概率是“1.” “a)”之类的内容，将其去掉
            # 将开头、结尾处的标点符号去掉，只保留中文，英文和空格（看一下是否需要将多个连续的空格变为单个）
            st_dx = 1
            for t in range(1, len(s)):
                if is_chn_eng_number(s[t]):
                    st_dx = t
                    break
            s = s[st_dx:]
    return s


def remove_python_str(s):
    """将字符串中包含python及python版本号的信息去除。如python, python 2.7, python3.6.2等"""
    # 先将字符串小写
    s = s.lower()
    # 寻找python或python版本号的字符串，如python, python 2.7, python3.6.2等内容
    python_strs = re.findall(r"python[ ]?[0-9.]*", s)
    python_strs = sorted(python_strs, key=lambda x: len(x), reverse=True)
    for python_str in python_strs:
        s = s.replace(python_str, "")
    return s


class ErrMsgValidation:

    @staticmethod
    def whole_sentence(all_err_msgs, all_mark_data):
        """除了首尾的空格，以及标点符号、Python等词汇，需要全词匹配。如果能够实现全词匹配，则记为1分，否则记为0分"""
        f = open("res.txt", "a")
        score = 0
        for aid in all_err_msgs:
            correct_err_msg = trim_n(remove_python_str(all_mark_data[aid].err_msg))
            origin_err_msg = trim_n(remove_python_str(all_err_msgs[aid]))
            if origin_err_msg == correct_err_msg:
                score += 1
            f.write("aid: %d\n" % aid)
            f.write("correct: ")
            f.write(correct_err_msg)
            f.write("\ntrained: ")
            try:
                f.write(origin_err_msg)
            except:
                f.write("<Exception>")
            f.write("\n\n")
        f.close()
        return score


class SolveValidation:

    @staticmethod
    def whole_lines(all_solve_lines, all_mark_data):
        """
        对比通过算法推算的解决问题的行标，与通过标注确认的解决问题行标。如果行标完全一致，则记1分，否则记0分。
        :param all_solve_lines: 通过算法确认的解决问题的行标
        :param all_mark_data: 通过标注确认的错误、场景、解决方案的信息
        :return: 分数、总分
        """
        line_score = 0  # 程序在行标判断中的得分
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
                    if solve_lines - half_c_solve_lines == c_solve_lines - half_c_solve_lines:
                        line_score += 1
        return line_score, total_score

    @staticmethod
    def line_ratio(all_solve_lines, all_mark_data):
        """
        对比通过算法推算的解决问题的行标，与通过标注确认的解决问题行标。计分规则为：（判定数据与正确数据的交集/判定数据与正确数据的并集）。半解决问题的信息不纳入统计。
        例如，标注的解决问题的行标是{1,3,5}，推算的行标是{3,5,6}，那么最后的得分是 0.5
        :param all_solve_lines: 通过算法确认的解决问题的行标
        :param all_mark_data: 通过标注确认的错误、场景、解决方案的信息
        :return: 分数、总分
        """
        f = open("res_lines.txt", "a")
        line_score = 0  # 程序在行标判断中的得分
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
                    line_score += len((solve_lines & c_solve_lines) - half_c_solve_lines) / len((solve_lines | c_solve_lines) - half_c_solve_lines)
                f.write("aid: %d\n" % aid)
                f.write("correct lines: ")
                f.write(str(c_solve_lines))
                f.write("\ntrained lines: ")
                try:
                    f.write(str(solve_lines))
                except:
                    f.write("<Exception>")
                f.write("\n\n")
        f.close()
        return line_score, total_score

    @staticmethod
    def whole_secs(all_solve_secs, all_mark_data: Dict[int, MarkData], handled_data, sec_mark_list_real: Dict[int, List[List[int]]]):
        """
        对比通过算法推算的解决问题的开始/结束信号，与通过标注确认的解决问题行标。需要在句子层面全部匹配，才能认定为准确
        :param all_solve_secs: 通过算法确认的解决问题的所在的段落ID
        :param all_mark_data: 通过标注确认的错误、场景、解决方案的信息
        :param handled_data: 经过一些初步处理后的各篇文章的内容，包括段落、句子，以及一些对应关系
        :param sec_mark_list_real: 标注好的（或程序生成的）区段信息
        :return: 分数、总分
        """
        f = open("res_secs.txt", "a")
        sec_score = 0  # 程序在行标判断中的得分
        ratio_score = 0
        total_score = 0  # 总分

        for aid in all_solve_secs:
            if all_mark_data[aid].err_msg != str():
                total_score += 1
                # 判断算法推算和标注的结果，在句子层面是否准确
                trained_sec = []
                is_correct = True
                correct_count = 0
                wrong_count = 0
                for para_it in range(handled_data[aid].num_para):
                    for sentence_it in range(len(handled_data[aid].sentences[para_it])):
                        if all_solve_secs[aid][para_it][sentence_it] != sec_mark_list_real[aid][para_it][sentence_it]:
                            is_correct = False
                            if all_solve_secs[aid][para_it][sentence_it] in [1, 3, 4] or sec_mark_list_real[aid][para_it][sentence_it] in [1, 3, 4]:
                                wrong_count += 1
                        if all_solve_secs[aid][para_it][sentence_it] == sec_mark_list_real[aid][para_it][sentence_it] and all_solve_secs[aid][para_it][sentence_it] in [1, 3, 4]:
                            correct_count += 1
                        if all_solve_secs[aid][para_it][sentence_it] == 1:
                            trained_sec.append("BEG: %d-%d\n" % (para_it, sentence_it))
                        elif all_solve_secs[aid][para_it][sentence_it] == 3:
                            trained_sec.append("END: %d-%d\n" % (para_it, sentence_it))
                        elif all_solve_secs[aid][para_it][sentence_it] == 4:
                            trained_sec.append("RV: %d-%d\n" % (para_it, sentence_it))
                if is_correct:
                    sec_score += 1
                ratio_score_1a = 1 if correct_count + wrong_count == 0 else correct_count / (correct_count + wrong_count)
                ratio_score += ratio_score_1a
                f.write("aid: %d\n" % aid)
                f.write("\ntrained secs: ")
                try:
                    for t in trained_sec:
                        f.write(t)
                    f.write("is correct: %d, correct_ratio: %.2f\n" % (is_correct, ratio_score_1a))
                except:
                    f.write("<Exception>")
                f.write("\n\n")
        f.close()
        return sec_score, ratio_score, total_score

    @staticmethod
    def sentence_ratio(all_solve_msgs, all_mark_data):
        """
        对比通过算法推算的解决问题的信息，与通过标注确认的解决问题信息。计分规则为：（判定数据与正确数据的交集/判定数据与正确数据的并集）。半解决问题的信息不纳入统计。
        例如：
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

        msg_score = 0  # 程序在信息判断中得分
        total_score = 0  # 总分
        for aid in all_solve_msgs:
            if all_mark_data[aid].err_msg != str():
                total_score += 1
                # 计算报错消息判断中的得分
                c_solve_msgs = []
                half_c_solve_msgs = []
                for t0 in range(len(all_mark_data[aid].solves)):
                    if all_mark_data[aid].solve_weight[t0] == 1:
                        c_solve_msgs.extend(clean_str_func([all_mark_data[aid].solves[t0]]))
                    elif all_mark_data[aid].solve_weight[t0] == 0.5:
                        half_c_solve_msgs.extend(clean_str_func([all_mark_data[aid].solves[t0]]))
                solve_msgs = clean_str_func(all_solve_msgs[aid])
                c_solve_msg_set = set()
                half_c_solve_msg_set = set()
                solve_msg_set = set()
                for msg in c_solve_msgs:
                    msg2 = trim_number(trim_n(msg))
                    c_solve_msg_set.add(msg2)
                for msg in half_c_solve_msgs:
                    msg2 = trim_number(trim_n(msg))
                    half_c_solve_msg_set.add(msg2)
                for msg in solve_msgs:
                    msg2 = trim_number(trim_n(msg))
                    solve_msg_set.add(msg2)
                if len(c_solve_msg_set) == 0:
                    if len(solve_msg_set - half_c_solve_msg_set) == 0:
                        msg_score += 1
                else:
                    msg_score += len((solve_msg_set & c_solve_msg_set) - half_c_solve_msg_set) / len((solve_msg_set | c_solve_msg_set) - half_c_solve_msg_set)
        return msg_score, total_score

    @staticmethod
    def line_ratio_2(all_solve_lines, all_mark_data, all_smbl):
        """
        对比通过算法推算的解决问题的行标，与通过标注确认的解决问题行标。计分规则为：（判定数据与正确数据的交集/判定数据与正确数据的并集）。半解决问题的信息不纳入统计。
        例如，标注的解决问题的行标是{1,3,5}，推算的行标是{3,5,6}，那么最后的得分是 0.5
        :param all_solve_lines: 通过算法确认的解决问题的行标
        :param all_mark_data: 通过标注确认的错误、场景、解决方案的信息
        :return: 分数、总分
        """
        from train_ml.solve2 import SolveSecsInput
        f = open("res_lines.txt", "a")
        line_score = 0  # 程序在行标判断中的得分
        total_score = 0  # 总分
        for aid in all_solve_lines:
            if all_mark_data[aid].err_msg != str():
                total_score += 1
                # 计算行标判断中的得分
                solve_lines = set(all_solve_lines[aid])
                c_solve_lines = set()
                half_c_solve_lines = set()
                for t0 in range(len(all_mark_data[aid].solve_lines)):
                    if all_mark_data[aid].solve_lines[t0] not in all_smbl[aid] or all_smbl[aid][all_mark_data[aid].solve_lines[t0]] == SolveSecsInput.S_IN:
                        continue
                    if all_mark_data[aid].solve_weight[t0] == 1:
                        c_solve_lines.add(all_mark_data[aid].solve_lines[t0])
                    elif all_mark_data[aid].solve_weight[t0] == 0.5:
                        half_c_solve_lines.add(all_mark_data[aid].solve_lines[t0])
                if len(c_solve_lines) == 0:
                    if len(solve_lines - half_c_solve_lines) == 0:
                        line_score += 1
                else:
                    line_score += len((solve_lines & c_solve_lines) - half_c_solve_lines) / len((solve_lines | c_solve_lines) - half_c_solve_lines)
                f.write("aid: %d\n" % aid)
                f.write("correct lines: ")
                f.write(str(c_solve_lines))
                f.write("\ntrained lines: ")
                try:
                    f.write(str(solve_lines))
                except:
                    f.write("<Exception>")
                f.write("\n\n")
        f.close()
        return line_score, total_score

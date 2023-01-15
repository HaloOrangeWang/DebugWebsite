from settings import *
from data_io.load_data import MarkData
from .funcs import WordVec1Para, WordVec1Article, has_chn_chr, is_chn_eng_number, is_eng_number, has_chn_or_eng_chr
from .word_vec import WvNormal, WvCode
from .model import BilstmClassifyModel, BilstmCRFModel, BilstmCRFExpandModel
from validations.basic import SolveValidation
from typing import List, Dict, Tuple
import numpy as np
import copy
import torch
import time


class SolveClassifyConfigC:
    """将英文当做<code>时，模型的配置"""
    embedding_size = 70  # 词向量维度
    hidden_size = 64  # BILSTM有多少个隐藏单元
    # dropout = 0.2
    nepoch = 20  # 一共训练多少轮
    batch_size = 30  # 每一个批次将多少组数据提交给BILSTM-CRF模型
    lr = 0.005
    lr_decay = 0.85  # 学习速率衰减
    # true_sample_ratio_lb = 0.25  # 至少多少比例的训练样本应当为正样本。用于避免负样本比例过大


class SolveMsgConfigC:
    """将英文当做<code>时，模型的配置"""
    embedding_size = 70  # 词向量维度
    hidden_size = 64  # BILSTM有多少个隐藏单元
    # dropout = 0.2
    nepoch = 20  # 一共训练多少轮
    batch_size = 30  # 每一个批次将多少组数据提交给BILSTM-CRF模型
    lr = 0.005
    lr_decay = 0.85  # 学习速率衰减


class SolveMsgConfigN:
    """英文正常解析时，模型的配置"""
    embedding_size = 68  # 词向量维度
    hidden_size = 64  # BILSTM有多少个隐藏单元
    # dropout = 0.2
    nepoch = 20  # 一共训练多少轮
    batch_size = 30  # 每一个批次将多少组数据提交给BILSTM-CRF模型
    lr = 0.005
    lr_decay = 0.85  # 学习速率衰减


def get_start_dx(text) -> Tuple[bool, int]:
    """
    从一段文本中，去除开头的 1. a) 等内容
    :param text:
    :return: 这段话是否有值得训练的内容，以及训练内容的起始位置
    """
    st_dx = len(text)
    for t in range(len(text)):
        if is_chn_eng_number(text[t]):
            st_dx = t
            break
    if st_dx >= len(text) - 1:
        return False, -1
    # 识别这个段落是否为1. a) 等为开头。如果是的话，它们不计入训练样本。
    if text[st_dx] in ["一", "二", "三", "四", "五", "六", "七", "八", "九"] or is_eng_number(text[st_dx]):
        if not is_chn_eng_number(text[st_dx + 1]):
            # 第一个字符为单个数字或汉字数字或英文，但第二个字符是空格或标点符号，说明大概率是“1.” “a)”之类的内容，将其去掉
            is_find = False
            for t in range(st_dx + 1, len(text)):
                if is_chn_eng_number(text[t]):
                    st_dx = t
                    is_find = True
                    break
            if not is_find:
                return False, -1
    # 如果段落中只有空格、标点和【一、】，则这一段不能作为训练样本
    if not has_chn_or_eng_chr(text[st_dx:]):
        return False, -1
    return True, st_dx


class SolveSecsInput:
    """这个类用于生成对解决问题的区间的训练数据"""
    # 一些enum值。分别表示"与解决方案无关"、"解决方案的起始"、"解决方案的中间"、"解决方案的终止信号"
    S_NO = 0
    S_BEG = 1
    S_IN = 2
    S_END = 3
    S_END_AND_BEG = 4  # 既有解决方案首尾的信号，也有下一个解决方案起始的信号

    def __init__(self, wdict: dict, rv_wdict: list):
        self.wdict = wdict
        self.rv_wdict = rv_wdict
        self.all_sec_marks = dict()  # 每篇文章每个段落是否包含解决方案的起始信号和终止信号的标注信息。数值为上面的S_XXX的enum值
        self.beg_train_input = []  # 训练“哪些段落是解决方案的起始”的模型输入数据
        self.beg_train_output = []  # 训练“哪些段落是解决方案的起始”的模型输出数据
        self.beg_train_lens = []  # 训练“哪些段落是解决方案的起始”的输入数据长度
        self.beg_train_para_ratio = []  # 训练“哪些段落是解决方案的起始”时，所挑选的段落在整个文章中的百分比（计算方法为 (该段落id+1)/文章段落数+1）
        self.end_train_input = []  # 训练“哪些段落是解决方案的终止”的模型输入数据
        self.end_train_output = []  # 训练“哪些段落是解决方案的终止”的模型输出数据
        self.end_train_lens = []  # 训练“哪些段落是解决方案的终止”的输入数据长度
        self.end_train_para_ratio = []  # 训练“哪些段落是解决方案的终止”时，所挑选的段落在整个文章中的百分比

    def get_solve_dx_list(self, paragraphs: list, solve_secs: list, lnbp_1article) -> List[int]:
        """
        根据文章内容，获取哪些段落是解决方案的起始，哪些段落是解决方案的终止
        :param paragraphs: 文章的段落列表
        :param solve_secs: 标注好的解决方案的区间
        :param lnbp_1article: 这篇文章中，每个段落对应的行标的列表
        :return: 每个段落是否包含解决方案的起始信号和终止信号的标注信息
        """
        solve_dx_lists = [self.S_NO for t in range(len(paragraphs))]  # 每一段的解决问题信息的列表
        # 如果这篇文章中没有解决方案的段落标注，则所有段落都可认为“不是解决方案的起始段落”
        if len(solve_secs) == 0:
            return solve_dx_lists
        # 构建行标和段落之间的对应关系
        line_dx_to_para_dx = dict()
        for para_it in range(len(paragraphs)):
            line_nos = range(lnbp_1article[para_it] - paragraphs[para_it].count('\n'), lnbp_1article[para_it] + 1)
            for line_no in line_nos:
                line_dx_to_para_dx[line_no] = para_it
        # 针对每一段，判断是否为解决方案的起始/终止信号
        solve_sec_dx = 0
        sec_beg_line_dx = solve_secs[solve_sec_dx][0]
        sec_end_line_dx = solve_secs[solve_sec_dx][1]
        for para_it in range(len(paragraphs)):
            if para_it < line_dx_to_para_dx[sec_beg_line_dx]:
                solve_dx_lists[para_it] = self.S_NO
            elif para_it == line_dx_to_para_dx[sec_beg_line_dx]:
                solve_dx_lists[para_it] = self.S_BEG
            elif sec_end_line_dx == "eof" or para_it < line_dx_to_para_dx[sec_end_line_dx]:
                solve_dx_lists[para_it] = self.S_IN
            elif para_it == line_dx_to_para_dx[sec_end_line_dx]:
                if len(solve_secs) > solve_sec_dx + 1:
                    solve_sec_dx += 1
                    sec_beg_line_dx = solve_secs[solve_sec_dx][0]
                    sec_end_line_dx = solve_secs[solve_sec_dx][1]
                    if para_it == line_dx_to_para_dx[sec_beg_line_dx]:
                        solve_dx_lists[para_it] = self.S_END_AND_BEG
                    else:
                        solve_dx_lists[para_it] = self.S_END
                else:
                    solve_dx_lists[para_it] = self.S_END
            else:
                solve_dx_lists[para_it] = self.S_NO
        return solve_dx_lists

    def get_data_1article(self, paragraphs: list, wcode_list: List[WordVec1Para], sec_mark_list: List[int], para_dx_to_wcode_dx: Dict[int, int]):
        """
        根据文章内容，生成训练数据
        :param paragraphs: 段落的正文内容
        :param wcode_list: 将文本编码之后的内容
        :param sec_mark_list: 每个段落是否包含解决方案的起始信号和终止信号的标注信息
        :param para_dx_to_wcode_dx: 段落的索引值和此编码数组索引值之间的关系
        """
        for para_it in range(len(paragraphs)):
            # 如果这段文字没有对应的词编码的话，则不作为解决问题信息的训练内容
            if para_it not in para_dx_to_wcode_dx:
                continue
            wcode_dx = para_dx_to_wcode_dx[para_it]
            # 去除开头的空格。如果段落开头以【一、】、【1.】开头，则将开头内容去掉
            is_valid_train, st_dx = get_start_dx(paragraphs[para_it])
            if not is_valid_train:
                continue
            # 生成训练样本
            para_ratio = (para_it + 1) / (len(paragraphs) + 1)
            para_ratio = (para_ratio - 0.5) * 4
            input_data = []
            for word_it in range(0, len(wcode_list[wcode_dx].vec)):
                if wcode_list[wcode_dx].vec_to_text_dx_list[word_it] + wcode_list[wcode_dx].vec_len_list[word_it] < st_dx:
                    continue
                input_data.append(wcode_list[wcode_dx].vec[word_it] + OUTPUT_DIC_SIZE)  # 输入数据需要加上OUTPUT_DIC_SIZE，以避免使用O_IDX等特殊字符
                # 判断是否要插入一条新的训练数据
                set_new_train_data = False
                if word_it == len(wcode_list[wcode_dx].vec) - 1:
                    # 如果到达了分段的位置，则判断本段。如果没有中文内容，且句子太短，则不作为训练内容，直接将input_data清空
                    if len(input_data) <= 2 and min(input_data) >= len(self.rv_wdict) + OUTPUT_DIC_SIZE:
                        input_data = []
                    else:
                        set_new_train_data = True
                else:
                    # 未达到分段位置：如果句子长度超过了100，则按照标点符号进行拆分。保证拆分后每条数据的长度都略微超过100。但如果该段落包含解决方案的起始/终止信号，则该段落不能拆分。
                    if len(input_data) >= 100 and wcode_list[wcode_dx].vec[word_it] == len(self.wdict) + CHN_PUNC_WRD and sec_mark_list[para_it] not in [self.S_BEG, self.S_END, self.S_END_AND_BEG]:
                        set_new_train_data = True
                if set_new_train_data:
                    if sec_mark_list[para_it] == self.S_NO:
                        if para_it != len(paragraphs) - 1:  # 每篇文章的最后一个段落不能包含解决方案的起始信号
                            self.beg_train_input.append(copy.copy(input_data))
                            self.beg_train_output.append(FALSE_IDX)
                            self.beg_train_lens.append(len(input_data))
                            self.beg_train_para_ratio.append(para_ratio)
                    elif sec_mark_list[para_it] == self.S_BEG:
                        if para_it != len(paragraphs) - 1:  # 每篇文章的最后一个段落不能包含解决方案的起始信号
                            self.beg_train_input.append(copy.copy(input_data))
                            self.beg_train_output.append(TRUE_IDX)
                            self.beg_train_lens.append(len(input_data))
                            self.beg_train_para_ratio.append(para_ratio)
                    elif sec_mark_list[para_it] == self.S_IN:
                        self.end_train_input.append(copy.copy(input_data))
                        self.end_train_output.append(FALSE_IDX)
                        self.end_train_lens.append(len(input_data))
                        self.end_train_para_ratio.append(para_ratio)
                    elif sec_mark_list[para_it] == self.S_END:
                        self.end_train_input.append(copy.copy(input_data))
                        self.end_train_output.append(TRUE_IDX)
                        self.end_train_lens.append(len(input_data))
                        self.end_train_para_ratio.append(para_ratio)
                    elif sec_mark_list[para_it] == self.S_END_AND_BEG:
                        self.end_train_input.append(copy.copy(input_data))
                        self.end_train_output.append(TRUE_IDX)
                        self.end_train_lens.append(len(input_data))
                        self.end_train_para_ratio.append(para_ratio)
                        if para_it != len(paragraphs) - 1:  # 每篇文章的最后一个段落不能包含解决方案的起始信号
                            self.beg_train_input.append(copy.copy(input_data))
                            self.beg_train_output.append(TRUE_IDX)
                            self.beg_train_lens.append(len(input_data))
                            self.beg_train_para_ratio.append(para_ratio)
                    input_data = []

    def get_model_data(self, articles, wcode_list: Dict[int, WordVec1Article], mark_data: Dict[int, MarkData], line_no_by_para, punc_code_set, test_aid_list):
        """
        根据所有标注好的文章，生成错误信息的训练数据和测试数据
        :param articles: 文章列表
        :param wcode_list: 每篇文章中，将词进行编码后的结果
        :param mark_data: 文章标注的情况
        :param line_no_by_para: 每篇文章中，每一段对应的尾行的行标
        :param punc_code_set: 哪些字符表示标点符号
        :param test_aid_list: 哪些文章是用于测试的
        """
        for aid in mark_data:
            if not mark_data[aid].err_msg:
                continue
            if aid in test_aid_list:
                continue
            paragraphs = articles[aid].text.rstrip('\n').replace('\n\n\n', '\n\n').split('\n\n')
            para_dx_to_wcode_dx = dict()
            for wcode_it in range(len(wcode_list[aid].text_c)):
                para_dx_to_wcode_dx[wcode_list[aid].text_c[wcode_it].paragraph_dx] = wcode_it
            self.all_sec_marks[aid] = self.get_solve_dx_list(paragraphs, mark_data[aid].solve_secs, line_no_by_para[aid])
            self.get_data_1article(paragraphs, wcode_list[aid].text_c, self.all_sec_marks[aid], para_dx_to_wcode_dx)


class SolveInParaInput:
    """生成段落内有解决问题的信息的训练数据"""

    def __init__(self, wdict, rv_wdict):
        self.wdict = wdict
        self.rv_wdict = rv_wdict
        self.train_input = []  # 训练用的输入数据。
        self.train_output = []  # 训练用的输出数据。
        self.train_lens = []  # 训练用的数据的长度
        self.para_ratio = []  # 训练解决方案时，所挑选的段落在整个文章中的百分比，并线性扩展至[-2, 2]区间内
        self.has_solve_sec = []  # 该段落所在的文章是否存在解决方案的区段。-1表示没有解决方案的区段，1表示有解决方案的区段
        self.solve_sec_mark = []  # 对于有解决方案区段的文章，该段落是否位于解决方案区段的起始位置。-1表示不为起始位置，1表示为起始位置

    # noinspection PyMethodMayBeStatic
    def get_solve_dx_list(self, paragraphs: list, solve_msgs: list, solve_lines: list, lnbp_1article) -> List[List[int]]:
        """
        获取一篇文章中，解决问题的信息所在的位置。返回的方式为[(段落id, 字符在段落中的位置)]。不处理标题，因为文章的解决方案不能在标题中
        :param paragraphs: 文章的段落列表
        :param solve_msgs: 解决问题的信息列表
        :param solve_lines: 解决问题的信息所在的行标
        :param lnbp_1article: 这篇文章中，每个段落对应的行标的列表
        :return:
        """
        solve_dx_lists = [list() for t in range(len(paragraphs))]  # 每一段的解决问题信息的列表
        # 构建行标和段落之间的对应关系
        line_dx_to_para_dx = dict()
        for para_it in range(len(paragraphs)):
            line_nos = range(lnbp_1article[para_it] - paragraphs[para_it].count('\n'), lnbp_1article[para_it] + 1)
            for line_no in line_nos:
                line_dx_to_para_dx[line_no] = para_it
        for solve_it in range(len(solve_msgs)):
            # 对于每一条解决信息，找到它们所在的段落。
            line = solve_lines[solve_it]
            paragraph_dx = line_dx_to_para_dx[line]
            # 这一段落应当包含该条解决问题的信息，将它记录下来
            dx = paragraphs[paragraph_dx].index(solve_msgs[solve_it])
            solve_dx_lists[paragraph_dx] = list(range(dx, dx + len(solve_msgs[solve_it])))
        return solve_dx_lists

    def get_data_1article(self, paragraphs: list, wcode_list: List[WordVec1Para], solve_dx_lists: List[List[int]], para_dx_to_wcode_dx: Dict[int, int], sec_mark_list: List[int]):
        """
        根据文章内容，生成训练数据
        :param paragraphs: 段落的正文内容
        :param wcode_list: 将文本编码之后的内容
        :param solve_dx_lists: 解决问题的信息在文章中的位置
        :param para_dx_to_wcode_dx: 段落的索引值和此编码数组索引值之间的关系
        :param sec_mark_list: 每个段落是否包含解决方案的起始信号和终止信号的标注信息
        """
        # 1.获取每一段话中，对应的解决方案和文本编码结果之间的对应关系
        is_solve_by_word = [list() for t in range(len(solve_dx_lists))]  # 每一段的解决问题信息的列表
        for para_it in range(len(solve_dx_lists)):
            if para_it not in para_dx_to_wcode_dx:
                continue
            wcode_dx = para_dx_to_wcode_dx[para_it]
            is_solve_by_word_1para = [False for t in range(len(wcode_list[wcode_dx].vec))]
            wcode_dx2 = 0
            solve_dx = 0
            while True:
                if wcode_dx2 >= len(wcode_list[wcode_dx].vec) or solve_dx >= len(solve_dx_lists[para_it]):
                    break
                if wcode_list[wcode_dx].vec_to_text_dx_list[wcode_dx2] == solve_dx_lists[para_it][solve_dx]:
                    is_solve_by_word_1para[wcode_dx2] = True
                    wcode_dx2 += 1
                    solve_dx += 1
                elif wcode_list[wcode_dx].vec_to_text_dx_list[wcode_dx2] < solve_dx_lists[para_it][solve_dx]:
                    wcode_dx2 += 1
                else:
                    solve_dx += 1
            is_solve_by_word[para_it] = is_solve_by_word_1para
        # 2.生成训练数据。
        for para_it in range(len(solve_dx_lists)):
            para_ratio = (para_it + 1) / (len(paragraphs) + 1)
            para_ratio = (para_ratio - 0.5) * 4
            has_solve_sec = -1
            for sec_mark in sec_mark_list:
                if sec_mark != SolveSecsInput.S_NO:
                    has_solve_sec = 1
                    break
            if sec_mark_list[para_it] != SolveSecsInput.S_NO:
                solve_sec_mark = 1
            else:
                solve_sec_mark = -1
            input_data = []  # copy.deepcopy(wcode.vec)
            output_data = []
            # 以下情况不作为训练数据：段落没有对应的词编码信息，段落处在解决方案的起始信号和终止信号之间，段落内容除了 1) a. 以外没有有效内容。
            if para_it not in para_dx_to_wcode_dx:
                continue
            if sec_mark_list[para_it] == SolveSecsInput.S_IN:
                continue
            wcode_dx = para_dx_to_wcode_dx[para_it]
            is_valid_train, st_dx = get_start_dx(paragraphs[para_it])
            if not is_valid_train:
                continue
            # 生成这段话的输入信息和输出信息
            is_first_word = True
            for word_it in range(0, len(wcode_list[wcode_dx].vec)):
                # 对于一段解决问题的信息，输出的数据应当为 [起始标志B_IDX、解决方案过程中的标志I_IDX、结束标志E_IDX]
                # 例如：对于 [进程   被    占用   ，     关掉   其他   进程   即可]，输出的内容应当为
                #          [O_IDX O_IDX O_IDX O_IDX B_IDX I_IDX E_IDX O_IDX]
                input_data.append(wcode_list[wcode_dx].vec[word_it] + OUTPUT_DIC_SIZE)  # 如果使用事先准备好的词向量列表，则输入数据需要加上OUTPUT_DIC_SIZE，以避免使用O_IDX等特殊字符
                if is_solve_by_word[para_it][word_it]:
                    if is_first_word or (not is_solve_by_word[para_it][word_it - 1]):
                        output_data.append(B_IDX)
                    elif word_it == len(wcode_list[wcode_dx].vec) - 1 or (not is_solve_by_word[para_it][word_it + 1]):
                        output_data.append(E_IDX)
                    else:
                        output_data.append(I_IDX)
                else:
                    output_data.append(O_IDX)
                is_first_word = False
                # 如果句子长度超过了50，则按照标点符号进行拆分。保证拆分后每条数据的长度都略微超过50，且解决问题的信息不能跨条目。
                if word_it == len(wcode_list[wcode_dx].vec) - 1 or (len(input_data) >= 50 and wcode_list[wcode_dx].vec[word_it] == len(self.wdict) + CHN_PUNC_WRD and output_data[-1] == O_IDX and output_data[-2] == O_IDX):
                    input_data.append(EOS_IDX)
                    output_data.append(EOS_IDX)
                    self.train_input.append(copy.copy(input_data))
                    self.train_output.append(copy.copy(output_data))
                    self.train_lens.append(len(input_data))
                    self.para_ratio.append(para_ratio)
                    self.has_solve_sec.append(has_solve_sec)
                    self.solve_sec_mark.append(solve_sec_mark)
                    input_data = []
                    output_data = []

    def get_model_data(self, articles, wcode_list: Dict[int, WordVec1Article], mark_data: Dict[int, MarkData], line_no_by_para, all_sec_marks: Dict[int, List[int]], punc_code_set, test_aid_list):
        """
        根据所有标注好的文章，生成错误信息的训练数据和测试数据
        :param articles: 文章列表
        :param wcode_list: 每篇文章中，将词进行编码后的结果
        :param mark_data: 文章标注的情况
        :param line_no_by_para: 每篇文章中，每一段对应的尾行的行标
        :param punc_code_set: 哪些字符表示标点符号
        :param test_aid_list: 哪些文章是用于测试的
        :param all_sec_marks: 每篇文章每个段落是否包含解决方案的起始信号和终止信号的标注信息。数值为上面的S_XXX的enum值
        """
        for aid in mark_data:
            if not mark_data[aid].err_msg:
                continue
            if aid in test_aid_list:
                continue
            paragraphs = articles[aid].text.rstrip('\n').replace('\n\n\n', '\n\n').split('\n\n')
            para_dx_to_wcode_dx = dict()
            for wcode_it in range(len(wcode_list[aid].text_c)):
                para_dx_to_wcode_dx[wcode_list[aid].text_c[wcode_it].paragraph_dx] = wcode_it
            solve_dx_list = self.get_solve_dx_list(paragraphs, mark_data[aid].solves, mark_data[aid].solve_lines, line_no_by_para[aid])
            self.get_data_1article(paragraphs, wcode_list[aid].text_c, solve_dx_list, para_dx_to_wcode_dx, all_sec_marks[aid])


class SolvePipe:
    """生成解决问题的训练数据、调度模型进行训练并生成测试结果的管理类"""

    def __init__(self, wv_n: WvNormal, wv_c: WvCode):
        """
        :param wv_n: 将英文正常编码的方式，词向量的编码内容
        :param wv_c: 将英文编码为<code>的方式，词向量的编码内容
        """
        self.wv_n = wv_n
        self.wv_c = wv_c
        self.sec_data_c = SolveSecsInput(self.wv_c.wv.key_to_index, self.wv_c.wv.index_to_key)  # 训练解决问题的段落时，用的数据
        self.msg_data_c = SolveInParaInput(self.wv_c.wv.key_to_index, self.wv_c.wv.index_to_key)  # 训练段落内解决问题的信息时，用的数据
        # self.msg_data_n = SolveInParaInput(self.wv_n.wv.key_to_index, self.wv_n.wv.index_to_key)  # 训练段落内解决问题的信息（英文不编码为<code>）时，用的数据
        self.config_c = SolveClassifyConfigC()
        self.msg_config_c = SolveMsgConfigC()
        # self.msg_config_n = SolveMsgConfigN()
        self.model_sec_beg = BilstmClassifyModel(self.config_c, self.wv_c.emb_mat, freeze=True)  # 训练解决方案起始信号的模型
        self.model_sec_end = BilstmClassifyModel(self.config_c, self.wv_c.emb_mat, freeze=True)  # 训练解决方案终止信号的模型
        self.model_msg_c = BilstmCRFExpandModel(self.msg_config_c, self.wv_c.emb_mat, freeze=True)  # 训练段落内的解决方案信息的模型
        # self.model_msg_n = BilstmCRFExpandModel(self.msg_config_n, self.wv_n.emb_mat, freeze=False)  # 训练段落内的解决方案信息（英文不编码为<code>）的模型
        self.test_aid_list = []

    def setup(self, articles, mark_data, line_no_by_para, punc_code_set):
        """准备模型的输入数据，以及相关配置等"""
        # 设置哪些数据用于训练，而哪些数据用于测试
        aid_list = [t for t in mark_data]
        aid_list = sorted(aid_list)
        self.test_aid_list = aid_list[int(TEST_DATA_START * len(aid_list)): int((TEST_DATA_START + TEST_DATA_RATIO) * len(aid_list))]
        # 准备训练数据。包括解决方案区段的训练数据
        self.sec_data_c.get_model_data(articles, self.wv_c.wcode_list, mark_data, line_no_by_para, punc_code_set, self.test_aid_list)
        self.msg_data_c.get_model_data(articles, self.wv_c.wcode_list, mark_data, line_no_by_para, self.sec_data_c.all_sec_marks, punc_code_set, self.test_aid_list)
        # self.msg_data_n.get_model_data(articles, self.wv_n.wcode_list, mark_data, line_no_by_para, self.sec_data_c.all_sec_marks, punc_code_set, self.test_aid_list)

    # noinspection PyMethodMayBeStatic
    def sec_batch_preprocess(self, train_input: List, train_output: List, train_lens: List, train_para_ratios: List, indexs, start_dx, end_dx):
        """
        对batch数据进行预处理：获取长度排名在start_dx~end_dx之间的数据，并通过填充PAD_IDX的方式，将数组中各项的长度一致
        :param train_input: 模型的训练输入数据列表
        :param train_output: 模型的训练输出数据列表
        :param train_lens: 模型的每组训练数据的长度的列表
        :param train_para_ratios: 每组训练数据的段落，在整篇文章中的位置
        :param indexs: 将训练数据根据长度从大到小排序后的索引值
        :param start_dx: 这个batch的起始位置
        :param end_dx: 这个batch的终止位置
        :return: 处理好的训练数据、训练预期输出、各组数据的长度
        """
        input_data = []
        output_data = []
        lens = []
        para_ratios = []
        # 1.对报错信息按照长短重新排序
        sorted_index = np.argsort(np.array(train_lens)[indexs[start_dx: (end_dx + 1)]] * (-1))
        max_len = train_lens[indexs[start_dx + sorted_index[0]]]
        input_data.append(train_input[indexs[start_dx + sorted_index[0]]])
        output_data.append(train_output[indexs[start_dx + sorted_index[0]]])
        lens.append(train_lens[indexs[start_dx + sorted_index[0]]])
        para_ratios.append(train_para_ratios[indexs[start_dx + sorted_index[0]]])
        for dx in range(1, len(sorted_index)):
            pad_num = max_len - train_lens[indexs[start_dx + sorted_index[dx]]]
            pad_list = [PAD_IDX for t in range(pad_num)]
            input_data.append(train_input[indexs[start_dx + sorted_index[dx]]] + pad_list)
            output_data.append(train_output[indexs[start_dx + sorted_index[dx]]])
            lens.append(train_lens[indexs[start_dx + sorted_index[dx]]])
            para_ratios.append(train_para_ratios[indexs[start_dx + sorted_index[dx]]])
        return torch.LongTensor(input_data), torch.LongTensor(output_data), torch.LongTensor(lens), torch.FloatTensor(para_ratios)

    # noinspection PyMethodMayBeStatic
    def msg_batch_preprocess(self, train_data: SolveInParaInput, indexs, start_dx, end_dx):
        """
        对batch数据进行预处理：获取长度排名在start_dx~end_dx之间的数据，并通过填充PAD_IDX的方式，将数组中各项的长度一致
        :param train_data: 使用哪类训练数据（可选项：self.data_n，self.data_c）
        :param indexs: 将训练数据根据长度从大到小排序后的索引值
        :param start_dx: 这个batch的起始位置
        :param end_dx: 这个batch的终止位置
        :return: 处理好的训练数据、训练预期输出、各组数据的长度
        """
        input_data = []
        output_data = []
        lens = []
        para_ratios = []
        has_solve_secs = []
        solve_sec_marks = []
        # 1.对报错信息按照长短重新排序
        sorted_index = np.argsort(np.array(train_data.train_lens)[indexs[start_dx: (end_dx + 1)]] * (-1))
        max_len = train_data.train_lens[indexs[start_dx + sorted_index[0]]]
        input_data.append(train_data.train_input[indexs[start_dx + sorted_index[0]]])
        output_data.append(train_data.train_output[indexs[start_dx + sorted_index[0]]])
        lens.append(train_data.train_lens[indexs[start_dx + sorted_index[0]]])
        para_ratios.append(train_data.para_ratio[indexs[start_dx + sorted_index[0]]])
        has_solve_secs.append(train_data.has_solve_sec[indexs[start_dx + sorted_index[0]]])
        solve_sec_marks.append(train_data.solve_sec_mark[indexs[start_dx + sorted_index[0]]])
        for dx in range(1, len(sorted_index)):
            pad_num = max_len - train_data.train_lens[indexs[start_dx + sorted_index[dx]]]
            pad_list = [PAD_IDX for t in range(pad_num)]
            input_data.append(train_data.train_input[indexs[start_dx + sorted_index[dx]]] + pad_list)
            output_data.append(train_data.train_output[indexs[start_dx + sorted_index[dx]]] + pad_list)
            lens.append(train_data.train_lens[indexs[start_dx + sorted_index[dx]]])
            para_ratios.append(train_data.para_ratio[indexs[start_dx + sorted_index[dx]]])
            has_solve_secs.append(train_data.has_solve_sec[indexs[start_dx + sorted_index[dx]]])
            solve_sec_marks.append(train_data.solve_sec_mark[indexs[start_dx + sorted_index[dx]]])
        return torch.LongTensor(input_data), torch.LongTensor(output_data), torch.LongTensor(lens), torch.FloatTensor(para_ratios), torch.LongTensor(has_solve_secs), torch.LongTensor(solve_sec_marks)

    def train_sec_model(self, model: BilstmClassifyModel, config, train_input: List[List[int]], train_output: List[int], train_lens: List[int], train_para_ratios: List[int]):
        # 预处理：对训练数据进行均衡，使得一半数据拥有解决方案的信息。
        # train_input, train_output, train_lens = self.data_balance(train_input2, train_output2, train_lens2)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_decay)  # 每执行一次optimizer之后，学习速率衰减一定比例
        for epoch in range(config.nepoch):
            model.train()
            total_loss = 0.
            start_time = time.time()
            # 1.将训练数据打乱顺序（待定）
            indexs = np.random.permutation(len(train_lens))
            # 2.进行训练
            num_batch = len(train_input) // config.batch_size
            for batch in range(num_batch):
                # 2.1 以填充PAD_IDX的方式，使各个数组的长度一致
                input_data, output_data, lengths, para_ratios = self.sec_batch_preprocess(train_input, train_output, train_lens, train_para_ratios, indexs, batch * config.batch_size, (batch + 1) * config.batch_size - 1)
                # 2.2 前向传输：根据参数计算损失函数
                optimizer.zero_grad()
                scores = model(input_data, lengths, para_ratios)
                loss = model.calc_loss(scores, output_data)
                # 2.3 反向传播：根据损失函数优化参数
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            end_time = time.time()
            print("epoch %d, time = %.2f, loss = %.6f" % (epoch, (end_time - start_time), total_loss))

    def train_msg_model(self, model: BilstmCRFExpandModel, config, train_data: SolveInParaInput):
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_decay)  # 每执行一词optimizer之后，学习速率衰减一定比例
        for epoch in range(config.nepoch):
            model.train()
            total_loss = 0.
            start_time = time.time()
            # 1.将训练数据打乱顺序（待定）
            indexs = np.random.permutation(len(train_data.train_lens))
            # 2.进行训练
            num_batch = len(train_data.train_input) // config.batch_size
            for batch in range(num_batch):
                # 2.1 以填充PAD_IDX的方式，使各个数组的长度一致
                input_data, output_data, lengths, para_ratios, has_solve_secs, solve_sec_marks = self.msg_batch_preprocess(train_data, indexs, batch * config.batch_size, (batch + 1) * config.batch_size - 1)
                # 2.2 前向传输：根据参数计算损失函数
                optimizer.zero_grad()
                scores = model(input_data, lengths, para_ratios, has_solve_secs, solve_sec_marks)
                loss = model.calc_loss(scores, output_data)
                # 2.3 反向传播：根据损失函数优化参数
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            end_time = time.time()
            print("epoch %d, time = %.2f, loss = %.6f" % (epoch, (end_time - start_time), total_loss))

    def train(self):
        """训练解决方案的起始信号、解决方案的终止信号"""
        self.train_sec_model(self.model_sec_beg, self.config_c, self.sec_data_c.beg_train_input, self.sec_data_c.beg_train_output, self.sec_data_c.beg_train_lens, self.sec_data_c.beg_train_para_ratio)
        self.train_sec_model(self.model_sec_end, self.config_c, self.sec_data_c.end_train_input, self.sec_data_c.end_train_output, self.sec_data_c.end_train_lens, self.sec_data_c.end_train_para_ratio)
        self.train_msg_model(self.model_msg_c, self.msg_config_c, self.msg_data_c)
        # self.train_msg_model(self.model_msg_n, self.msg_config_n, self.msg_data_n)

    def test_1article_sec(self, aid, wcode_1article: WordVec1Article, article):
        """
        测试一篇文章的解决方案的起始信号和终止信号是否准确
        :param aid: 文章ID
        :param wcode_1article: 这一篇文章的词编码
        :param article: 这一篇文章的标题和正文
        """
        paragraphs = article.text.rstrip('\n').replace('\n\n\n', '\n\n').split('\n\n')
        para_dx_to_wcode_dx = dict()
        for wcode_it in range(len(wcode_1article.text_c)):
            para_dx_to_wcode_dx[wcode_1article.text_c[wcode_it].paragraph_dx] = wcode_it
        # 获取一篇文章的起始信号和终止信号
        beg_para_dxs = list()  # 记录哪些段落包含解决方案的起始信号
        end_para_dxs = list()  # 记录哪些段落包含解决方案的终止信号
        status = False  # 当前段落是否进入解决方案的内部
        for para_it in range(len(paragraphs)):
            # 如果这段文字没有对应的词编码的话，则不作为解决问题信息的训练内容
            if para_it not in para_dx_to_wcode_dx:
                continue
            # 1.生成输入数据
            wcode_dx = para_dx_to_wcode_dx[para_it]
            wcode_1para = wcode_1article.text_c[wcode_dx]
            # 去除开头的空格。如果段落开头以【一、】、【1.】开头，则将开头内容去掉
            is_valid_para, st_dx = get_start_dx(paragraphs[para_it])
            if not is_valid_para:
                continue
            # 生成model的输入数据
            input_data = list()
            for word_it in range(0, len(wcode_1para.vec)):
                if wcode_1para.vec_to_text_dx_list[word_it] + wcode_1para.vec_len_list[word_it] < st_dx:
                    continue
                input_data.append(wcode_1para.vec[word_it] + OUTPUT_DIC_SIZE)
            # 如果到达了分段的位置，则判断本段。如果没有中文内容，且句子太短，则不处理这一条
            if len(input_data) <= 2 and min(input_data) >= len(self.wv_c.wv.index_to_key) + OUTPUT_DIC_SIZE:
                continue
            # 2.使用model进行预测。如果已经进入解决方案的区段，则判断解决方案的结束信号，否则判断解决方案的起始信号
            para_ratio = (para_it + 1) / (len(paragraphs) + 1)
            para_ratio = (para_ratio - 0.5) * 4
            para_ratio = torch.FloatTensor([para_ratio])
            length = torch.LongTensor([len(input_data)])
            input_data = torch.LongTensor(input_data).unsqueeze(dim=0)
            if status:
                output_data = self.model_sec_end.predict(input_data, length, para_ratio)[0]
                if output_data == TRUE_IDX:
                    end_para_dxs.append(para_it)
                    output_data2 = self.model_sec_beg.predict(input_data, length, para_ratio)[0]
                    if output_data2 == TRUE_IDX:
                        beg_para_dxs.append(para_it)
                        status = True
                    else:
                        status = False
            else:
                output_data = self.model_sec_beg.predict(input_data, length, para_ratio)[0]
                if output_data == TRUE_IDX:
                    beg_para_dxs.append(para_it)
                    status = True
        return beg_para_dxs, end_para_dxs

    # noinspection PyMethodMayBeStatic
    def get_sec_mark_by_para(self, paragraphs: list, solve_secs: list) -> List[int]:
        """
        根据文章内容，获取哪些段落是解决方案的起始，哪些段落是解决方案的终止。和 SolveSecsInput.get_solve_dx_list 的区别是，这个函数的输入参数solve_secs是段落标记，而不是行标
        :param paragraphs: 文章的段落列表
        :param solve_secs: 标注好的解决方案的区间
        :return: 每个段落是否包含解决方案的起始信号和终止信号的标注信息
        """
        solve_dx_lists = [SolveSecsInput.S_NO for t in range(len(paragraphs))]  # 每一段的解决问题信息的列表
        # 如果这篇文章中没有解决方案的段落标注，则所有段落都可认为“不是解决方案的起始段落”
        if len(solve_secs) == 0:
            return solve_dx_lists
        # 针对每一段，判断是否为解决方案的起始/终止信号
        solve_sec_dx = 0
        sec_beg_para_dx = solve_secs[solve_sec_dx][0]
        sec_end_para_dx = solve_secs[solve_sec_dx][1]
        for para_it in range(len(paragraphs)):
            if para_it < sec_beg_para_dx:
                solve_dx_lists[para_it] = SolveSecsInput.S_NO
            elif para_it == sec_beg_para_dx:
                solve_dx_lists[para_it] = SolveSecsInput.S_BEG
            elif sec_end_para_dx == "eof" or para_it < sec_end_para_dx:
                solve_dx_lists[para_it] = SolveSecsInput.S_IN
            elif para_it == sec_end_para_dx:
                if len(solve_secs) > solve_sec_dx + 1:
                    solve_sec_dx += 1
                    sec_beg_para_dx = solve_secs[solve_sec_dx][0]
                    sec_end_para_dx = solve_secs[solve_sec_dx][1]
                    if para_it == sec_beg_para_dx:
                        solve_dx_lists[para_it] = SolveSecsInput.S_END_AND_BEG
                    else:
                        solve_dx_lists[para_it] = SolveSecsInput.S_END
                else:
                    solve_dx_lists[para_it] = SolveSecsInput.S_END
            else:
                solve_dx_lists[para_it] = SolveSecsInput.S_NO
        return solve_dx_lists

    def test_1article_msg(self, aid, wcode_1article: WordVec1Article, article, solve_secs: list, line_no_by_para: Dict[int, List[int]], model: BilstmCRFExpandModel) -> Tuple[List[str], List[int]]:
        """
        测试一篇文章在解决问题的起止信号之外的内容是否存在解决方案的信息
        :param aid: 文章ID
        :param wcode_1article: 这一篇文章的词编码
        :param article: 这一篇文章的标题和正文
        :param solve_secs: 前一步算法判定的解决方案的区间
        :param line_no_by_para: 每篇文章中，每一段对应的尾行的行标
        :param model: 测试模型
        """
        solve_lines_1article = []
        solve_msg_1article = []
        paragraphs = article.text.rstrip('\n').replace('\n\n\n', '\n\n').split('\n\n')
        sec_mark_list = self.get_sec_mark_by_para(paragraphs, solve_secs)
        para_dx_to_wcode_dx = dict()
        for wcode_it in range(len(wcode_1article.text_c)):
            para_dx_to_wcode_dx[wcode_1article.text_c[wcode_it].paragraph_dx] = wcode_it
        for para_it in range(len(paragraphs)):
            # 以下情况不作为训练数据：段落没有对应的词编码信息，段落处在解决方案的起始信号和终止信号之间，段落内容除了 1) a. 以外没有有效内容。
            if para_it not in para_dx_to_wcode_dx:
                continue
            if sec_mark_list[para_it] == SolveSecsInput.S_IN:
                continue
            # 1.生成model的输入数据
            input_data = []
            wcode_dx = para_dx_to_wcode_dx[para_it]
            is_valid_train, st_dx = get_start_dx(paragraphs[para_it])
            if not is_valid_train:
                continue
            for word_it in range(0, len(wcode_1article.text_c[wcode_dx].vec)):
                input_data.append(wcode_1article.text_c[wcode_dx].vec[word_it] + OUTPUT_DIC_SIZE)  # 如果使用事先准备好的词向量列表，则输入数据需要加上OUTPUT_DIC_SIZE，以避免使用O_IDX等特殊字符
            input_data.append(EOS_IDX)
            # model的输入数据，除了段落内容以外，还包括段落在文章中的位置，文章中是否包含解决区段的起止信号等
            para_ratio = (para_it + 1) / (len(paragraphs) + 1)
            para_ratio = (para_ratio - 0.5) * 4
            para_ratio = torch.FloatTensor([para_ratio])
            has_solve_sec = -1
            for sec_mark in sec_mark_list:
                if sec_mark != SolveSecsInput.S_NO:
                    has_solve_sec = 1
                    break
            has_solve_sec = torch.LongTensor([has_solve_sec])
            if sec_mark_list[para_it] != SolveSecsInput.S_NO:
                solve_sec_mark = 1
            else:
                solve_sec_mark = -1
            solve_sec_mark = torch.LongTensor([solve_sec_mark])
            # 2.生成model的输出数据
            length = torch.LongTensor([len(input_data)])
            input_data = torch.LongTensor(input_data).unsqueeze(dim=0)
            output_data = model.predict(input_data, length, para_ratio, has_solve_sec, solve_sec_mark)
            # 3.处理model的输出数据
            has_solve_msg = False
            start_word_dx = -1
            end_word_dx = len(output_data) - 1
            for it in range(len(output_data) - 1):  # 这里直到len(output_data) - 1，是因为最后一项应该是作为信息末尾标志EOS_IDX，不代表句子中的正式内容
                if output_data[it] in [B_IDX, I_IDX, E_IDX]:
                    has_solve_msg = True
                    if start_word_dx == -1:
                        start_word_dx = it
                if output_data[it] == O_IDX:
                    if has_solve_msg:
                        end_word_dx = it
                        break
            if has_solve_msg:
                start_char_dx = wcode_1article.text_c[wcode_dx].vec_to_text_dx_list[start_word_dx]
                end_char_dx = wcode_1article.text_c[wcode_dx].vec_to_text_dx_list[end_word_dx - 1] + wcode_1article.text_c[wcode_dx].vec_len_list[end_word_dx - 1]
                start_line_dx = line_no_by_para[aid][para_it] - paragraphs[para_it].count("\n") + paragraphs[para_it][:start_char_dx].count("\n")
                end_line_dx = line_no_by_para[aid][para_it] - paragraphs[para_it].count("\n") + paragraphs[para_it][:end_char_dx].count("\n")
                solve_msg_1article.append(paragraphs[para_it][start_char_dx: end_char_dx])
                solve_lines_1article.extend(list(range(start_line_dx, end_line_dx + 1)))
        return solve_msg_1article, list(set(solve_lines_1article))

    def test_1article(self, aid, wcode_1article: WordVec1Article, article, line_no_by_para: Dict[int, List[int]], line_dx_to_para_dx) -> Tuple[list, List[str], List[int]]:
        """
        生成一片文章对应的解决方案的行标和具体内容
        :param aid: 文章ID
        :param wcode_1article: 这一篇文章的词编码
        :param article: 这一篇文章的标题和正文
        :param line_no_by_para: 每篇文章中，每一段对应的尾行的行标
        :param line_dx_to_para_dx: 行标和段落标记之间的关系
        """
        paragraphs = article.text.rstrip('\n').replace('\n\n\n', '\n\n').split('\n\n')
        # 1.生成这篇文章的解决方案的起始和收尾段落
        solve_beg_para_dxs, solve_end_para_dxs = self.test_1article_sec(aid, wcode_1article, article)
        assert len(solve_beg_para_dxs) - len(solve_end_para_dxs) in [0, 1]
        if len(solve_beg_para_dxs) - len(solve_end_para_dxs) == 1:
            solve_end_para_dxs.append("eof")
        solve_secs = list(zip(solve_beg_para_dxs, solve_end_para_dxs))
        # 2.将起始和收尾段落之间的全部内容一律认定为解决方案信息
        solve_lines_1article = []
        solve_msg_1article = []
        for line in line_dx_to_para_dx:
            para_dx = line_dx_to_para_dx[line]
            for sec in solve_secs:
                if para_dx > sec[0] and (sec[1] == "eof" or para_dx < sec[1]):
                    solve_lines_1article.append(line)
        for para_it in range(len(paragraphs)):
            for sec in solve_secs:
                if para_it > sec[0] and (sec[1] == "eof" or para_it < sec[1]):
                    solve_msg_1article.append(paragraphs[para_it])
        # 3.判断解决区段的起止信号之外的内容，是否仍然存在解决方案的信息
        solve_msg_tmp, solve_lines_tmp = self.test_1article_msg(aid, wcode_1article, article, solve_secs, line_no_by_para, self.model_msg_c)
        solve_msg_1article.extend(solve_msg_tmp)
        solve_lines_1article.extend(solve_lines_tmp)
        return solve_secs, solve_msg_1article, solve_lines_1article

    def test(self, articles, line_no_by_para, mark_data):
        """使用测试数据，验证解决方案起始信号和结束信号的识别的准确率"""

        # 1.记录每篇文章的行标和段落标记之间的关系
        line_dx_to_para_dx_all = dict()
        for aid in mark_data:
            line_dx_to_para_dx = dict()
            paragraphs = articles[aid].text.rstrip('\n').replace('\n\n\n', '\n\n').split('\n\n')
            for para_it in range(len(paragraphs)):
                line_nos = range(line_no_by_para[aid][para_it] - paragraphs[para_it].count('\n'), line_no_by_para[aid][para_it] + 1)
                for line_no in line_nos:
                    line_dx_to_para_dx[line_no] = para_it
            line_dx_to_para_dx_all[aid] = line_dx_to_para_dx
        # 2.生成算法对每篇文章的解决方案的判断结果
        all_solve_secs = dict()  # 每篇文章的解决区段的起止信号的位置
        all_solve_msgs = dict()  # 每篇文章的解决问题的信息
        all_solve_lines = dict()  # 每篇文章的解决方案所在的行标
        self.model_sec_beg.eval()
        self.model_sec_end.eval()
        self.model_msg_c.eval()
        with torch.no_grad():
            for aid in mark_data:
                if aid in self.test_aid_list and mark_data[aid].err_msg != str():
                    solve_sec_1article, solve_msg_1article, solve_lines_1article = self.test_1article(aid, self.wv_c.wcode_list[aid], articles[aid], line_no_by_para, line_dx_to_para_dx_all[aid])
                    all_solve_secs[aid] = solve_sec_1article
                    all_solve_msgs[aid] = solve_msg_1article
                    all_solve_lines[aid] = solve_lines_1article
        # 3.验证解决问题的起始和结束段落的判断结果
        solve_sec_score, total_solve_score, sec_score_beg, total_score_beg, sec_score_end, total_score_end = SolveValidation.whole_secs(all_solve_secs, mark_data, line_dx_to_para_dx_all)
        print('solve_sec_score = %.2f / %d' % (solve_sec_score, total_solve_score))
        print('sec_begin_score = %.2f / %d' % (sec_score_beg, total_score_beg))
        print('sec_end_score = %.2f / %d' % (sec_score_end, total_score_end))
        # 4.验证解决问题的行标和内容的判断结果
        solve_line_score, total_solve_score = SolveValidation.line_ratio(all_solve_lines, mark_data)
        solve_msg_score, total_solve_msg_score = SolveValidation.sentence_ratio(all_solve_msgs, mark_data)
        print('solve_line_score = %.2f / %d' % (solve_line_score, total_solve_score))
        print('solve_msg_score = %.2f / %d' % (solve_msg_score, total_solve_msg_score))

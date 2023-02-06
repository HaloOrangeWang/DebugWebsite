from settings import *
from data_io.load_data import MarkData
from .funcs import WordVec1Para, WordVec1Article, has_chn_chr, is_chn_chr, is_chn_eng_number, is_eng_number, has_chn_or_eng_chr
from .word_vec import WvNormal, WvCode
from .model import BilstmClassifyModel, BilstmCRFModel, BilstmCRFExpandModel
from validations.basic import SolveValidation
from typing import List, Dict, Tuple, Optional
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


def judge_code(text) -> bool:
    """
    判断一段文字是否为纯代码。
    如果一段文字中完全没有中文字词，则认定为纯代码。
    如果一段文字中有中文字词，但中文字词的前面包含了 # // ' " 这四个字符，那么我们认为该中文字词是代码中的注释或字符串，同样认定为纯代码。
    :param text: 文本内容
    :return: 是否为纯代码
    """
    for t in range(len(text)):
        if is_chn_chr(text[t]):
            return False
        # 发现了 # // ' " 但还没发现中文字符，则可以直接返回False
        if text[t] in ['#', '\'', '"'] or (t <= len(text) - 2 and text[t] == '/' and text[t + 1] == '/'):
            return True
    return True


class HandledData:
    """这里存放经过一些预处理（如分段）之后的文本数据"""

    class A1:
        """每篇文章经预处理之后的相关内容"""
        def __init__(self):
            self.paragraphs: Optional[List[str]] = None  # 将文章切分成段落之后的列表
            self.num_para: Optional[int] = None  # 文章一共有多少个段落
            self.line_no_by_para: Optional[List[int]] = None  # 切分成段落后，每一段的尾行的行标
            self.para_dx_to_wcode_dx: Optional[Dict[int, int]] = None  # 切分成段落后，每个段落索引值对应的词向量列表的索引值
            self.line_dx_to_para_dx: Optional[Dict[int, int]] = None  # 文本中行标和段落索引值之间的对应关系
            self.has_useful_data: Optional[List[bool]] = None  # 段落中是否包含有效内容。用于排除 空格段落、仅有标点符号的段落，以及仅有 1. a) 等序号的段落
            self.useful_data_st_dx: Optional[List[int]] = None  # 如果段落中包含有效内容。从段落哪个字符开始是有效内容
            self.not_only_code: Optional[List[bool]] = None  # 段落中是否包含除了代码以外的其余内容。判定方法为包含中文字词，且中文字词不是代码注释
            self.para_ratio: Optional[List[float]] = None  # 段落在整篇文章中的比例，在0-1之间（做了平滑处理）。
            self.sentences: Optional[List[List[str]]] = None  # 将文章按照段落和部分中文标点符号（包括'，', '。', '；', '：', '\n'）切分成句子后的列表
            self.sentence_hud: Optional[List[List[bool]]] = None  # 每个句子是否含有有效的内容。排除空句子。
            self.sentence_start_dx: Optional[List[List[int]]] = None  # 每个句子的第一个字（排除起始的 1. a) 序号内容）在段落中的位置。
            self.sentence_end_dx: Optional[List[List[int]]] = None  # 每个句子的最后一个字在段落中的位置。
            self.sentence_ratio: Optional[List[List[float]]] = None  # 句子在整篇文章中的比例，在0-1之间（做了平滑处理）

    def __init__(self):
        self.data_by_article = dict()

    def __getitem__(self, item):
        return self.data_by_article[item]

    def preprocess(self, articles, wcode_list: Dict[int, WordVec1Article], line_no_by_para):
        """
        生成预处理后的相关内容
        :param articles: 文章列表
        :param wcode_list: 每篇文章中，将词进行编码后的结果
        :param line_no_by_para: 每篇文章中，每一段对应的尾行的行标
        """
        for aid in articles:
            data_1a = self.A1()
            # 1.将文章切分成段落
            data_1a.paragraphs = articles[aid].text.rstrip('\n').replace('\n\n\n', '\n\n').split('\n\n')
            data_1a.num_para = len(data_1a.paragraphs)
            # 2.将文章切分成段落后，构建段落索引值和词向量列表的索引值之间的对应关系
            data_1a.line_no_by_para = line_no_by_para[aid]
            data_1a.para_dx_to_wcode_dx = dict()
            for wcode_it in range(len(wcode_list[aid].text_c)):
                data_1a.para_dx_to_wcode_dx[wcode_list[aid].text_c[wcode_it].paragraph_dx] = wcode_it
            # 3.将文章切分成段落后，构建文章中行标和段落索引值之间的对应关系
            data_1a.line_dx_to_para_dx = dict()
            for para_it in range(data_1a.num_para):
                line_nos = range(line_no_by_para[aid][para_it] - data_1a.paragraphs[para_it].count('\n'), line_no_by_para[aid][para_it] + 1)
                for line_no in line_nos:
                    data_1a.line_dx_to_para_dx[line_no] = para_it

            data_1a.has_useful_data = [0 for t0 in range(data_1a.num_para)]
            data_1a.useful_data_st_dx = [0 for t0 in range(data_1a.num_para)]
            data_1a.not_only_code = [0 for t0 in range(data_1a.num_para)]
            data_1a.para_ratio = [0. for t0 in range(data_1a.num_para)]
            data_1a.sentences = [list() for t0 in range(data_1a.num_para)]
            data_1a.sentence_hud = [list() for t0 in range(data_1a.num_para)]
            data_1a.sentence_start_dx = [list() for t0 in range(data_1a.num_para)]
            data_1a.sentence_end_dx = [list() for t0 in range(data_1a.num_para)]
            data_1a.sentence_ratio = [list() for t0 in range(data_1a.num_para)]
            for para_it in range(data_1a.num_para):
                # 4.判断每个段落是否有有效内容
                data_1a.has_useful_data[para_it], data_1a.useful_data_st_dx[para_it] = get_start_dx(data_1a.paragraphs[para_it])
                # 5.判断段落中是否为纯代码
                data_1a.not_only_code[para_it] = (not judge_code(data_1a.paragraphs[para_it]))
                # 6.判断段落在文本中的位置。
                data_1a.para_ratio[para_it] = (para_it + 1) / (data_1a.num_para + 1)
                # 7.将段落拆分成句子
                tmp_start_dx = 0
                for t in range(len(data_1a.paragraphs[para_it]) + 1):
                    if t == len(data_1a.paragraphs[para_it]) or data_1a.paragraphs[para_it][t] in ['，', '。', '；', '：', '\n']:
                        if t > tmp_start_dx:
                            sentence = data_1a.paragraphs[para_it][tmp_start_dx: t]
                            data_1a.sentences[para_it].append(sentence)
                            sentence_hud, sentence_udst = get_start_dx(sentence)
                            data_1a.sentence_hud[para_it].append(sentence_hud)
                            data_1a.sentence_start_dx[para_it].append(tmp_start_dx + sentence_udst)
                            data_1a.sentence_end_dx[para_it].append(t - 1)
                            sentence_ratio = (para_it + 1 + (tmp_start_dx / len(data_1a.paragraphs[para_it]))) / (data_1a.num_para + 1)
                            data_1a.sentence_ratio[para_it].append(sentence_ratio)
                        tmp_start_dx = t + 1

            self.data_by_article[aid] = data_1a


def get_solve_dx_list(handled_articles: HandledData, mark_data: Dict[int, MarkData]) -> Dict[int, List[List[int]]]:
    """
    获取一篇文章中，解决问题的信息所在的位置。返回的方式为[(段落id, 字符在段落中的位置)]。不处理标题，因为文章的解决方案不能在标题中
    :param handled_articles: 经过一些初步处理后的各篇文章的内容，包括段落、句子，以及一些对应关系
    :param mark_data: 文章标注的情况
    :return:
    """
    solve_dx_list_all = dict()
    for aid in mark_data:
        if not mark_data[aid].err_msg:
            continue
        solve_dx_list_1a = [list() for t in range(handled_articles[aid].num_para)]  # 每一段的解决问题信息的列表
        for solve_it in range(len(mark_data[aid].solves)):
            # 对于每一条解决信息，找到它们所在的段落。
            line = mark_data[aid].solve_lines[solve_it]
            para_dx = handled_articles[aid].line_dx_to_para_dx[line]
            # 这一段落应当包含该条解决问题的信息，将它记录下来
            dx = handled_articles[aid].paragraphs[para_dx].index(mark_data[aid].solves[solve_it])
            solve_dx_list_1a[para_dx] = list(range(dx, dx + len(mark_data[aid].solves[solve_it])))
        solve_dx_list_all[aid] = solve_dx_list_1a
    return solve_dx_list_all


class SolveSecsInput:
    """这个类用于生成对解决问题的区间的训练数据"""
    # 一些enum值。分别表示"与解决方案无关"、"解决方案的起始"、"解决方案的中间"、"解决方案的终止信号"
    S_NO = 0
    S_BEG = 1
    S_IN = 2
    S_END = 3
    S_RV_BEG = 4  # 反起始信号

    def __init__(self, wdict: dict, rv_wdict: list):
        self.wdict = wdict
        self.rv_wdict = rv_wdict
        self.all_sec_marks: Dict[int, List[List[int]]] = dict()  # 每篇文章每个段落每个句子是否包含解决方案的起始信号和终止信号的标注信息。数值为上面的S_XXX的enum值
        self.beg_train_input = []  # 训练“哪些段落是解决方案的起始”的模型输入数据
        self.beg_train_output = []  # 训练“哪些段落是解决方案的起始”的模型输出数据
        self.beg_train_lens = []  # 训练“哪些段落是解决方案的起始”的输入数据长度
        self.beg_train_sentence_ratio = []  # 训练“哪些段落是解决方案的起始”时，所挑选的段落在整个文章中的百分比（计算方法为 (该段落id+1)/文章段落数+1）
        self.end_train_input = []  # 训练“哪些段落是解决方案的终止”的模型输入数据
        self.end_train_output = []  # 训练“哪些段落是解决方案的终止”的模型输出数据
        self.end_train_lens = []  # 训练“哪些段落是解决方案的终止”的输入数据长度
        self.end_train_sentence_ratio = []  # 训练“哪些段落是解决方案的终止”时，所挑选的段落在整个文章中的百分比
        self.rv_train_input = []  # 训练“哪些段落是解决方案的反起始信号”的模型输入数据
        self.rv_train_output = []  # 训练“哪些段落是解决方案的反起始信号”的模型输出数据
        self.rv_train_lens = []  # 训练“哪些段落是解决方案的反起始信号”的输入数据长度
        self.rv_train_sentence_ratio = []  # 训练“哪些段落是解决方案的反起始信号”时，所挑选的段落在整个文章中的百分比

    # noinspection PyMethodMayBeStatic
    def get_marked_sec_loc(self, data_1a: HandledData.A1, marked_secs: list) -> List[List[Optional[Tuple[int, int]]]]:
        """
        获取标注好的各个解决起止信号，分别对应文章中的哪个段落和哪个句子
        :param data_1a: 这篇文章标注好的信息，包括段落、句子，以及一些对应关系
        :param marked_secs: 解决方案的标注信息
        :return: 被标注的段落在文章中的位置
        """
        sec_mark_output: List[List[Optional[Tuple[int, int]]]] = [[None, None] for t in range(len(marked_secs))]
        for t in range(len(marked_secs)):
            if marked_secs[t][0] != "nan":
                # 如果第一项不为nan，说明该信号是一对正常的起止信号
                st_para_dx = data_1a.line_dx_to_para_dx[marked_secs[t][0][0]]
                for sentence_it in range(len(data_1a.sentences[st_para_dx])):
                    if marked_secs[t][0][1] in data_1a.sentences[st_para_dx][sentence_it]:
                        st_sentence_dx = sentence_it
                        break
                sec_mark_output[t][0] = (st_para_dx, st_sentence_dx)
                if marked_secs[t][1] != "eof":
                    ed_para_dx = data_1a.line_dx_to_para_dx[marked_secs[t][1][0]]
                    for sentence_it in range(len(data_1a.sentences[ed_para_dx])):
                        if marked_secs[t][1][1] in data_1a.sentences[ed_para_dx][sentence_it]:
                            ed_sentence_dx = sentence_it
                            break
                    sec_mark_output[t][1] = (ed_para_dx, ed_sentence_dx)
            else:
                # 如果第一项为nan，说明该信号是一个反起始信号
                rv_para_dx = data_1a.line_dx_to_para_dx[marked_secs[t][1][0]]
                for sentence_it in range(len(data_1a.sentences[rv_para_dx])):
                    if marked_secs[t][1][1] in data_1a.sentences[rv_para_dx][sentence_it]:
                        rv_sentence_dx = sentence_it
                        break
                sec_mark_output[t][1] = (rv_para_dx, rv_sentence_dx)
        return sec_mark_output

    def get_sec_mark_by_sentence(self, data_1a: HandledData.A1, marked_secs: list) -> List[List[int]]:
        """
        根据文章内容，获取哪些句子是解决方案的起始，哪些句子是解决方案的终止，哪些句子是反起始信号
        :param data_1a: 这篇文章标注好的信息，包括段落、句子，以及一些对应关系
        :param marked_secs: 解决方案的标注信息
        :return: 每个段落是否包含解决方案的起始信号和终止信号的标注信息
        """
        sec_mark_lists = [list() for t in range(data_1a.num_para)]  # 标注每个段落中，每个句子是否涵盖解决方案的起止信号
        sec_mark_loc = self.get_marked_sec_loc(data_1a, marked_secs)
        # 如果这篇文章中没有解决方案的段落标注，则所有段落都可认为“不是解决方案的起始段落”
        if len(marked_secs) == 0:
            for para_it in range(data_1a.num_para):
                sec_mark_lists[para_it] = [self.S_NO for t in range(len(data_1a.sentences[para_it]))]
            return sec_mark_lists
        # 针对每一段中的每一个句子，判断是否为解决方案的起始/终止信号
        solve_sec_dx = 0
        for para_it in range(data_1a.num_para):
            for sentence_it in range(len(data_1a.sentences[para_it])):
                if sec_mark_loc[solve_sec_dx][0] is not None:
                    # 解决方案的第一项不为None，说明是一对正常的起止信号
                    if para_it < sec_mark_loc[solve_sec_dx][0][0] or (para_it == sec_mark_loc[solve_sec_dx][0][0] and sentence_it < sec_mark_loc[solve_sec_dx][0][1]):
                        sec_mark_lists[para_it].append(self.S_NO)
                    elif para_it == sec_mark_loc[solve_sec_dx][0][0] and sentence_it == sec_mark_loc[solve_sec_dx][0][1]:
                        sec_mark_lists[para_it].append(self.S_BEG)
                    elif sec_mark_loc[solve_sec_dx][1] is None or para_it < sec_mark_loc[solve_sec_dx][1][0] or (para_it == sec_mark_loc[solve_sec_dx][1][0] and sentence_it < sec_mark_loc[solve_sec_dx][1][1]):
                        sec_mark_lists[para_it].append(self.S_IN)
                    elif para_it == sec_mark_loc[solve_sec_dx][1][0] and sentence_it == sec_mark_loc[solve_sec_dx][1][1]:
                        sec_mark_lists[para_it].append(self.S_END)
                        if solve_sec_dx < len(sec_mark_loc) - 1:
                            solve_sec_dx += 1
                    else:
                        sec_mark_lists[para_it].append(self.S_NO)
                else:
                    # 解决方案的第一项为None，说明是一个反起始信号
                    if para_it < sec_mark_loc[solve_sec_dx][1][0] or (para_it == sec_mark_loc[solve_sec_dx][1][0] and sentence_it < sec_mark_loc[solve_sec_dx][1][1]):
                        sec_mark_lists[para_it].append(self.S_NO)
                    elif para_it == sec_mark_loc[solve_sec_dx][1][0] and sentence_it == sec_mark_loc[solve_sec_dx][1][1]:
                        sec_mark_lists[para_it].append(self.S_RV_BEG)
                        if solve_sec_dx < len(sec_mark_loc) - 1:
                            solve_sec_dx += 1
                    else:
                        sec_mark_lists[para_it].append(self.S_NO)
        return sec_mark_lists

    def get_data_1article(self, data_1a: HandledData.A1, wcode_list: List[WordVec1Para], sec_mark_list: List[List[int]]):
        """
        根据文章内容，生成训练数据
        :param data_1a: 这篇文章标注好的信息，包括段落、句子，以及一些对应关系
        :param wcode_list: 将文本编码之后的内容
        :param sec_mark_list: 每个段落的每个句子是否包含解决方案的起始信号和终止信号的标注信息
        """
        for para_it in range(data_1a.num_para):
            # 如果这段文字没有对应的词编码，或者没有有用的信息，或者为纯代码的话，则不作为解决问题信息的训练内容
            if para_it not in data_1a.para_dx_to_wcode_dx or (not data_1a.has_useful_data[para_it]) or (not data_1a.not_only_code[para_it]):
                continue
            wcode_dx = data_1a.para_dx_to_wcode_dx[para_it]
            for sentence_it in range(len(data_1a.sentences[para_it])):
                if (not data_1a.sentence_hud[para_it][sentence_it]) or (not has_chn_chr(data_1a.sentences[para_it][sentence_it])):
                    continue
                # 将句子在文章中的比例，扩展到（-2~2）区间段内
                sentence_ratio = (data_1a.sentence_ratio[para_it][sentence_it] - 0.5) * 4
                # 针对这句话生成训练样本
                input_data = []
                for word_it in range(0, len(wcode_list[wcode_dx].vec)):
                    if wcode_list[wcode_dx].vec_to_text_dx_list[word_it] + wcode_list[wcode_dx].vec_len_list[word_it] - 1 < data_1a.sentence_start_dx[para_it][sentence_it]:
                        continue
                    input_data.append(wcode_list[wcode_dx].vec[word_it] + OUTPUT_DIC_SIZE)  # 输入数据需要加上OUTPUT_DIC_SIZE，以避免使用O_IDX等特殊字符
                    # 如果到达了段落的末尾，或句子的末尾，则将新的训练样本插入到数组中
                    if word_it == len(wcode_list[wcode_dx].vec) - 1 or wcode_list[wcode_dx].vec_to_text_dx_list[word_it + 1] > data_1a.sentence_end_dx[para_it][sentence_it]:
                        if len(input_data) == 0 or min(input_data) < len(self.rv_wdict) + OUTPUT_DIC_SIZE:  # 完全由标点符号、英文内容组成的句子，不能作为训练数据
                            if sec_mark_list[para_it][sentence_it] == self.S_NO:
                                self.beg_train_input.append(copy.copy(input_data))
                                self.beg_train_output.append(FALSE_IDX)
                                self.beg_train_lens.append(len(input_data))
                                self.beg_train_sentence_ratio.append(sentence_ratio)
                                self.rv_train_input.append(copy.copy(input_data))
                                self.rv_train_output.append(FALSE_IDX)
                                self.rv_train_lens.append(len(input_data))
                                self.rv_train_sentence_ratio.append(sentence_ratio)
                            elif sec_mark_list[para_it][sentence_it] == self.S_BEG:
                                self.beg_train_input.append(copy.copy(input_data))
                                self.beg_train_output.append(TRUE_IDX)
                                self.beg_train_lens.append(len(input_data))
                                self.beg_train_sentence_ratio.append(sentence_ratio)
                                # 起始信号的判断优先级高于反起始信号，所以这里只插入beg_train就可以了
                            elif sec_mark_list[para_it][sentence_it] == self.S_IN:
                                self.end_train_input.append(copy.copy(input_data))
                                self.end_train_output.append(FALSE_IDX)
                                self.end_train_lens.append(len(input_data))
                                self.end_train_sentence_ratio.append(sentence_ratio)
                            elif sec_mark_list[para_it][sentence_it] == self.S_END:
                                self.end_train_input.append(copy.copy(input_data))
                                self.end_train_output.append(TRUE_IDX)
                                self.end_train_lens.append(len(input_data))
                                self.end_train_sentence_ratio.append(sentence_ratio)
                            elif sec_mark_list[para_it][sentence_it] == self.S_RV_BEG:
                                self.beg_train_input.append(copy.copy(input_data))
                                self.beg_train_output.append(FALSE_IDX)
                                self.beg_train_lens.append(len(input_data))
                                self.beg_train_sentence_ratio.append(sentence_ratio)
                                self.rv_train_input.append(copy.copy(input_data))
                                self.rv_train_output.append(TRUE_IDX)
                                self.rv_train_lens.append(len(input_data))
                                self.rv_train_sentence_ratio.append(sentence_ratio)
                        break

    def get_model_data(self, handled_articles: HandledData, wcode_list: Dict[int, WordVec1Article], mark_data: Dict[int, MarkData], test_aid_list):
        """
        根据所有标注好的文章，生成错误信息的训练数据和测试数据
        :param handled_articles: 经过一些初步处理后的各篇文章的内容，包括段落、句子，以及一些对应关系
        :param wcode_list: 每篇文章中，将词进行编码后的结果
        :param mark_data: 文章标注的情况
        :param test_aid_list: 哪些文章是用于测试的
        """
        for aid in mark_data:
            if not mark_data[aid].err_msg:
                continue
            self.all_sec_marks[aid] = self.get_sec_mark_by_sentence(handled_articles[aid], mark_data[aid].solve_secs)
            if aid in test_aid_list:
                continue
            self.get_data_1article(handled_articles[aid], wcode_list[aid].text_c, self.all_sec_marks[aid])


class SolveInSecInput:
    """生成如何判断段落内的内容（段落在起止信号之间，或刚刚处于反起始信号之前）是否为解决问题的信息的训练数据"""

    def __init__(self, wdict: dict, rv_wdict: list):
        self.wdict = wdict
        self.rv_wdict = rv_wdict
        self.train_input = []  # 训练用的输入数据。
        self.train_output = []  # 训练用的输出数据。
        self.train_lens = []  # 训练用的数据的长度
        self.sentence_ratio = []  # 训练解决方案时，所挑选的句子/段落在整个文章中的百分比，并线性扩展至[-2, 2]区间内

    def get_data_1article(self, data_1a: HandledData.A1, wcode_list: List[WordVec1Para], sec_mark_list: List[List[int]], solve_dx_lists: List[List[int]]):
        # 从后往前遍历，因为如果按照反起始信号判断，它是看之前的段落。
        just_before_rv_seg = False
        for para_it in range(data_1a.num_para - 1, -1, -1):
            # 1.判断信号属于一个句子还是属于一个段落。如果这一段的内容完全为S_IN（说明它完全包在起始信号和结束信号中），或反起始信号位于下x段的第一个句子，则根据整个段落判断
            # mode为生成训练数据的模式。0表示不是训练数据；1表示作为训练数据，但按照句子插入；2表示作为训练数据，且整体性插入
            all_in_sec_mark = True
            beg_mark_dx = -1
            end_mark_dx = -1
            rv_mark_dx = -1
            for sentence_it in range(len(sec_mark_list[para_it])):
                if sec_mark_list[para_it][sentence_it] != SolveSecsInput.S_IN:
                    all_in_sec_mark = False
                if sec_mark_list[para_it][sentence_it] == SolveSecsInput.S_BEG:
                    beg_mark_dx = sentence_it
                if sec_mark_list[para_it][sentence_it] == SolveSecsInput.S_END:
                    end_mark_dx = sentence_it
                if sec_mark_list[para_it][sentence_it] == SolveSecsInput.S_RV_BEG:
                    rv_mark_dx = sentence_it
            if just_before_rv_seg or all_in_sec_mark:
                mode = 2
            elif (beg_mark_dx != -1 and end_mark_dx != -1 and end_mark_dx > beg_mark_dx) or (rv_mark_dx >= 1):
                mode = 1
            else:
                mode = 0
            # 2.如果这段文字没有对应的词编码，或者没有有用的信息，或者为纯代码的话，则不作为解决问题信息的训练内容
            if para_it not in data_1a.para_dx_to_wcode_dx or (not data_1a.has_useful_data[para_it]) or (not data_1a.not_only_code[para_it]):
                pass
            else:
                wcode_dx = data_1a.para_dx_to_wcode_dx[para_it]
                if mode == 2:
                    # 这一段作为训练数据，且整体性插入
                    # 将段落在文章中的比例，扩展到（-2~2）区间段内
                    para_ratio = (data_1a.para_ratio[para_it] - 0.5) * 4
                    # 生成输入数据
                    input_data = []
                    for word_it in range(0, len(wcode_list[wcode_dx].vec)):
                        if wcode_list[wcode_dx].vec_to_text_dx_list[word_it] + wcode_list[wcode_dx].vec_len_list[word_it] - 1 < data_1a.useful_data_st_dx[para_it]:
                            continue
                        input_data.append(wcode_list[wcode_dx].vec[word_it] + OUTPUT_DIC_SIZE)  # 输入数据需要加上OUTPUT_DIC_SIZE，以避免使用O_IDX等特殊字符
                    # 生成输出数据。如果这一段内容确实包含解决方案的信息，则预期输出为TRUE_IDX，否则预期输出为FALSE_IDX
                    output_data = FALSE_IDX
                    if len(solve_dx_lists[para_it]) != 0:
                        output_data = TRUE_IDX
                    if len(input_data) == 0 or min(input_data) < len(self.rv_wdict) + OUTPUT_DIC_SIZE:  # 完全由标点符号、英文内容组成的句子，不能作为训练数据
                        self.train_input.append(copy.copy(input_data))
                        self.train_output.append(output_data)
                        self.train_lens.append(len(input_data))
                        self.sentence_ratio.append(para_ratio)
                elif mode == 1:
                    # 作为训练数据，但按照句子插入
                    just_before_rv_seg_sentence = False
                    for sentence_it in range(len(data_1a.sentences[para_it]) - 1, -1, -1):
                        # 生成输出数据。如果这一句话的内容中确实包含解决方案的信息，则预期输出为TRUE_IDX，否则预期输出为FALSE_IDX
                        output_data = FALSE_IDX
                        for t in solve_dx_lists[para_it]:
                            if data_1a.sentence_start_dx[para_it][sentence_it] <= t <= data_1a.sentence_end_dx[para_it][sentence_it]:
                                output_data = TRUE_IDX
                                break
                        if (not data_1a.sentence_hud[para_it][sentence_it]) or (not has_chn_chr(data_1a.sentences[para_it][sentence_it])):
                            pass
                        else:
                            # 将句子在文章中的比例，扩展到（-2~2）区间段内
                            sentence_ratio = (data_1a.sentence_ratio[para_it][sentence_it] - 0.5) * 4
                            # 只有起始信号和终止信号之间的句子，以及反起始信号之前的句子，才能够作为训练内容
                            if (beg_mark_dx != -1 and end_mark_dx != -1 and beg_mark_dx < sentence_it < end_mark_dx) or (end_mark_dx < sentence_it < rv_mark_dx and just_before_rv_seg_sentence):
                                # 生成输入数据
                                input_data = []
                                for word_it in range(0, len(wcode_list[wcode_dx].vec)):
                                    if wcode_list[wcode_dx].vec_to_text_dx_list[word_it] + wcode_list[wcode_dx].vec_len_list[word_it] - 1 < data_1a.sentence_start_dx[para_it][sentence_it]:
                                        continue
                                    input_data.append(wcode_list[wcode_dx].vec[word_it] + OUTPUT_DIC_SIZE)  # 输入数据需要加上OUTPUT_DIC_SIZE，以避免使用O_IDX等特殊字符
                                    if word_it == len(wcode_list[wcode_dx].vec) - 1 or wcode_list[wcode_dx].vec_to_text_dx_list[word_it + 1] > data_1a.sentence_end_dx[para_it][sentence_it]:
                                        break
                                if len(input_data) == 0 or min(input_data) < len(self.rv_wdict) + OUTPUT_DIC_SIZE:  # 完全由标点符号、英文内容组成的句子，不能作为训练数据
                                    self.train_input.append(copy.copy(input_data))
                                    self.train_output.append(output_data)
                                    self.train_lens.append(len(input_data))
                                    self.sentence_ratio.append(sentence_ratio)
                        if sentence_it == rv_mark_dx:
                            just_before_rv_seg_sentence = True
                        elif just_before_rv_seg_sentence and output_data == FALSE_IDX:
                            just_before_rv_seg_sentence = False
            # 收尾处理
            if rv_mark_dx == 0:
                just_before_rv_seg = True
            elif just_before_rv_seg and len(solve_dx_lists[para_it]) == 0:
                just_before_rv_seg = False

    def get_model_data(self, handled_articles: HandledData, wcode_list: Dict[int, WordVec1Article], mark_data: Dict[int, MarkData], all_sec_marks: Dict[int, List[List[int]]], all_solve_dx_list: Dict[int, List[List[int]]], test_aid_list):
        """
        根据所有标注好的文章，生成错误信息的训练数据和测试数据
        :param handled_articles: 经过一些初步处理后的各篇文章的内容，包括段落、句子，以及一些对应关系
        :param wcode_list: 每篇文章中，将词进行编码后的结果
        :param mark_data: 文章标注的情况
        :param all_sec_marks: 每篇文章每个段落是否包含解决方案的起始信号和终止信号的标注信息。数值为上面的S_XXX的enum值
        :param all_solve_dx_list: 每篇文章的解决问题的信息在文章中的位置
        :param test_aid_list: 哪些文章是用于测试的
        """
        for aid in mark_data:
            if not mark_data[aid].err_msg:
                continue
            if aid in test_aid_list:
                continue
            self.get_data_1article(handled_articles[aid], wcode_list[aid].text_c, all_sec_marks[aid], all_solve_dx_list[aid])


class SolveNotInSecInput:
    """生成如何判断段落内的内容（段落不在起止信号之间，也不是刚刚处于反起始信号之前）是否为解决问题的信息的训练数据"""

    def __init__(self, wdict, rv_wdict):
        self.wdict = wdict
        self.rv_wdict = rv_wdict
        self.train_input = []  # 训练用的输入数据。
        self.train_output = []  # 训练用的输出数据。
        self.train_lens = []  # 训练用的数据的长度
        self.sentence_ratio = []  # 训练解决方案时，所挑选的句子/段落在整个文章中的百分比，并线性扩展至[-2, 2]区间内
        self.has_solve_sec = []  # 该段落所在的文章是否已存在解决方案的区段。-1表示没有解决方案的区段，1表示有解决方案的区段
        self.solve_sec_mark = []  # 对于已有解决方案区段的文章，当前的句子/段落是否位于解决方案区段的起止位置（包括起止信号和反起始信号所在的句子/段）。-1表示不为起止位置，1表示为起止位置

    def get_data_1article(self, data_1a: HandledData.A1, wcode_list: List[WordVec1Para], sec_mark_list: List[List[int]], solve_dx_lists: List[List[int]]):
        """
        根据文章内容，生成训练数据
        :param data_1a: 这篇文章标注好的信息，包括段落、句子，以及一些对应关系
        :param wcode_list: 将文本编码之后的内容
        :param sec_mark_list: 每个段落是否包含解决方案的起始信号和终止信号的标注信息
        :param solve_dx_lists: 解决问题的信息在文章中的位置
        """
        # 1.获取每一段话中，对应的解决方案和文本编码结果之间的对应关系
        is_solve_by_word = [list() for t in range(len(solve_dx_lists))]  # 每一段的解决问题信息的列表
        for para_it in range(data_1a.num_para):
            if para_it not in data_1a.para_dx_to_wcode_dx:
                continue
            wcode_dx = data_1a.para_dx_to_wcode_dx[para_it]
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
        # 从后往前遍历，因为如果按照反起始信号判断，它是看之前的段落。
        just_before_rv_seg = False
        for para_it in range(data_1a.num_para - 1, -1, -1):
            # 2.1 如果这段文字没有对应的词编码，或者没有有用的信息，或者为纯代码的话，则不作为解决问题信息的训练内容
            if para_it not in data_1a.para_dx_to_wcode_dx or (not data_1a.has_useful_data[para_it]) or (not data_1a.not_only_code[para_it]):
                continue
            wcode_dx = data_1a.para_dx_to_wcode_dx[para_it]
            # 2.2 判断这段文字是作为一个整体插入到训练样本中，还是分句插入到训练样本中
            # mode为生成训练数据的模式。0表示不是训练数据；1表示作为训练数据，但按照句子插入；2表示作为训练数据，且整体性插入
            all_in_sec_mark = True
            beg_mark_dx = -1
            end_mark_dx = -1
            rv_mark_dx = -1
            for sentence_it in range(len(sec_mark_list[para_it])):
                if sec_mark_list[para_it][sentence_it] != SolveSecsInput.S_IN:
                    all_in_sec_mark = False
                if sec_mark_list[para_it][sentence_it] == SolveSecsInput.S_BEG:
                    beg_mark_dx = sentence_it
                if sec_mark_list[para_it][sentence_it] == SolveSecsInput.S_END:
                    end_mark_dx = sentence_it
                if sec_mark_list[para_it][sentence_it] == SolveSecsInput.S_RV_BEG:
                    rv_mark_dx = sentence_it
            if just_before_rv_seg or all_in_sec_mark:
                mode = 0  # 位于起止信号之间，或刚刚在反起始信号之前的段落，不在这里训练。它应当在SolveInSecInput中
            elif (beg_mark_dx != -1 and end_mark_dx != -1 and end_mark_dx > beg_mark_dx) or (rv_mark_dx >= 1):
                mode = 1  # 这段信号同时有起止信号，或反起始信号不处于第一个句子，说明区段是按句子给出的，应当分句给出训练内容
            else:
                mode = 2
            # 2.3 生成正式的训练内容
            if mode == 2:
                # 将段落在文章中的比例，扩展到（-2~2）区间段内
                para_ratio = (data_1a.para_ratio[para_it] - 0.5) * 4
                has_solve_sec = -1
                for t in sec_mark_list:
                    for t0 in t:
                        if t0 != SolveSecsInput.S_NO:
                            has_solve_sec = 1
                            break
                    if has_solve_sec == 1:
                        break
                solve_sec_mark = -1
                for sec_mark in sec_mark_list[para_it]:
                    if sec_mark != SolveSecsInput.S_NO:
                        solve_sec_mark = 1
                        break
                input_data = []  # copy.deepcopy(wcode.vec)
                output_data = []
                # 生成这段话的输入信息和输出信息
                is_first_word = True
                for word_it in range(0, len(wcode_list[wcode_dx].vec)):
                    # 对于一段解决问题的信息，输出的数据应当为 [起始标志B_IDX、解决方案过程中的标志I_IDX、结束标志E_IDX]
                    # 例如：对于 [进程   被    占用   ，     关掉   其他   进程   即可]，输出的内容应当为
                    #          [O_IDX O_IDX O_IDX O_IDX B_IDX I_IDX E_IDX O_IDX]
                    if wcode_list[wcode_dx].vec_to_text_dx_list[word_it] + wcode_list[wcode_dx].vec_len_list[word_it] - 1 < data_1a.useful_data_st_dx[para_it]:
                        continue
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
                        if len(input_data) == 0 or min(input_data) < len(self.rv_wdict) + OUTPUT_DIC_SIZE:  # 完全由标点符号、英文内容组成的句子，不能作为训练数据
                            input_data.append(EOS_IDX)
                            output_data.append(EOS_IDX)
                            self.train_input.append(copy.copy(input_data))
                            self.train_output.append(copy.copy(output_data))
                            self.train_lens.append(len(input_data))
                            self.sentence_ratio.append(para_ratio)
                            self.has_solve_sec.append(has_solve_sec)
                            self.solve_sec_mark.append(solve_sec_mark)
                        input_data = []
                        output_data = []
            elif mode == 1:
                # 作为训练数据，但按照句子插入
                for sentence_it in range(len(data_1a.sentences[para_it])):
                    if not data_1a.sentence_hud[para_it][sentence_it]:
                        continue
                    # 只有起始信号和终止信号之外的句子，以及反起始信号之后的句子（均含边界），才能够作为训练内容
                    if (beg_mark_dx != -1 and end_mark_dx != -1 and (sentence_it <= beg_mark_dx or sentence_it >= end_mark_dx)) or (rv_mark_dx != -1 and sentence_it >= rv_mark_dx and (beg_mark_dx == -1 or sentence_it < beg_mark_dx)):
                        # 将句子在文章中的比例，扩展到（-2~2）区间段内
                        sentence_ratio = (data_1a.sentence_ratio[para_it][sentence_it] - 0.5) * 4
                        has_solve_sec = -1
                        for t in sec_mark_list:
                            for t0 in t:
                                if t0 != SolveSecsInput.S_NO:
                                    has_solve_sec = 1
                                    break
                            if has_solve_sec == 1:
                                break
                        if sec_mark_list[para_it][sentence_it] != SolveSecsInput.S_NO:
                            solve_sec_mark = 1
                        else:
                            solve_sec_mark = -1
                        input_data = []  # copy.deepcopy(wcode.vec)
                        output_data = []
                        # 生成这段话的输入信息和输出信息
                        is_first_word = True
                        for word_it in range(0, len(wcode_list[wcode_dx].vec)):
                            # 对于一段解决问题的信息，输出的数据应当为 [起始标志B_IDX、解决方案过程中的标志I_IDX、结束标志E_IDX]
                            # 例如：对于 [进程   被    占用   ，     关掉   其他   进程   即可]，输出的内容应当为
                            #          [O_IDX O_IDX O_IDX O_IDX B_IDX I_IDX E_IDX O_IDX]
                            if wcode_list[wcode_dx].vec_to_text_dx_list[word_it] + wcode_list[wcode_dx].vec_len_list[word_it] - 1 < data_1a.sentence_start_dx[para_it][sentence_it]:
                                continue
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
                            # 如果到达了段落的末尾，或句子的末尾，则将新的训练样本插入到数组中
                            if word_it == len(wcode_list[wcode_dx].vec) - 1 or wcode_list[wcode_dx].vec_to_text_dx_list[word_it + 1] > data_1a.sentence_end_dx[para_it][sentence_it]:
                                if len(input_data) == 0 or min(input_data) < len(self.rv_wdict) + OUTPUT_DIC_SIZE:  # 完全由标点符号、英文内容组成的句子，不能作为训练数据
                                    input_data.append(EOS_IDX)
                                    output_data.append(EOS_IDX)
                                    self.train_input.append(copy.copy(input_data))
                                    self.train_output.append(copy.copy(output_data))
                                    self.train_lens.append(len(input_data))
                                    self.sentence_ratio.append(sentence_ratio)
                                    self.has_solve_sec.append(has_solve_sec)
                                    self.solve_sec_mark.append(solve_sec_mark)
                                break
            # 收尾处理
            if rv_mark_dx == 0:
                just_before_rv_seg = True
            elif just_before_rv_seg and len(solve_dx_lists[para_it]) == 0:
                just_before_rv_seg = False

    def get_model_data(self, handled_articles: HandledData, wcode_list: Dict[int, WordVec1Article], mark_data: Dict[int, MarkData], all_sec_marks: Dict[int, List[List[int]]], all_solve_dx_list: Dict[int, List[List[int]]], test_aid_list):
        """
        根据所有标注好的文章，生成错误信息的训练数据和测试数据
        :param handled_articles: 经过一些初步处理后的各篇文章的内容，包括段落、句子，以及一些对应关系
        :param wcode_list: 每篇文章中，将词进行编码后的结果
        :param mark_data: 文章标注的情况
        :param all_sec_marks: 每篇文章每个段落是否包含解决方案的起始信号和终止信号的标注信息。数值为上面的S_XXX的enum值
        :param all_solve_dx_list: 每篇文章的解决问题的信息在文章中的位置
        :param test_aid_list: 哪些文章是用于测试的
        """
        for aid in mark_data:
            if not mark_data[aid].err_msg:
                continue
            if aid in test_aid_list:
                continue
            self.get_data_1article(handled_articles[aid], wcode_list[aid].text_c, all_sec_marks[aid], all_solve_dx_list[aid])


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
        self.msg_in_data_c = SolveInSecInput(self.wv_c.wv.key_to_index, self.wv_c.wv.index_to_key)  # 判断起止信号之间的内容是否为解决方案的信息时，用的数据
        self.msg_out_data_c = SolveNotInSecInput(self.wv_c.wv.key_to_index, self.wv_c.wv.index_to_key)  # 判断起止信号之外的内容是否为解决方案的信息时，用的数据
        # self.msg_data_n = SolveInParaInput(self.wv_n.wv.key_to_index, self.wv_n.wv.index_to_key)  # 训练段落内解决问题的信息（英文不编码为<code>）时，用的数据
        self.config_c = SolveClassifyConfigC()
        self.msg_config_c = SolveMsgConfigC()
        # self.msg_config_n = SolveMsgConfigN()
        self.model_sec_beg = BilstmClassifyModel(self.config_c, self.wv_c.emb_mat, freeze=True)  # 训练解决方案起始信号的模型
        self.model_sec_end = BilstmClassifyModel(self.config_c, self.wv_c.emb_mat, freeze=True)  # 训练解决方案终止信号的模型
        self.model_sec_rv = BilstmClassifyModel(self.config_c, self.wv_c.emb_mat, freeze=True)  # 训练解决方案反起始信号的模型
        self.model_msg_in_c = BilstmClassifyModel(self.config_c, self.wv_c.emb_mat, freeze=True)  # 训练解决方案终止信号的模型
        self.model_msg_out_c = BilstmCRFExpandModel(self.msg_config_c, self.wv_c.emb_mat, freeze=True)  # 训练起止信号之外的内容是否为解决方案的信息的模型
        # self.model_msg_n = BilstmCRFExpandModel(self.msg_config_n, self.wv_n.emb_mat, freeze=False)  # 训练段落内的解决方案信息（英文不编码为<code>）的模型
        self.test_aid_list = []
        self.handled_data = HandledData()  # 经过分段等处理之后的内容
        self.all_solve_dx_list = dict()  # 解决方案在各篇文章中的位置

    def setup(self, articles, mark_data, line_no_by_para, punc_code_set):
        """准备模型的输入数据，以及相关配置等"""
        # 设置哪些数据用于训练，而哪些数据用于测试
        aid_list = [t for t in mark_data]
        aid_list = sorted(aid_list)
        self.test_aid_list = aid_list[int(TEST_DATA_START * len(aid_list)): int((TEST_DATA_START + TEST_DATA_RATIO) * len(aid_list))]
        # 对各篇文章的内容进行预处理
        self.handled_data.preprocess(articles, self.wv_c.wcode_list, line_no_by_para)
        self.all_solve_dx_list = get_solve_dx_list(self.handled_data, mark_data)
        # 准备训练数据。包括解决方案区段的训练数据
        self.sec_data_c.get_model_data(self.handled_data, self.wv_c.wcode_list, mark_data, self.test_aid_list)
        self.msg_in_data_c.get_model_data(self.handled_data, self.wv_c.wcode_list, mark_data, self.sec_data_c.all_sec_marks, self.all_solve_dx_list, self.test_aid_list)
        self.msg_out_data_c.get_model_data(self.handled_data, self.wv_c.wcode_list, mark_data, self.sec_data_c.all_sec_marks, self.all_solve_dx_list, self.test_aid_list)
        # self.msg_data_n.get_model_data(articles, self.wv_n.wcode_list, mark_data, line_no_by_para, self.sec_data_c.all_sec_marks, punc_code_set, self.test_aid_list)

    # noinspection PyMethodMayBeStatic
    def sec_batch_preprocess(self, train_input: List, train_output: List, train_lens: List, train_sentence_ratios: List, indexs, start_dx, end_dx):
        """
        对batch数据进行预处理：获取长度排名在start_dx~end_dx之间的数据，并通过填充PAD_IDX的方式，将数组中各项的长度一致
        :param train_input: 模型的训练输入数据列表
        :param train_output: 模型的训练输出数据列表
        :param train_lens: 模型的每组训练数据的长度的列表
        :param train_sentence_ratios: 每组训练数据的句子或段落，在整篇文章中的位置
        :param indexs: 将训练数据根据长度从大到小排序后的索引值
        :param start_dx: 这个batch的起始位置
        :param end_dx: 这个batch的终止位置
        :return: 处理好的训练数据、训练预期输出、各组数据的长度
        """
        input_data = []
        output_data = []
        lens = []
        sentence_ratios = []
        # 1.对报错信息按照长短重新排序
        sorted_index = np.argsort(np.array(train_lens)[indexs[start_dx: (end_dx + 1)]] * (-1))
        max_len = train_lens[indexs[start_dx + sorted_index[0]]]
        input_data.append(train_input[indexs[start_dx + sorted_index[0]]])
        output_data.append(train_output[indexs[start_dx + sorted_index[0]]])
        lens.append(train_lens[indexs[start_dx + sorted_index[0]]])
        sentence_ratios.append(train_sentence_ratios[indexs[start_dx + sorted_index[0]]])
        for dx in range(1, len(sorted_index)):
            pad_num = max_len - train_lens[indexs[start_dx + sorted_index[dx]]]
            pad_list = [PAD_IDX for t in range(pad_num)]
            input_data.append(train_input[indexs[start_dx + sorted_index[dx]]] + pad_list)
            output_data.append(train_output[indexs[start_dx + sorted_index[dx]]])
            lens.append(train_lens[indexs[start_dx + sorted_index[dx]]])
            sentence_ratios.append(train_sentence_ratios[indexs[start_dx + sorted_index[dx]]])
        return torch.LongTensor(input_data), torch.LongTensor(output_data), torch.LongTensor(lens), torch.FloatTensor(sentence_ratios)

    # noinspection PyMethodMayBeStatic
    def msg_batch_preprocess(self, train_data: SolveNotInSecInput, indexs, start_dx, end_dx):
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
        sentence_ratios = []
        has_solve_secs = []
        solve_sec_marks = []
        # 1.对报错信息按照长短重新排序
        sorted_index = np.argsort(np.array(train_data.train_lens)[indexs[start_dx: (end_dx + 1)]] * (-1))
        max_len = train_data.train_lens[indexs[start_dx + sorted_index[0]]]
        input_data.append(train_data.train_input[indexs[start_dx + sorted_index[0]]])
        output_data.append(train_data.train_output[indexs[start_dx + sorted_index[0]]])
        lens.append(train_data.train_lens[indexs[start_dx + sorted_index[0]]])
        sentence_ratios.append(train_data.sentence_ratio[indexs[start_dx + sorted_index[0]]])
        has_solve_secs.append(train_data.has_solve_sec[indexs[start_dx + sorted_index[0]]])
        solve_sec_marks.append(train_data.solve_sec_mark[indexs[start_dx + sorted_index[0]]])
        for dx in range(1, len(sorted_index)):
            pad_num = max_len - train_data.train_lens[indexs[start_dx + sorted_index[dx]]]
            pad_list = [PAD_IDX for t in range(pad_num)]
            input_data.append(train_data.train_input[indexs[start_dx + sorted_index[dx]]] + pad_list)
            output_data.append(train_data.train_output[indexs[start_dx + sorted_index[dx]]] + pad_list)
            lens.append(train_data.train_lens[indexs[start_dx + sorted_index[dx]]])
            sentence_ratios.append(train_data.sentence_ratio[indexs[start_dx + sorted_index[dx]]])
            has_solve_secs.append(train_data.has_solve_sec[indexs[start_dx + sorted_index[dx]]])
            solve_sec_marks.append(train_data.solve_sec_mark[indexs[start_dx + sorted_index[dx]]])
        return torch.LongTensor(input_data), torch.LongTensor(output_data), torch.LongTensor(lens), torch.FloatTensor(sentence_ratios), torch.LongTensor(has_solve_secs), torch.LongTensor(solve_sec_marks)

    def train_sec_model(self, model: BilstmClassifyModel, config, train_input: List[List[int]], train_output: List[int], train_lens: List[int], train_sentence_ratios: List[int]):
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
                input_data, output_data, lengths, para_ratios = self.sec_batch_preprocess(train_input, train_output, train_lens, train_sentence_ratios, indexs, batch * config.batch_size, (batch + 1) * config.batch_size - 1)
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

    def train_msg_model(self, model: BilstmCRFExpandModel, config, train_data: SolveNotInSecInput):
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_decay)  # 每执行一次optimizer之后，学习速率衰减一定比例
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
        self.train_sec_model(self.model_sec_beg, self.config_c, self.sec_data_c.beg_train_input, self.sec_data_c.beg_train_output, self.sec_data_c.beg_train_lens, self.sec_data_c.beg_train_sentence_ratio)
        self.train_sec_model(self.model_sec_end, self.config_c, self.sec_data_c.end_train_input, self.sec_data_c.end_train_output, self.sec_data_c.end_train_lens, self.sec_data_c.end_train_sentence_ratio)
        self.train_sec_model(self.model_sec_rv, self.config_c, self.sec_data_c.rv_train_input, self.sec_data_c.rv_train_output, self.sec_data_c.rv_train_lens, self.sec_data_c.rv_train_sentence_ratio)
        self.train_sec_model(self.model_msg_in_c, self.config_c, self.msg_in_data_c.train_input, self.msg_in_data_c.train_output, self.msg_in_data_c.train_lens, self.msg_in_data_c.sentence_ratio)
        self.train_msg_model(self.model_msg_out_c, self.msg_config_c, self.msg_out_data_c)
        # self.train_msg_model(self.model_msg_n, self.msg_config_n, self.msg_data_n)

    def test_1article_only_sec(self, aid, data_1a: HandledData.A1, wcode_list: List[WordVec1Para]) -> List[List[int]]:
        """
        测试一篇文章的解决方案的起始信号终止信号和反起始信号的识别结果
        :param aid: 文章ID
        :param data_1a: 这篇文章标注好的信息，包括段落、句子，以及一些对应关系
        :param wcode_list: 将文本编码之后的内容
        """
        sec_mark_list = [list() for t in range(data_1a.num_para)]  # 程序认定的每个段落中，每个句子是否涵盖解决方案的起止信号。并非真实标注结果
        # 1.获取一篇文章的起始信号和终止信号。它的优先级高于反起始信号，所以优先判断。
        status = False  # 当前段落是否进入解决方案的内部
        for para_it in range(data_1a.num_para):
            # 如果这段文字没有对应的词编码，或者没有有用的信息，或者为纯代码的话，不作为训练内容。以上一段的判断结果作为本段的标识
            if para_it not in data_1a.para_dx_to_wcode_dx or (not data_1a.has_useful_data[para_it]) or (not data_1a.not_only_code[para_it]):
                if status:
                    sec_mark_list[para_it] = [SolveSecsInput.S_IN for t in range(len(data_1a.sentences[para_it]))]
                else:
                    sec_mark_list[para_it] = [SolveSecsInput.S_NO for t in range(len(data_1a.sentences[para_it]))]
                continue
            wcode_dx = data_1a.para_dx_to_wcode_dx[para_it]
            for sentence_it in range(len(data_1a.sentences[para_it])):
                if (not data_1a.sentence_hud[para_it][sentence_it]) or (not has_chn_chr(data_1a.sentences[para_it][sentence_it])):
                    if status:
                        sec_mark_list[para_it].append(SolveSecsInput.S_IN)
                    else:
                        sec_mark_list[para_it].append(SolveSecsInput.S_NO)
                    continue
                # 生成输入数据
                # 将句子在文章中的比例，扩展到（-2~2）区间段内
                sentence_ratio = (data_1a.sentence_ratio[para_it][sentence_it] - 0.5) * 4
                # 针对这句话生成训练样本
                input_data = []
                for word_it in range(0, len(wcode_list[wcode_dx].vec)):
                    if wcode_list[wcode_dx].vec_to_text_dx_list[word_it] + wcode_list[wcode_dx].vec_len_list[word_it] - 1 < data_1a.sentence_start_dx[para_it][sentence_it]:
                        continue
                    input_data.append(wcode_list[wcode_dx].vec[word_it] + OUTPUT_DIC_SIZE)  # 输入数据需要加上OUTPUT_DIC_SIZE，以避免使用O_IDX等特殊字符
                    if word_it == len(wcode_list[wcode_dx].vec) - 1 or wcode_list[wcode_dx].vec_to_text_dx_list[word_it + 1] > data_1a.sentence_end_dx[para_it][sentence_it]:
                        break
                if len(input_data) == 0 or min(input_data) >= len(self.wv_c.wv.index_to_key) + OUTPUT_DIC_SIZE:  # 完全由标点符号、英文内容组成的句子，不作为信号的判断依据。它将会沿用上一句话的判断结果
                    if status:
                        sec_mark_list[para_it].append(SolveSecsInput.S_IN)
                    else:
                        sec_mark_list[para_it].append(SolveSecsInput.S_NO)
                    continue
                # 使用model进行预测。如果已经进入解决方案的区段，则判断解决方案的结束信号，否则判断解决方案的起始信号
                sentence_ratio = torch.FloatTensor([sentence_ratio])
                length = torch.LongTensor([len(input_data)])
                input_data = torch.LongTensor(input_data).unsqueeze(dim=0)
                if status:
                    output_data = self.model_sec_end.predict(input_data, length, sentence_ratio)[0]
                    if output_data == TRUE_IDX:
                        sec_mark_list[para_it].append(SolveSecsInput.S_END)
                        status = False
                    else:
                        sec_mark_list[para_it].append(SolveSecsInput.S_IN)
                else:
                    output_data = self.model_sec_beg.predict(input_data, length, sentence_ratio)[0]
                    if output_data == TRUE_IDX:
                        sec_mark_list[para_it].append(SolveSecsInput.S_BEG)
                        status = True
                    else:
                        sec_mark_list[para_it].append(SolveSecsInput.S_NO)
        # 2.获取一篇文章的反起始信号
        for para_it in range(data_1a.num_para):
            # 如果这段文字没有对应的词编码，或者没有有用的信息，或者为纯代码的话，不作为训练内容。以上一段的判断结果作为本段的标识
            if para_it not in data_1a.para_dx_to_wcode_dx or (not data_1a.has_useful_data[para_it]) or (not data_1a.not_only_code[para_it]):
                continue
            wcode_dx = data_1a.para_dx_to_wcode_dx[para_it]
            for sentence_it in range(len(data_1a.sentences[para_it])):
                if (not data_1a.sentence_hud[para_it][sentence_it]) or (not has_chn_chr(data_1a.sentences[para_it][sentence_it])):
                    continue
                # 对于已被认定为起始信号，终止信号，或位于起止信号之间的内容，一定不是反起始信号，无需训练。
                if sec_mark_list[para_it][sentence_it] != SolveSecsInput.S_NO:
                    continue
                # 生成输入数据
                # 将句子在文章中的比例，扩展到（-2~2）区间段内
                sentence_ratio = (data_1a.sentence_ratio[para_it][sentence_it] - 0.5) * 4
                # 针对这句话生成训练样本
                input_data = []
                for word_it in range(0, len(wcode_list[wcode_dx].vec)):
                    if wcode_list[wcode_dx].vec_to_text_dx_list[word_it] + wcode_list[wcode_dx].vec_len_list[word_it] - 1 < data_1a.sentence_start_dx[para_it][sentence_it]:
                        continue
                    input_data.append(wcode_list[wcode_dx].vec[word_it] + OUTPUT_DIC_SIZE)  # 输入数据需要加上OUTPUT_DIC_SIZE，以避免使用O_IDX等特殊字符
                    if word_it == len(wcode_list[wcode_dx].vec) - 1 or wcode_list[wcode_dx].vec_to_text_dx_list[word_it + 1] > data_1a.sentence_end_dx[para_it][sentence_it]:
                        break
                if len(input_data) == 0 or min(input_data) >= len(self.wv_c.wv.index_to_key) + OUTPUT_DIC_SIZE:  # 完全由标点符号、英文内容组成的句子，不能作为反起始信号
                    continue
                # 使用model进行预测。如果已经进入解决方案的区段，则判断解决方案的结束信号，否则判断解决方案的起始信号
                sentence_ratio = torch.FloatTensor([sentence_ratio])
                length = torch.LongTensor([len(input_data)])
                input_data = torch.LongTensor(input_data).unsqueeze(dim=0)
                output_data = self.model_sec_rv.predict(input_data, length, sentence_ratio)[0]
                if output_data == TRUE_IDX:
                    sec_mark_list[para_it][sentence_it] = SolveSecsInput.S_RV_BEG
        return sec_mark_list

    def test_1article_only_msg(self, aid, data_1a: HandledData.A1, wcode_list: List[WordVec1Para], sec_mark_list: List[List[int]]) -> Tuple[List[str], List[int]]:
        """
        测试在起始信号、终止信号和反起始信号明确的情况下，解决方案的识别结果是否准确
        :param aid: 文章ID
        :param data_1a: 这篇文章标注好的信息，包括段落、句子，以及一些对应关系
        :param wcode_list: 将文本编码之后的内容
        :param sec_mark_list: 标注好的（或程序生成的）区段信息
        """
        judged = [[False for t0 in range(len(data_1a.sentences[t]))] for t in range(data_1a.num_para)]  # 这个变量表示每个句子是否被遍历过
        is_solve_by_sentence = [[False for t0 in range(len(data_1a.sentences[t]))] for t in range(data_1a.num_para)]
        # 1.先判断起始信号和终止信号之间的内容是否为解决方案的信息。如果一个段落全部落在起始信号和终止信号之间，则整个段落一并判断；如果起始信号和终止信号在同一个段落中，则将中间的内容分句判断。
        for para_it in range(data_1a.num_para):
            # 先判断这个段落作为一个整体来判断，还是分句判断，还是暂缓判断
            all_in_sec_mark = True
            beg_mark_dx = -1
            end_mark_dx = -1
            for sentence_it in range(len(sec_mark_list[para_it])):
                if sec_mark_list[para_it][sentence_it] != SolveSecsInput.S_IN:
                    all_in_sec_mark = False
                if sec_mark_list[para_it][sentence_it] == SolveSecsInput.S_BEG:
                    beg_mark_dx = sentence_it
                if sec_mark_list[para_it][sentence_it] == SolveSecsInput.S_END:
                    end_mark_dx = sentence_it
            if all_in_sec_mark:
                mode = 2
            elif beg_mark_dx != -1 and end_mark_dx != -1 and end_mark_dx > beg_mark_dx:
                mode = 1
            else:
                mode = 0
            if mode == 2:
                # 这一段作为训练数据，且整体性插入
                judged[para_it] = [True for t in range(len(data_1a.sentences[para_it]))]
                # 如果这段文字没有对应的词编码，或者没有有用的信息，或者为纯代码的话，则直接认定为解决方案
                if para_it not in data_1a.para_dx_to_wcode_dx or (not data_1a.has_useful_data[para_it]) or (not data_1a.not_only_code[para_it]):
                    is_solve_by_sentence[para_it] = [True for t in range(len(data_1a.sentences[para_it]))]
                    continue
                wcode_dx = data_1a.para_dx_to_wcode_dx[para_it]
                # 将段落在文章中的比例，扩展到（-2~2）区间段内
                para_ratio = (data_1a.para_ratio[para_it] - 0.5) * 4
                # 生成输入数据
                input_data = []
                for word_it in range(0, len(wcode_list[wcode_dx].vec)):
                    if wcode_list[wcode_dx].vec_to_text_dx_list[word_it] + wcode_list[wcode_dx].vec_len_list[word_it] - 1 < data_1a.useful_data_st_dx[para_it]:
                        continue
                    input_data.append(wcode_list[wcode_dx].vec[word_it] + OUTPUT_DIC_SIZE)  # 输入数据需要加上OUTPUT_DIC_SIZE，以避免使用O_IDX等特殊字符
                # 完全由标点符号、英文内容组成的句子，直接认定为解决方案
                if len(input_data) == 0 or min(input_data) >= len(self.wv_c.wv.index_to_key) + OUTPUT_DIC_SIZE:
                    is_solve_by_sentence[para_it] = [True for t in range(len(data_1a.sentences[para_it]))]
                    continue
                # 使用model进行预测。如果已经进入解决方案的区段，则判断解决方案的结束信号，否则判断解决方案的起始信号
                sentence_ratio = torch.FloatTensor([para_ratio])
                length = torch.LongTensor([len(input_data)])
                input_data = torch.LongTensor(input_data).unsqueeze(dim=0)
                output_data = self.model_msg_in_c.predict(input_data, length, sentence_ratio)[0]
                if output_data == TRUE_IDX:
                    is_solve_by_sentence[para_it] = [True for t in range(len(data_1a.sentences[para_it]))]
            elif mode == 1:
                # 作为训练数据，但按照句子插入
                # 如果这段文字没有对应的词编码，或者没有有用的信息，或者为纯代码的话，则直接认定为解决方案
                if para_it not in data_1a.para_dx_to_wcode_dx or (not data_1a.has_useful_data[para_it]) or (not data_1a.not_only_code[para_it]):
                    judged[para_it] = [True for t in range(len(data_1a.sentences[para_it]))]
                    is_solve_by_sentence[para_it] = [True for t in range(len(data_1a.sentences[para_it]))]
                    continue
                wcode_dx = data_1a.para_dx_to_wcode_dx[para_it]
                for sentence_it in range(len(data_1a.sentences[para_it])):
                    # 只有起始信号和终止信号之间的句子，才能够作为训练内容
                    if beg_mark_dx != -1 and end_mark_dx != -1 and beg_mark_dx < sentence_it < end_mark_dx:
                        judged[para_it][sentence_it] = True
                        # 如果这个句子没有有用的信息，或者为纯代码的话，则直接认定为解决方案
                        if (not data_1a.sentence_hud[para_it][sentence_it]) or (not has_chn_chr(data_1a.sentences[para_it][sentence_it])):
                            is_solve_by_sentence[para_it][sentence_it] = True
                            continue
                        # 将句子在文章中的比例，扩展到（-2~2）区间段内
                        sentence_ratio = (data_1a.sentence_ratio[para_it][sentence_it] - 0.5) * 4
                        # 生成输入数据
                        input_data = []
                        for word_it in range(0, len(wcode_list[wcode_dx].vec)):
                            if wcode_list[wcode_dx].vec_to_text_dx_list[word_it] + wcode_list[wcode_dx].vec_len_list[word_it] - 1 < data_1a.sentence_start_dx[para_it][sentence_it]:
                                continue
                            input_data.append(wcode_list[wcode_dx].vec[word_it] + OUTPUT_DIC_SIZE)  # 输入数据需要加上OUTPUT_DIC_SIZE，以避免使用O_IDX等特殊字符
                            if word_it == len(wcode_list[wcode_dx].vec) - 1 or wcode_list[wcode_dx].vec_to_text_dx_list[word_it + 1] > data_1a.sentence_end_dx[para_it][sentence_it]:
                                break
                        # 完全由标点符号、英文内容组成的句子，直接认定为解决方案
                        if len(input_data) == 0 or min(input_data) >= len(self.wv_c.wv.index_to_key) + OUTPUT_DIC_SIZE:
                            is_solve_by_sentence[para_it][sentence_it] = True
                            continue
                        # 使用model进行预测。如果已经进入解决方案的区段，则判断解决方案的结束信号，否则判断解决方案的起始信号
                        sentence_ratio = torch.FloatTensor([sentence_ratio])
                        length = torch.LongTensor([len(input_data)])
                        input_data = torch.LongTensor(input_data).unsqueeze(dim=0)
                        output_data = self.model_msg_in_c.predict(input_data, length, sentence_ratio)[0]
                        if output_data == TRUE_IDX:
                            is_solve_by_sentence[para_it][sentence_it] = True
        # 2.再判断反起始信号之前的内容是否为解决方案的信息。如果反起始信号位于每一段的第一个句子，则回溯之前的段落，否则只回溯本段之前的句子。
        just_before_rv_seg = False
        for para_it in range(data_1a.num_para - 1, -1, -1):
            # 先判断这个段落作为一个整体来判断，还是分句判断，还是暂缓判断
            beg_mark_dx = -1
            end_mark_dx = -1
            rv_mark_dx = -1
            for sentence_it in range(len(sec_mark_list[para_it])):
                if sec_mark_list[para_it][sentence_it] == SolveSecsInput.S_BEG:
                    beg_mark_dx = sentence_it
                if sec_mark_list[para_it][sentence_it] == SolveSecsInput.S_END:
                    end_mark_dx = sentence_it
                if sec_mark_list[para_it][sentence_it] == SolveSecsInput.S_RV_BEG:
                    rv_mark_dx = sentence_it
            if beg_mark_dx == -1 and end_mark_dx == -1 and just_before_rv_seg:  # 这一段内容中，如果有正式的起止信号，就不能再作为一整个段落来判断反起始信号了
                mode = 2
            elif rv_mark_dx >= 1:
                mode = 1
            else:
                mode = 0
            is_solve_by_para = False  # mode=2的模式下，如果整段内容被认定为解决问题的信息，则该变量赋值为True
            if mode == 2:
                # 这一段作为训练数据，且整体性插入
                judged[para_it] = [True for t in range(len(data_1a.sentences[para_it]))]
                # 如果这段文字没有对应的词编码，或者没有有用的信息，或者为纯代码的话，则直接认定为解决方案
                if para_it not in data_1a.para_dx_to_wcode_dx or (not data_1a.has_useful_data[para_it]) or (not data_1a.not_only_code[para_it]):
                    is_solve_by_sentence[para_it] = [True for t in range(len(data_1a.sentences[para_it]))]
                    is_solve_by_para = True
                else:
                    wcode_dx = data_1a.para_dx_to_wcode_dx[para_it]
                    # 将段落在文章中的比例，扩展到（-2~2）区间段内
                    para_ratio = (data_1a.para_ratio[para_it] - 0.5) * 4
                    # 生成输入数据
                    input_data = []
                    for word_it in range(0, len(wcode_list[wcode_dx].vec)):
                        if wcode_list[wcode_dx].vec_to_text_dx_list[word_it] + wcode_list[wcode_dx].vec_len_list[word_it] - 1 < data_1a.useful_data_st_dx[para_it]:
                            continue
                        input_data.append(wcode_list[wcode_dx].vec[word_it] + OUTPUT_DIC_SIZE)  # 输入数据需要加上OUTPUT_DIC_SIZE，以避免使用O_IDX等特殊字符
                    # 完全由标点符号、英文内容组成的句子，直接认定为解决方案
                    if len(input_data) == 0 or min(input_data) >= len(self.wv_c.wv.index_to_key) + OUTPUT_DIC_SIZE:
                        is_solve_by_sentence[para_it] = [True for t in range(len(data_1a.sentences[para_it]))]
                        is_solve_by_para = True
                    else:
                        # 使用model进行预测。如果已经进入解决方案的区段，则判断解决方案的结束信号，否则判断解决方案的起始信号
                        sentence_ratio = torch.FloatTensor([para_ratio])
                        length = torch.LongTensor([len(input_data)])
                        input_data = torch.LongTensor(input_data).unsqueeze(dim=0)
                        output_data = self.model_msg_in_c.predict(input_data, length, sentence_ratio)[0]
                        if output_data == TRUE_IDX:
                            is_solve_by_sentence[para_it] = [True for t in range(len(data_1a.sentences[para_it]))]
                            is_solve_by_para = True
            elif mode == 1:
                # 作为训练数据，但按照句子插入
                # 如果这段文字没有对应的词编码，或者没有有用的信息，或者为纯代码的话，则直接认定为解决方案
                if para_it not in data_1a.para_dx_to_wcode_dx or (not data_1a.has_useful_data[para_it]) or (not data_1a.not_only_code[para_it]):
                    judged[para_it] = [True for t in range(len(data_1a.sentences[para_it]))]
                    is_solve_by_sentence[para_it] = [True for t in range(len(data_1a.sentences[para_it]))]
                else:
                    wcode_dx = data_1a.para_dx_to_wcode_dx[para_it]
                    for sentence_it in range(len(data_1a.sentences[para_it]) - 1, -1, -1):
                        # 只有刚刚位于反起始信号之前的句子，才能够作为训练内容
                        if end_mark_dx < sentence_it < rv_mark_dx:
                            judged[para_it][sentence_it] = True
                            # 如果这个句子没有有用的信息，或者为纯代码的话，则直接认定为解决方案
                            if (not data_1a.sentence_hud[para_it][sentence_it]) or (not has_chn_chr(data_1a.sentences[para_it][sentence_it])):
                                is_solve_by_sentence[para_it][sentence_it] = True
                            else:
                                # 将句子在文章中的比例，扩展到（-2~2）区间段内
                                sentence_ratio = (data_1a.sentence_ratio[para_it][sentence_it] - 0.5) * 4
                                # 生成输入数据
                                input_data = []
                                for word_it in range(0, len(wcode_list[wcode_dx].vec)):
                                    if wcode_list[wcode_dx].vec_to_text_dx_list[word_it] + wcode_list[wcode_dx].vec_len_list[word_it] - 1 < data_1a.sentence_start_dx[para_it][sentence_it]:
                                        continue
                                    input_data.append(wcode_list[wcode_dx].vec[word_it] + OUTPUT_DIC_SIZE)  # 输入数据需要加上OUTPUT_DIC_SIZE，以避免使用O_IDX等特殊字符
                                    if word_it == len(wcode_list[wcode_dx].vec) - 1 or wcode_list[wcode_dx].vec_to_text_dx_list[word_it + 1] > data_1a.sentence_end_dx[para_it][sentence_it]:
                                        break
                                if len(input_data) == 0 or min(input_data) >= len(self.wv_c.wv.index_to_key) + OUTPUT_DIC_SIZE:
                                    is_solve_by_sentence[para_it][sentence_it] = True
                                else:
                                    # 使用model进行预测。如果已经进入解决方案的区段，则判断解决方案的结束信号，否则判断解决方案的起始信号
                                    sentence_ratio = torch.FloatTensor([sentence_ratio])
                                    length = torch.LongTensor([len(input_data)])
                                    input_data = torch.LongTensor(input_data).unsqueeze(dim=0)
                                    output_data = self.model_msg_in_c.predict(input_data, length, sentence_ratio)[0]
                                    if output_data == TRUE_IDX:
                                        is_solve_by_sentence[para_it][sentence_it] = True
                            if not is_solve_by_sentence[para_it][sentence_it]:
                                break
            # 收尾处理
            if rv_mark_dx == 0:
                just_before_rv_seg = True
            elif just_before_rv_seg and (not is_solve_by_para):
                just_before_rv_seg = False
        # 3.再判断其他内容是否为解决问题的信息，并组装出完整的解决方案的信息
        solve_lines_1article = []
        solve_msg_1article = []
        for para_it in range(data_1a.num_para):
            # 先判断这个段落作为一个整体来判断，还是分句判断，还是已经判断过了
            # mode=0表示已经判断过了，mode=1表示分句判断，mode=2表示整体性判断。
            if all(judged[para_it]):
                mode = 0
            elif any(judged[para_it]):
                mode = 1
            else:
                mode = 2
            if mode == 2:
                # mode=2表示整体性判断
                # 如果这段文字没有对应的词编码，或者没有有用的信息，或者为纯代码的话，则不认定为解决方案
                if para_it not in data_1a.para_dx_to_wcode_dx or (not data_1a.has_useful_data[para_it]) or (not data_1a.not_only_code[para_it]):
                    continue
                # 生成model的输入数据
                input_data = []
                para_start_word_dx = -1
                wcode_dx = data_1a.para_dx_to_wcode_dx[para_it]
                for word_it in range(0, len(wcode_list[wcode_dx].vec)):
                    if wcode_list[wcode_dx].vec_to_text_dx_list[word_it] + wcode_list[wcode_dx].vec_len_list[word_it] - 1 < data_1a.useful_data_st_dx[para_it]:
                        continue
                    if para_start_word_dx == -1:
                        para_start_word_dx = word_it
                    input_data.append(wcode_list[wcode_dx].vec[word_it] + OUTPUT_DIC_SIZE)  # 如果使用事先准备好的词向量列表，则输入数据需要加上OUTPUT_DIC_SIZE，以避免使用O_IDX等特殊字符
                # 完全由标点符号、英文内容组成的句子，不能认定为解决方案
                if len(input_data) == 0 or min(input_data) >= len(self.wv_c.wv.index_to_key) + OUTPUT_DIC_SIZE:
                    continue
                input_data.append(EOS_IDX)
                # model的输入数据，除了段落内容以外，还包括段落在文章中的位置，文章中是否包含解决区段的起止信号等
                para_ratio = (data_1a.para_ratio[para_it] - 0.5) * 4
                has_solve_sec = -1
                for t in sec_mark_list:
                    for t0 in t:
                        if t0 != SolveSecsInput.S_NO:
                            has_solve_sec = 1
                            break
                    if has_solve_sec == 1:
                        break
                solve_sec_mark = -1
                for sec_mark in sec_mark_list[para_it]:
                    if sec_mark != SolveSecsInput.S_NO:
                        solve_sec_mark = 1
                        break
                para_ratio = torch.FloatTensor([para_ratio])
                has_solve_sec = torch.LongTensor([has_solve_sec])
                solve_sec_mark = torch.LongTensor([solve_sec_mark])
                # 2.生成model的输出数据
                length = torch.LongTensor([len(input_data)])
                input_data = torch.LongTensor(input_data).unsqueeze(dim=0)
                output_data = self.model_msg_out_c.predict(input_data, length, para_ratio, has_solve_sec, solve_sec_mark)
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
                    start_char_dx = wcode_list[wcode_dx].vec_to_text_dx_list[start_word_dx + para_start_word_dx]
                    end_char_dx = wcode_list[wcode_dx].vec_to_text_dx_list[end_word_dx + para_start_word_dx - 1] + wcode_list[wcode_dx].vec_len_list[end_word_dx + para_start_word_dx - 1]
                    start_line_dx = data_1a.line_no_by_para[para_it] - data_1a.paragraphs[para_it].count("\n") + data_1a.paragraphs[para_it][:start_char_dx].count("\n")
                    end_line_dx = data_1a.line_no_by_para[para_it] - data_1a.paragraphs[para_it].count("\n") + data_1a.paragraphs[para_it][:end_char_dx].count("\n")
                    if has_chn_or_eng_chr(data_1a.paragraphs[para_it][start_char_dx: end_char_dx]):
                        solve_msg_1article.append(data_1a.paragraphs[para_it][start_char_dx: end_char_dx])
                        solve_lines_1article.extend(list(range(start_line_dx, end_line_dx + 1)))
            elif mode == 1:
                # mode=1表示这段话分句进行判断
                # 如果这段文字没有对应的词编码，或者没有有用的信息，或者为纯代码的话，则不认定为解决方案
                if para_it not in data_1a.para_dx_to_wcode_dx or (not data_1a.has_useful_data[para_it]) or (not data_1a.not_only_code[para_it]):
                    continue
                wcode_dx = data_1a.para_dx_to_wcode_dx[para_it]
                # 生成model的输入数据
                start_char_dx = -1
                tmp_solve_lines = list()
                for sentence_it in range(len(data_1a.sentences[para_it])):
                    if not data_1a.sentence_hud[para_it][sentence_it]:
                        continue
                    if judged[para_it][sentence_it]:
                        # 如果这句话已经在前面被判断过了，那么以前面判断的内容为准，不再重复判断
                        if start_char_dx == -1 and is_solve_by_sentence[para_it][sentence_it]:
                            start_char_dx = data_1a.sentence_start_dx[para_it][sentence_it]
                        if start_char_dx != -1 and sentence_it != 0 and (sentence_it == len(data_1a.sentences[para_it]) - 1 or (not is_solve_by_sentence[para_it][sentence_it])):
                            end_char_dx = data_1a.sentence_end_dx[para_it][sentence_it - 1] + 1
                            start_line_dx = data_1a.line_no_by_para[para_it] - data_1a.paragraphs[para_it].count("\n") + data_1a.paragraphs[para_it][:start_char_dx].count("\n")
                            end_line_dx = data_1a.line_no_by_para[para_it] - data_1a.paragraphs[para_it].count("\n") + data_1a.paragraphs[para_it][:end_char_dx].count("\n")
                            if has_chn_or_eng_chr(data_1a.paragraphs[para_it][start_char_dx: end_char_dx]):
                                solve_msg_1article.append(data_1a.paragraphs[para_it][start_char_dx: end_char_dx])
                                solve_lines_1article.extend(list(range(start_line_dx, end_line_dx + 1)))
                            start_char_dx = -1
                    else:
                        # 如果这句话还没有被判断过，需要判断它是否应当作为解决方案
                        # 将句子在文章中的比例，扩展到（-2~2）区间段内
                        sentence_ratio = (data_1a.sentence_ratio[para_it][sentence_it] - 0.5) * 4
                        has_solve_sec = -1
                        for t in sec_mark_list:
                            for t0 in t:
                                if t0 != SolveSecsInput.S_NO:
                                    has_solve_sec = 1
                                    break
                            if has_solve_sec == 1:
                                break
                        if sec_mark_list[para_it][sentence_it] != SolveSecsInput.S_NO:
                            solve_sec_mark = 1
                        else:
                            solve_sec_mark = -1
                        input_data = []
                        sentence_start_word_dx = -1
                        # 生成这段话的输入信息和输出信息
                        for word_it in range(0, len(wcode_list[wcode_dx].vec)):
                            if wcode_list[wcode_dx].vec_to_text_dx_list[word_it] + wcode_list[wcode_dx].vec_len_list[word_it] - 1 < data_1a.sentence_start_dx[para_it][sentence_it]:
                                continue
                            if sentence_start_word_dx == -1:
                                sentence_start_word_dx = word_it
                            input_data.append(wcode_list[wcode_dx].vec[word_it] + OUTPUT_DIC_SIZE)  # 如果使用事先准备好的词向量列表，则输入数据需要加上OUTPUT_DIC_SIZE，以避免使用O_IDX等特殊字符
                            if word_it == len(wcode_list[wcode_dx].vec) - 1 or wcode_list[wcode_dx].vec_to_text_dx_list[word_it + 1] > data_1a.sentence_end_dx[para_it][sentence_it]:
                                break
                        # 完全由标点符号、英文内容组成的句子，不能认定为解决方案
                        if len(input_data) == 0 or min(input_data) >= len(self.wv_c.wv.index_to_key) + OUTPUT_DIC_SIZE:
                            continue
                        input_data.append(EOS_IDX)
                        sentence_ratio = torch.FloatTensor([sentence_ratio])
                        has_solve_sec = torch.LongTensor([has_solve_sec])
                        solve_sec_mark = torch.LongTensor([solve_sec_mark])
                        # 2.生成model的输出数据
                        length = torch.LongTensor([len(input_data)])
                        input_data = torch.LongTensor(input_data).unsqueeze(dim=0)
                        output_data = self.model_msg_out_c.predict(input_data, length, sentence_ratio, has_solve_sec, solve_sec_mark)
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
                        if ((not has_solve_msg) or start_word_dx != 0) and start_char_dx != -1 and sentence_it != 0:
                            end_char_dx = data_1a.sentence_end_dx[para_it][sentence_it - 1] + 1
                            start_line_dx = data_1a.line_no_by_para[para_it] - data_1a.paragraphs[para_it].count("\n") + data_1a.paragraphs[para_it][:start_char_dx].count("\n")
                            end_line_dx = data_1a.line_no_by_para[para_it] - data_1a.paragraphs[para_it].count("\n") + data_1a.paragraphs[para_it][:end_char_dx].count("\n")
                            if has_chn_or_eng_chr(data_1a.paragraphs[para_it][start_char_dx: end_char_dx]):
                                solve_msg_1article.append(data_1a.paragraphs[para_it][start_char_dx: end_char_dx])
                                tmp_solve_lines.extend(list(range(start_line_dx, end_line_dx + 1)))
                            start_char_dx = -1
                        if has_solve_msg and end_word_dx != len(output_data) - 1:
                            if start_char_dx == -1:
                                start_char_dx = wcode_list[wcode_dx].vec_to_text_dx_list[start_word_dx + sentence_start_word_dx]
                            end_char_dx = wcode_list[wcode_dx].vec_to_text_dx_list[end_word_dx + sentence_start_word_dx - 1] + wcode_list[wcode_dx].vec_len_list[end_word_dx + sentence_start_word_dx - 1]
                            start_line_dx = data_1a.line_no_by_para[para_it] - data_1a.paragraphs[para_it].count("\n") + data_1a.paragraphs[para_it][:start_char_dx].count("\n")
                            end_line_dx = data_1a.line_no_by_para[para_it] - data_1a.paragraphs[para_it].count("\n") + data_1a.paragraphs[para_it][:end_char_dx].count("\n")
                            if has_chn_or_eng_chr(data_1a.paragraphs[para_it][start_char_dx: end_char_dx]):
                                solve_msg_1article.append(data_1a.paragraphs[para_it][start_char_dx: end_char_dx])
                                tmp_solve_lines.extend(list(range(start_line_dx, end_line_dx + 1)))
                            start_char_dx = -1
                        if has_solve_msg and end_word_dx == len(output_data) - 1:
                            if start_word_dx != 0 or start_char_dx == -1:
                                start_char_dx = wcode_list[wcode_dx].vec_to_text_dx_list[start_word_dx + sentence_start_word_dx]
                if tmp_solve_lines:
                    solve_lines_1article.extend(list(set(tmp_solve_lines)))
            else:
                # 这段内容在前面已经被判断过了，直接将内容拼接即可
                tmp_solve_lines = list()
                start_char_dx = 0
                for sentence_it in range(len(data_1a.sentences[para_it])):
                    if is_solve_by_sentence[para_it][sentence_it]:
                        if sentence_it == 0 or (not is_solve_by_sentence[para_it][sentence_it - 1]):
                            start_char_dx = data_1a.sentence_start_dx[para_it][sentence_it]
                        if sentence_it == len(data_1a.sentences[para_it]) - 1 or (not is_solve_by_sentence[para_it][sentence_it + 1]):
                            end_char_dx = data_1a.sentence_end_dx[para_it][sentence_it] + 1
                            start_line_dx = data_1a.line_no_by_para[para_it] - data_1a.paragraphs[para_it].count("\n") + data_1a.paragraphs[para_it][:start_char_dx].count("\n")
                            end_line_dx = data_1a.line_no_by_para[para_it] - data_1a.paragraphs[para_it].count("\n") + data_1a.paragraphs[para_it][:end_char_dx].count("\n")
                            if has_chn_or_eng_chr(data_1a.paragraphs[para_it][start_char_dx: end_char_dx]):
                                solve_msg_1article.append(data_1a.paragraphs[para_it][start_char_dx: end_char_dx])
                                tmp_solve_lines.extend(list(range(start_line_dx, end_line_dx + 1)))
                if tmp_solve_lines:
                    solve_lines_1article.extend(list(set(tmp_solve_lines)))
        return solve_msg_1article, solve_lines_1article

    def test_1article(self, aid, data_1a: HandledData.A1, wcode_list: List[WordVec1Para], sec_mark_list_real: List[List[int]]) -> Tuple[List[List[int]], List[str], List[int], Optional[List[str]], Optional[List[int]]]:
        """
        生成一片文章对应的解决方案的行标和具体内容
        :param aid: 文章ID
        :param data_1a: 这篇文章标注好的信息，包括段落、句子，以及一些对应关系
        :param wcode_list: 将文本编码之后的内容
        :param sec_mark_list_real: 标注好的（或程序生成的）区段信息
        """
        # 1.生成由程序判断的这篇文章的起止段落标记
        sec_mark_list = self.test_1article_only_sec(aid, data_1a, wcode_list)
        # 2.根据程序判断的起止段落的标记，来决定解决问题的信息的位置
        solve_msg, solve_lines = self.test_1article_only_msg(aid, data_1a, wcode_list, sec_mark_list)
        solve_msg_realsec, solve_lines_realsec = self.test_1article_only_msg(aid, data_1a, wcode_list, sec_mark_list_real)
        return sec_mark_list, solve_msg, solve_lines, solve_msg_realsec, solve_lines_realsec

    def test_all_secs(self, mark_data):
        all_solve_secs = dict()  # 每篇文章的解决区段的起止信号的位置
        self.model_sec_beg.eval()
        self.model_sec_end.eval()
        self.model_sec_rv.eval()
        with torch.no_grad():
            for aid in mark_data:
                if aid in self.test_aid_list and mark_data[aid].err_msg != str():
                    sec_mark_list = self.test_1article_only_sec(aid, self.handled_data[aid], self.wv_c.wcode_list[aid].text_c)
                    all_solve_secs[aid] = sec_mark_list
        solve_sec_score, sec_ratio_score, total_solve_score = SolveValidation.whole_secs(all_solve_secs, mark_data, self.handled_data, self.sec_data_c.all_sec_marks)
        print('solve_sec_score = %.2f / %d' % (solve_sec_score, total_solve_score))
        print('sec_ratio_score = %.2f / %d' % (sec_ratio_score, total_solve_score))

    def test_all_msgs(self, mark_data):
        all_solve_msgs_realsec = dict()  # 每篇文章的解决问题的信息（使用标注好的区段信息）
        all_solve_lines_realsec = dict()  # 每篇文章的解决方案所在的行标（使用标注好的区段信息）
        self.model_msg_in_c.eval()
        self.model_msg_out_c.eval()
        with torch.no_grad():
            for aid in mark_data:
                if aid in self.test_aid_list and mark_data[aid].err_msg != str():
                    solve_msg_realsec, solve_lines_realsec = self.test_1article_only_msg(aid, self.handled_data[aid], self.wv_c.wcode_list[aid].text_c, self.sec_data_c.all_sec_marks[aid])
                    all_solve_msgs_realsec[aid] = solve_msg_realsec
                    all_solve_lines_realsec[aid] = solve_lines_realsec
        solve_line_score, total_solve_score = SolveValidation.line_ratio(all_solve_lines_realsec, mark_data)
        solve_msg_score, total_solve_msg_score = SolveValidation.sentence_ratio(all_solve_msgs_realsec, mark_data)
        print('solve_line_score = %.2f / %d' % (solve_line_score, total_solve_score))
        print('solve_msg_score = %.2f / %d' % (solve_msg_score, total_solve_msg_score))

    def test(self, mark_data):
        """使用测试数据，验证解决方案起始信号和结束信号的识别的准确率"""
        # 2.生成算法对每篇文章的解决方案的判断结果
        all_solve_secs = dict()  # 每篇文章的解决区段的起止信号的位置
        all_solve_msgs = dict()  # 每篇文章的解决问题的信息
        all_solve_lines = dict()  # 每篇文章的解决方案所在的行标
        all_solve_msgs_realsec = dict()  # 每篇文章的解决问题的信息（使用标注好的区段信息）
        all_solve_lines_realsec = dict()  # 每篇文章的解决方案所在的行标（使用标注好的区段信息）
        self.model_sec_beg.eval()
        self.model_sec_end.eval()
        self.model_sec_rv.eval()
        self.model_msg_in_c.eval()
        self.model_msg_out_c.eval()
        with torch.no_grad():
            for aid in mark_data:
                if aid in self.test_aid_list and mark_data[aid].err_msg != str():
                    solve_sec, solve_msg, solve_lines, solve_msg_realsec, solve_lines_realsec = self.test_1article(aid, self.handled_data[aid], self.wv_c.wcode_list[aid].text_c, self.sec_data_c.all_sec_marks[aid])
                    all_solve_secs[aid] = solve_sec
                    all_solve_msgs[aid] = solve_msg
                    all_solve_lines[aid] = solve_lines
                    all_solve_msgs_realsec[aid] = solve_msg_realsec
                    all_solve_lines_realsec[aid] = solve_lines_realsec
        # 3.验证解决问题的起始和结束段落的判断结果
        solve_sec_score, sec_ratio_score, total_solve_score = SolveValidation.whole_secs(all_solve_secs, mark_data, self.handled_data, self.sec_data_c.all_sec_marks)
        print('solve_sec_score = %.2f / %d' % (solve_sec_score, total_solve_score))
        print('sec_ratio_score = %.2f / %d' % (sec_ratio_score, total_solve_score))
        # print('sec_begin_score = %.2f / %d' % (sec_score_beg, total_score_beg))
        # print('sec_end_score = %.2f / %d' % (sec_score_end, total_score_end))
        # 4.验证解决问题的行标和内容的判断结果
        solve_line_score, total_solve_score = SolveValidation.line_ratio(all_solve_lines, mark_data)
        solve_msg_score, total_solve_msg_score = SolveValidation.sentence_ratio(all_solve_msgs, mark_data)
        print('solve_line_score = %.2f / %d' % (solve_line_score, total_solve_score))
        print('solve_msg_score = %.2f / %d' % (solve_msg_score, total_solve_msg_score))

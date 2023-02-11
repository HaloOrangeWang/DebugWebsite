from settings import *
from .model import BilstmCRFModel
from .funcs import WordVec1Para, WordVec1Article, has_chn_chr
from .word_vec import WvNormal, WvCode
from validations.basic import ErrMsgValidation
import torch.nn as nn
import torch
import time
import numpy as np
import copy
from typing import Dict


class ErrMsgConfigN:
    """英文正常解析时，模型的配置"""
    embedding_size = 68  # 词向量维度
    hidden_size = 64  # BILSTM有多少个隐藏单元
    # dropout = 0.2
    nepoch = 20  # 一共训练多少轮
    batch_size = 30  # 每一个批次将多少组数据提交给BILSTM-CRF模型
    lr = 0.005
    lr_decay = 0.85  # 学习速率衰减


class ErrMsgConfigC:
    """将英文当做<code>时，模型的配置"""
    embedding_size = 70  # 词向量维度
    hidden_size = 64  # BILSTM有多少个隐藏单元
    # dropout = 0.2
    nepoch = 20  # 一共训练多少轮
    batch_size = 30  # 每一个批次将多少组数据提交给BILSTM-CRF模型
    lr = 0.005
    lr_decay = 0.85  # 学习速率衰减


class ErrMsgInput:
    """对数据做整理，提供输入给训练ErrorMsg用的模型的数据内容"""

    def __init__(self, wdict, rv_wdict):
        self.wdict = wdict
        self.rv_wdict = rv_wdict
        self.train_input = []  # 训练用的输入数据。
        self.train_output = []  # 训练用的输出数据。
        self.train_lens = []  # 训练用的数据的长度
        # self.test_input = []
        # self.test_output = []

    # noinspection PyMethodMayBeStatic
    def get_err_msg_dx_list(self, para_dx, text, err_msg, err_lines, lnbp_1article, is_title):
        """
        获取一个段落中，错误信息所在的位置。如果没有错误信息，则返回空数组
        :param para_dx: 这个段落是文章中的第几个段落
        :param text: 段落文字
        :param err_msg: 错误信息
        :param err_lines: 错误信息所在的行标
        :param lnbp_1article: 这篇文章中，每个段落对应的行标的列表
        :param is_title: 是否为标题
        :return:
        """
        # 1.先判断这一段是否包含错误信息
        if is_title:
            if "title" not in err_lines:
                return list()
        else:
            line_no_set = set(range(lnbp_1article[para_dx] - text.count('\n'), lnbp_1article[para_dx] + 1))
            if len(line_no_set & set(err_lines)) == 0:
                return list()
        # 2.如果这一段包含错误信息，再获取错误信息在段落中的位置
        dx = text.index(err_msg)
        return list(range(dx, dx + len(err_msg)))

    def get_data_1para(self, wcode: WordVec1Para, err_msg_dx_list):
        """
        根据一个段落的内容，生成训练数据或测试数据
        :param wcode: 将文本编码之后的内容
        :param err_msg_dx_list: 错误信息在这段话中的位置
        """
        # 1.获取这一段话对应的错误信息和文本编码结果之间的对应关系
        is_err_msg_by_word = [False for t in range(len(wcode.vec))]
        wcode_dx = 0
        err_msg_dx = 0
        while True:
            if wcode_dx >= len(wcode.vec) or err_msg_dx >= len(err_msg_dx_list):
                break
            if wcode.vec_to_text_dx_list[wcode_dx] == err_msg_dx_list[err_msg_dx]:
                is_err_msg_by_word[wcode_dx] = True
                wcode_dx += 1
                err_msg_dx += 1
            elif wcode.vec_to_text_dx_list[wcode_dx] < err_msg_dx_list[err_msg_dx]:
                wcode_dx += 1
            else:
                err_msg_dx += 1
        # 2.生成训练数据。
        input_data = []  # copy.deepcopy(wcode.vec)
        output_data = []
        for word_it in range(0, len(wcode.vec)):
            # 对于一段报错内容，输出的数据应当为 [起始标志B_IDX、报错过程中的标志I_IDX、结束标志E_IDX]
            # 例如：对于 [Python 报错   NameError name  not   defined]，输出的内容应当为
            #          [O_IDX  O_IDX B_IDX     I_IDX I_IDX E_IDX]
            input_data.append(wcode.vec[word_it] + OUTPUT_DIC_SIZE)  # 如果使用事先准备好的词向量列表，则输入数据需要加上OUTPUT_DIC_SIZE，以避免使用O_IDX等特殊字符
            if is_err_msg_by_word[word_it]:
                if word_it == 0 or (not is_err_msg_by_word[word_it - 1]):
                    output_data.append(B_IDX)
                elif word_it == len(wcode.vec) - 1 or (not is_err_msg_by_word[word_it + 1]):
                    output_data.append(E_IDX)
                else:
                    output_data.append(I_IDX)
            else:
                output_data.append(O_IDX)
            # 如果句子长度超过了50，则按照标点符号进行拆分。保证拆分后每条数据的长度都略微超过50，且错误信息不能跨条目。
            if word_it == len(wcode.vec) - 1 or (len(input_data) >= 50 and wcode.vec[word_it] == len(self.wdict) + CHN_PUNC_WRD and output_data[-1] == O_IDX and output_data[-2] == O_IDX):
                input_data.append(EOS_IDX)
                output_data.append(EOS_IDX)
                self.train_input.append(copy.copy(input_data))
                self.train_output.append(copy.copy(output_data))
                self.train_lens.append(len(input_data))
                input_data = []
                output_data = []

    def get_model_data(self, articles, wcode_list: Dict[int, WordVec1Article], mark_data, line_no_by_para, punc_code_set, test_aid_list):
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
            if aid in test_aid_list:
                continue
            # 1.处理标题信息
            err_msg_dx_title = self.get_err_msg_dx_list(-1, articles[aid].title, mark_data[aid].err_msg, mark_data[aid].err_lines, line_no_by_para[aid], True)
            self.get_data_1para(wcode_list[aid].title_c, err_msg_dx_title)
            # 2.分析程序需要根据哪几个正文段落生成正文对应的错误信息训练数据
            paragraphs = articles[aid].text.rstrip('\n').replace('\n\n\n', '\n\n').split('\n\n')
            end_dx = len(paragraphs)
            if not articles[aid].text.rstrip('\n'):
                end_dx = 0  # 如果正文内容为空，则不要尝试从正文中提取报错信息
            elif len(mark_data[aid].err_lines) == 0:
                end_dx = min(len(paragraphs), 5)
            elif mark_data[aid].err_lines[-1] == "title":
                end_dx = 0
            else:
                for it in range(len(paragraphs)):
                    if line_no_by_para[aid][it] - paragraphs[it].count("\n") > mark_data[aid].err_lines[-1]:
                        end_dx = it
                        break
            # 3.处理正文信息。
            para_dx_to_wcode_dx = dict()
            for wcode_it in range(len(wcode_list[aid].text_c)):
                para_dx_to_wcode_dx[wcode_list[aid].text_c[wcode_it].paragraph_dx] = wcode_it
            for para_it in range(end_dx):
                if para_it in para_dx_to_wcode_dx:  # 如果一个段落没有对应的编码，可能它是由纯标点符号、空格组成的。此时不尝试从这一段话中提取报错信息。
                    err_msg_dx_list = self.get_err_msg_dx_list(para_it, paragraphs[para_it], mark_data[aid].err_msg, mark_data[aid].err_lines, line_no_by_para[aid], False)
                    self.get_data_1para(wcode_list[aid].text_c[para_dx_to_wcode_dx[para_it]], err_msg_dx_list)


class ErrMsgPipe:
    """生成错误信息的训练数据、调度模型进行训练并生成测试结果的管理类"""

    def __init__(self, wv_n: WvNormal, wv_c: WvCode):
        """
        :param wv_n: 将英文正常编码的方式，词向量的编码内容
        :param wv_c: 将英文编码为<code>的方式，词向量的编码内容
        """
        self.wv_n = wv_n
        self.wv_c = wv_c
        self.data_n = ErrMsgInput(self.wv_n.wv.key_to_index, self.wv_n.wv.index_to_key)  # 训练和测试用的数据
        self.data_c = ErrMsgInput(self.wv_c.wv.key_to_index, self.wv_c.wv.index_to_key)  # 训练和测试用的数据
        self.config_n = ErrMsgConfigN()
        self.config_c = ErrMsgConfigC()
        self.model_n = BilstmCRFModel(self.config_n, self.wv_n.emb_mat, freeze=False)  # 将英文正常编码训练模型
        self.model_c = BilstmCRFModel(self.config_c, self.wv_c.emb_mat, freeze=True)  # 将英文编码为<code>训练模型
        self.test_aid_list = []

    def setup(self, articles, mark_data, line_no_by_para, punc_code_set):
        """准备模型的输入数据，以及相关配置等"""
        # 设置哪些数据用于训练，而哪些数据用于测试。在正式生成阶段，所有数据均只用于训练
        if MODE != 'Generate':
            aid_list = [t for t in mark_data]
            aid_list = sorted(aid_list)
            self.test_aid_list = aid_list[int(TEST_DATA_START * len(aid_list)): int((TEST_DATA_START + TEST_DATA_RATIO) * len(aid_list))]
        # 准备训练数据。包括将英文编码为<code>的训练数据，以及将英文正常编码的训练数据
        self.data_n.get_model_data(articles, self.wv_n.wcode_list, mark_data, line_no_by_para, punc_code_set, self.test_aid_list)
        self.data_c.get_model_data(articles, self.wv_c.wcode_list, mark_data, line_no_by_para, punc_code_set, self.test_aid_list)

    # noinspection PyMethodMayBeStatic
    def batch_preprocess(self, train_data: ErrMsgInput, indexs, start_dx, end_dx):
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
        # 1.对报错信息按照长短重新排序
        sorted_index = np.argsort(np.array(train_data.train_lens)[indexs[start_dx: (end_dx + 1)]] * (-1))
        max_len = train_data.train_lens[indexs[start_dx + sorted_index[0]]]
        input_data.append(train_data.train_input[indexs[start_dx + sorted_index[0]]])
        output_data.append(train_data.train_output[indexs[start_dx + sorted_index[0]]])
        lens.append(train_data.train_lens[indexs[start_dx + sorted_index[0]]])
        for dx in range(1, len(sorted_index)):
            pad_num = max_len - train_data.train_lens[indexs[start_dx + sorted_index[dx]]]
            pad_list = [PAD_IDX for t in range(pad_num)]
            input_data.append(train_data.train_input[indexs[start_dx + sorted_index[dx]]] + pad_list)
            output_data.append(train_data.train_output[indexs[start_dx + sorted_index[dx]]] + pad_list)
            lens.append(train_data.train_lens[indexs[start_dx + sorted_index[dx]]])
        return torch.LongTensor(input_data), torch.LongTensor(output_data), torch.LongTensor(lens)

    def train_1model(self, model: BilstmCRFModel, config, train_data: ErrMsgInput):
        """
        训练模型
        :param model: 训练那种模型（可选项: self.model_n, self.model_c）
        :param config: 使用那种模型配置（可选项：self.config_n, self.config_c）
        :param train_data: 使用那种训练数据（可选项：self.data_n，self.data_c）
        """
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
                input_data, output_data, lengths = self.batch_preprocess(train_data, indexs, batch * config.batch_size, (batch + 1) * config.batch_size - 1)
                # 2.2 前向传输：根据参数计算损失函数
                optimizer.zero_grad()
                scores = model(input_data, lengths)
                loss = model.calc_loss(scores, output_data)
                # 2.3 反向传播：根据损失函数优化参数
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            end_time = time.time()
            print("epoch %d, time = %.2f, loss = %.6f" % (epoch, (end_time - start_time), total_loss))

    def train(self):
        """训练model_n和model_c"""
        self.train_1model(self.model_n, self.config_n, self.data_n)
        self.train_1model(self.model_c, self.config_c, self.data_c)

    # noinspection PyMethodMayBeStatic
    def get_err_msg_1para_1model(self, wcode: WordVec1Para, model: BilstmCRFModel):
        """
        根据一种特定的模型，以及一段输入的内容，来获取对应输出的错误信息
        :param wcode: 编码内容
        :param model: 使用哪种训练错误信息的模型
        :return 是否存在错误信息，以及错误信息在原字符串中的索引值
        """
        has_err_msg = False
        # 1.使用BILSTM-CRF模型，生成这个模型下，“每个词是否为错误信息”的数组
        input_data = [t + OUTPUT_DIC_SIZE for t in wcode.vec] + [EOS_IDX]
        length = torch.LongTensor([len(input_data)])
        input_data = torch.LongTensor(input_data).unsqueeze(dim=0)
        output_data = model.predict(input_data, length)
        # 2.根据“每个词是否为报错内容”的信息，生成报错信息
        start_word_dx = -1
        end_word_dx = len(output_data) - 1
        for it in range(len(output_data) - 1):  # 这里直到len(output_data) - 1，是因为最后一项应该是作为信息末尾标志EOS_IDX，不代表句子中的正式内容
            if output_data[it] in [B_IDX, I_IDX, E_IDX]:
                has_err_msg = True
                if start_word_dx == -1:
                    start_word_dx = it
            if output_data[it] == O_IDX:
                if has_err_msg:
                    end_word_dx = it
                    break
        if has_err_msg:
            start_char_dx = wcode.vec_to_text_dx_list[start_word_dx]
            end_char_dx = wcode.vec_to_text_dx_list[end_word_dx - 1] + wcode.vec_len_list[end_word_dx - 1]
            return True, start_char_dx, end_char_dx
        else:
            return False, -1, -1

    def get_err_msg_1para(self, wcode_n: WordVec1Para, wcode_c: WordVec1Para, text):
        """
        根据一个段落，获取对应的错误信息
        :param wcode_n: 将英文内容正常编码时的词编码内容
        :param wcode_c: 将英文内容编码为<code>时的词编码内容
        :param text: 这段话原始的内容
        :return: 是否为错误信息
        """
        # 1.分别生成英文正常编码，以及英文编码为<code>时的错误信息
        if wcode_n is not None:
            has_err_msg_n, start_char_dx_n, end_char_dx_n = self.get_err_msg_1para_1model(wcode_n, self.model_n)
        else:
            has_err_msg_n, start_char_dx_n, end_char_dx_n = False, -1, -1
        if wcode_c is not None:
            has_err_msg_c, start_char_dx_c, end_char_dx_c = self.get_err_msg_1para_1model(wcode_c, self.model_c)
        else:
            has_err_msg_c, start_char_dx_c, end_char_dx_c = False, -1, -1
        if (not has_err_msg_n) and (not has_err_msg_c):
            return False, str()
        # 2.如果英文编码为<code>的错误信息完全包含了英文正常编码的错误信息，且多出来的内容为纯英文，则以英文编码为<code>时输出的错误信息为准
        if has_err_msg_c:
            if not has_err_msg_n:
                if not has_chn_chr(text[start_char_dx_c: end_char_dx_c]):
                    return True, text[start_char_dx_c: end_char_dx_c]
            else:
                if start_char_dx_n >= start_char_dx_c and end_char_dx_n <= end_char_dx_c:
                    if (not has_chn_chr(text[start_char_dx_c: start_char_dx_n])) and (not has_chn_chr(text[end_char_dx_n: end_char_dx_c])):
                        return True, text[start_char_dx_c: end_char_dx_c]
        # 3.如果不符合上述条件，则以英文正常编码时，输出的词向量为准
        if has_err_msg_n:
            return True, text[start_char_dx_n: end_char_dx_n]
        else:
            return False, str()

    def test_1article(self, aid, article):
        """
        根据一篇测试文章，验证错误信息的生成结果是否准确。
        :param aid: 文章ID
        :param article: 这一篇文章的标题和正文
        :return:
        """
        wcode_n = self.wv_n.wcode_list[aid]
        wcode_c = self.wv_c.wcode_list[aid]
        # 1.根据标题判断是否为标题信息
        is_err_msg_title, err_msg = self.get_err_msg_1para(wcode_n.title_c, wcode_c.title_c, article.title)
        if not is_err_msg_title:
            # 2.对正文进行逐段分析，判断是否为报错信息
            paragraphs = article.text.rstrip('\n').replace('\n\n\n', '\n\n').split('\n\n')
            para_dx_to_wcode_dx_n = dict()
            para_dx_to_wcode_dx_c = dict()
            for wcode_it in range(len(wcode_n.text_c)):
                para_dx_to_wcode_dx_n[wcode_n.text_c[wcode_it].paragraph_dx] = wcode_it
            for wcode_it in range(len(wcode_c.text_c)):
                para_dx_to_wcode_dx_c[wcode_c.text_c[wcode_it].paragraph_dx] = wcode_it
            for para_it in range(min(5, len(paragraphs))):
                if para_it in para_dx_to_wcode_dx_n:  # 如果一个段落没有对应的编码，可能它是由纯标点符号、空格组成的。此时不尝试从这一段话中提取报错信息。
                    wcode_n_1para = wcode_n.text_c[para_dx_to_wcode_dx_n[para_it]]
                else:
                    wcode_n_1para = None
                if para_it in para_dx_to_wcode_dx_c:  # 如果一个段落没有对应的编码，可能它是由纯标点符号、空格组成的。此时不尝试从这一段话中提取报错信息。
                    wcode_c_1para = wcode_c.text_c[para_dx_to_wcode_dx_c[para_it]]
                else:
                    wcode_c_1para = None
                is_err_msg, err_msg = self.get_err_msg_1para(wcode_n_1para, wcode_c_1para, paragraphs[para_it])
                if is_err_msg:
                    break
        return err_msg

    def test(self, articles, mark_data):
        """使用测试数据，验证训练的成功率"""

        all_err_msgs = dict()
        self.model_n.eval()
        self.model_c.eval()
        with torch.no_grad():
            for aid in mark_data:
                if aid in self.test_aid_list:
                    err_msg = self.test_1article(aid, articles[aid])
                    all_err_msgs[aid] = err_msg
        score = ErrMsgValidation.whole_sentence(all_err_msgs, mark_data)
        print("test score = %.2f / %d" % (score, len(self.test_aid_list)))

    def generate(self, articles, err_aid_set) -> Dict[int, str]:
        """生成全部文章对应的错误信息"""
        all_err_msgs = dict()
        self.model_n.eval()
        self.model_c.eval()
        with torch.no_grad():
            __cnt = 0
            for aid in articles:
                if aid not in err_aid_set:
                    err_msg = self.test_1article(aid, articles[aid])
                    all_err_msgs[aid] = err_msg
                else:
                    all_err_msgs[aid] = str()
                __cnt += 1
                if __cnt % 100 == 0:
                    print('__cnt = %d, aid = %d' % (__cnt, aid))
        return all_err_msgs

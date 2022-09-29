from settings import *
from .funcs import has_chn_chr, has_eng_chr, has_chn_or_eng_chr, has_chn_or_eng_or_digit_chr, WordVec1Para, WordVec1Article
from jieba import lcut, add_word
from whoosh.analysis import RegexTokenizer, LowercaseFilter, StemFilter
from gensim.models import word2vec
from typing import List, Dict
import numpy as np
import re


NegativeWords = ['error', 'warning', 'errno', 'except', 'traceback', 'fail', 'failed']
SubNegativeWords = ['not', 'no', 'null']


def add_words():
    f = open('../SpiderData/new_words.txt', 'r', encoding='utf8')
    new_words = [t.replace('\n', '') for t in f.readlines()]
    f.close()
    for word in new_words:
        add_word(word, freq=10000)


class WvNormal:
    """对英文进行正常编码的词向量编码方式"""

    def __init__(self):
        self.wv: word2vec.KeyedVectors  # 词向量编码的内容
        self.wcode_list: Dict[int, WordVec1Article]  # 对每篇文章进行词编码后的内容
        self.emb_mat: np.array  # 对每篇文章进行词编码后的内容
        self.eng_tokenizer = RegexTokenizer() | LowercaseFilter() | StemFilter()
        self.err_aid_set = set()

    def handle_text_before_2vec(self, text_input, stopwords):
        """
        构造词向量之前，先对文章中的句子进行Tokenizer操作
        :param text_input: 输入的内容
        :param stopwords: 停用词列表
        :return 正则化之后的词列表（给词向量训练用，给正式训练用）
        """
        # 1.先对文章内容按照换行符进行分隔
        paragraphs = text_input.strip('\n').replace('\n\n\n', '\n').replace('\n\n', '\n').split('\n')
        tokenizer_texts = []
        for paragraph in paragraphs:
            # 如果这一行内容既没有中文，也没有英文，则这行内容将不作为词向量的训练样本
            if not has_chn_or_eng_chr(paragraph):
                continue
            words = lcut(paragraph)
            clean_words = []
            # 2.去除标点符号和停用词。中文直接加进clean_words中，英文先改为小写并去除ed、ing等后缀后再加进clean_words中
            for word in words:
                is_chn_word = has_chn_chr(word)
                is_eng_word = (not is_chn_word) and has_chn_or_eng_chr(word)
                if is_chn_word:
                    # 中文词：如果不是停用词，则要添加进clean_words中
                    if word not in stopwords:
                        clean_words.append(word)
                elif is_eng_word:
                    clean_word = [t.text for t in self.eng_tokenizer(word)]
                    assert len(clean_word) == 1
                    clean_word = clean_word[0]
                    # if clean_word not in stopwords:
                    clean_words.append(clean_word)
                else:
                    # 如果这个词中既没有中文也没有英文，说明可能是标点符号、数字或下划线。这段话不能用于词向量的训练
                    pass
            if len(clean_words) != 0:
                tokenizer_texts.append(clean_words)
        return tokenizer_texts

    # noinspection PyAttributeOutsideInit
    def word_to_vec(self, articles, stopwords):
        """训练词向量"""
        # 1.对原始文章进行分词
        __cnt = 0
        corpus = []
        for aid in articles:
            title = articles[aid].title
            text = articles[aid].text
            tokenized_title = self.handle_text_before_2vec(title, stopwords)
            tokenized_text = self.handle_text_before_2vec(text, stopwords)
            corpus.extend(tokenized_title)
            corpus.extend(tokenized_text)
            __cnt += 1
            if __cnt % 100 == 0:
                print('__cnt = %d, aid = %d\n' % (__cnt, aid))
        # 2.训练词向量
        model = word2vec.Word2Vec(corpus, vector_size=64, min_count=3, window=5, workers=4)
        # 词向量生成之后，先不做归一化操作。尽管里面有数值是超过1的。
        self.wv = model.wv

    def text_to_vec(self, text_input, stopwords: list) -> List[WordVec1Para]:
        """
        将样本转化为词向量
        :param text_input: 输入的内容
        :param stopwords: 停用词列表
        :return 转化为词向量之后的列表
        """
        # 先将文章按照'\n\n\n'划分成段落
        paragraph_list = []
        paragraphs_large = text_input.split('\n\n\n')
        for t in range(len(paragraphs_large)):
            paragraphs = paragraphs_large[t].split('\n\n')
            for t0 in range(len(paragraphs)):
                if t0 == 0:
                    paragraph_list.append((paragraphs[t0], True))
                else:
                    paragraph_list.append((paragraphs[t0], False))
        # 按照段落，生成每一段中词在wv中的索引值。忽略停用词。
        word_vector_1article = []
        for paragraph_it in range(len(paragraph_list)):
            paragraph = paragraph_list[paragraph_it][0]
            if not paragraph:
                continue
            words = lcut(paragraph)
            wv_index_1para = []
            vec_in_text_dx_list = []
            vec_len_list = []
            last_word_dx = 0
            word_dx = 0
            while True:
                word = words[word_dx]
                is_chn_word = has_chn_chr(word)
                is_eng_word = (not is_chn_word) and has_chn_or_eng_chr(word)
                is_digit = (not is_chn_word) and (not is_eng_word) and has_chn_or_eng_or_digit_chr(word)
                if is_chn_word:
                    # 中文词：如果不是停用词，则找到对应的词向量
                    if word not in stopwords:
                        if word in self.wv:
                            wv_index = self.wv.key_to_index[word]
                        else:
                            wv_index = len(self.wv) + SPC_CHN_WRD
                        wv_index_1para.append(wv_index)
                        vec_in_text_dx_list.append(last_word_dx)
                        vec_len_list.append(len(word))
                    word_dx += 1
                    last_word_dx += len(word)
                elif is_eng_word:
                    # 英文词：先先改为小写并去除ed、ing等后缀。如果不在停用词列表中，则找到对应的词向量
                    clean_word = [t.text for t in self.eng_tokenizer(word)]
                    assert len(clean_word) == 1
                    clean_word = clean_word[0]
                    # if clean_word not in stopwords:
                    if clean_word in self.wv:
                        wv_index = self.wv.key_to_index[clean_word]
                    else:
                        wv_index = len(self.wv) + SPC_ENG_WRD
                    wv_index_1para.append(wv_index)
                    vec_in_text_dx_list.append(last_word_dx)
                    vec_len_list.append(len(word))
                    word_dx += 1
                    last_word_dx += len(word)
                elif is_digit:
                    # 数字：不需要寻找对应的词向量，直接以digit的方式加入到列表中
                    wv_index_1para.append(len(self.wv) + DIGIT_WRD)
                    vec_in_text_dx_list.append(last_word_dx)
                    vec_len_list.append(len(word))
                    word_dx += 1
                    last_word_dx += len(word)
                else:
                    # 如果这个词中既没有中文也没有英文也没有数字，说明可能是标点符号或下划线。
                    # 如果是逗号、句号、冒号、分号或换行符，则认为是一个分隔的标点符号。其他情况不计入训练样本
                    if word[0] in ['，', '。', '；', '：', '\n']:
                        wv_index = len(self.wv) + CHN_PUNC_WRD
                        wv_index_1para.append(wv_index)
                        vec_in_text_dx_list.append(last_word_dx)
                        vec_len_list.append(1)
                    elif word[0] in [',', ';', ':']:
                        wv_index = len(self.wv) + ENG_PUNC_WRD
                        wv_index_1para.append(wv_index)
                        vec_in_text_dx_list.append(last_word_dx)
                        vec_len_list.append(1)
                    word_dx += 1
                    last_word_dx += len(word)
                if word_dx >= len(words):
                    break
            if len(wv_index_1para) != 0:
                word_vec_tosave = WordVec1Para()
                word_vec_tosave.vec = wv_index_1para
                word_vec_tosave.vec_to_text_dx_list = vec_in_text_dx_list
                word_vec_tosave.vec_len_list = vec_len_list
                word_vec_tosave.paragraph_dx = paragraph_it
                word_vec_tosave.is_first_para = paragraph_list[paragraph_it][1]
                word_vector_1article.append(word_vec_tosave)
        return word_vector_1article

    # noinspection PyAttributeOutsideInit
    def get_word_vec_marked(self, articles, mark_data, stopwords):
        """
        获取标注好的文章的词向量列表
        :param articles: 文章列表
        :param mark_data: 哪些文章被标注了
        :param stopwords: 停用词列表
        :return:
        """
        __cnt = 0
        wv_list = {aid: WordVec1Article() for aid in articles}
        for aid in articles:
            title = articles[aid].title.rstrip("\n")
            text = articles[aid].text.rstrip("\n")
            wv_title = self.text_to_vec(title, stopwords)
            wv_text = self.text_to_vec(text, stopwords)
            if len(wv_title) == 0:
                self.err_aid_set.add(aid)
                continue
            wv_list[aid].title_c = wv_title[0]
            wv_list[aid].text_c = wv_text
            __cnt += 1
            if __cnt % 100 == 0:
                print('__cnt = %d, aid = %d\n' % (__cnt, aid))
        self.wcode_list = wv_list

    # noinspection PyAttributeOutsideInit
    def construct_wv(self):
        """构建用于训练的Embedding初始向量列表。"""
        # 输出的词向量一共有68维，其中前64维是中文的词编码，65为数字，66为标点符号，67和68分别为中文低频词和英文低频词
        emb_mat = np.zeros(shape=(len(self.wv) + OUTPUT_DIC_SIZE + NUM_SPC_WORDS, 68), dtype=np.float64)
        emb_mat[OUTPUT_DIC_SIZE: OUTPUT_DIC_SIZE + len(self.wv), :64] = self.wv.vectors
        emb_mat[OUTPUT_DIC_SIZE + len(self.wv) + SPC_CHN_WRD, 66] = 1
        emb_mat[OUTPUT_DIC_SIZE + len(self.wv) + SPC_ENG_WRD, 67] = 1
        emb_mat[OUTPUT_DIC_SIZE + len(self.wv) + DIGIT_WRD, 64] = 1
        emb_mat[OUTPUT_DIC_SIZE + len(self.wv) + CHN_PUNC_WRD, 65] = 1
        emb_mat[OUTPUT_DIC_SIZE + len(self.wv) + ENG_PUNC_WRD, 65] = 1
        self.emb_mat = emb_mat


class WvCode:
    """将英文视为<code>的词向量编码方式"""

    def __init__(self):
        self.wv: word2vec.KeyedVectors  # 词向量编码的内容
        self.wcode_list: Dict[int, WordVec1Article]  # 对每篇文章进行词编码后的内容
        self.emb_mat: np.array  # 对每篇文章进行词编码后的内容
        self.err_aid_set = set()

    # noinspection PyMethodMayBeStatic
    def handle_text_before_2vec(self, text_input, stopwords):
        """
        构造词向量之前，先对文章中的句子进行Tokenizer操作
        :param text_input: 输入的内容
        :param stopwords: 停用词列表
        :return 正则化之后的词列表（给词向量训练用，给正式训练用）
        """
        # 1.先对文章内容按照换行符进行分隔
        paragraphs = text_input.strip('\n').replace('\n\n\n', '\n').replace('\n\n', '\n').split('\n')
        tokenizer_texts = []
        for paragraph in paragraphs:
            # 如果这一行内容为全英文，则可能是代码展示。这行内容将不作为词向量的训练样本
            if not has_chn_chr(paragraph):
                continue
            words = lcut(paragraph)
            clean_words = []
            is_last_word_chn = None  # 标注上一个字符是不是中文字符
            # 3.去除标点符号和停用词，并将英文视为代码段
            for word in words:
                is_chn_word = has_chn_chr(word)
                is_eng_word = (not is_chn_word) and has_chn_or_eng_chr(word)
                if is_chn_word:
                    # 中文词：如果不是停用词，则要添加进clean_words中
                    is_last_word_chn = True
                    if word not in stopwords:
                        clean_words.append(word)
                elif is_eng_word:
                    # 英文词：当做代码段处理。但多个连续的英文词只作为一个代码段即可。
                    if is_last_word_chn is True or is_last_word_chn is None:
                        clean_words.append("<code>")
                    is_last_word_chn = False
                else:
                    # 如果这个词中既没有中文也没有英文，说明可能是标点符号、数字或下划线。这段话不能用于词向量的训练
                    pass
            if len(clean_words) != 0:
                tokenizer_texts.append(clean_words)
        return tokenizer_texts

    # noinspection PyAttributeOutsideInit
    def word_to_vec(self, articles, stopwords):
        """训练词向量"""
        # 1.对原始文章进行分词
        __cnt = 0
        corpus = []
        for aid in articles:
            title = articles[aid].title
            text = articles[aid].text
            tokenized_title = self.handle_text_before_2vec(title, stopwords)
            tokenized_text = self.handle_text_before_2vec(text, stopwords)
            corpus.extend(tokenized_title)
            corpus.extend(tokenized_text)
            __cnt += 1
            if __cnt % 100 == 0:
                print('__cnt = %d, aid = %d\n' % (__cnt, aid))
        # 2.训练词向量
        model = word2vec.Word2Vec(corpus, vector_size=64, min_count=3, window=5, workers=4)
        # 词向量生成之后，先不做归一化操作。尽管里面有数值是超过1的。
        self.wv = model.wv

    # noinspection PyMethodMayBeStatic
    def classify_eng_words(self, text, word_list):
        """
        对英文词进行分类。
        :param text: 英文内容。如：python 3.8
        :param word_list: jieba.cut切分之后的词列表。如：['python', ' ', '3.8']
        :return [报错的权重, 库名的权重, 代码段的权重, 路径名的权重]
        """
        # 预处理
        text_lower = text.lower()
        word_list_lower = []
        is_eng_word = [0 for t in range(len(word_list))]
        for word_it in range(len(word_list)):
            if has_eng_chr(word_list[word_it]):
                is_eng_word[word_it] = 1
        for word in word_list:
            word_list_lower.append(word.lower())
        # 1. 如果正文中包含Error、Fail等词汇，且除了这些词汇以外还有其他英文词，则认定为报错信息
        for word in NegativeWords:
            if word in text_lower:
                if has_eng_chr(text_lower.replace(word, '')):
                    return ERR_ENG_WRD
        # 2.如果出现not/no/null等词汇，且长度超过3个字符，则认定为 80%报错信息，20%代码信息
        if sum(is_eng_word) >= 3:
            for word in word_list_lower:
                for word2 in SubNegativeWords:
                    if word == word2:
                        return ERC_ENG_WRD
        # 3.如果内容中的标点符号只有"+-_.=,"，且没有连续2个英文单词中间有空格的情况，则认定为场景信息
        is_scene_sentence = 2  # 2表示是场景信息，1表示不太可能是场景信息，0表示不是场景信息
        last_word_eng = False
        for word_it in range(len(word_list_lower)):
            if not is_eng_word[word_it]:
                # 如果是空格或标点符号，则标点符号必须在指定范围内，否则不能认定为场景信息
                if word_list_lower[word_it][0] in "()[]（）":
                    is_scene_sentence = 1
                elif word_list_lower[word_it][0] not in "+-_.=,， 0123456789" and word_list_lower[word_it][0] != chr(0x0a):
                    is_scene_sentence = 0
                    break
                if word_list_lower[word_it][0] != " " and word_list_lower[word_it][0] != chr(0x0a):
                    last_word_eng = False
            else:
                # 如果是英文词。则需要判断它和上一个英文词中间是否只有空格作为分隔。如果是的话，也不能认定为场景信息。
                if last_word_eng:
                    if last_word_eng:
                        is_scene_sentence = 0
                        break
                    if word_list_lower[word_it] not in ['python', 'visual']:
                        last_word_eng = True
        if is_scene_sentence == 2:
            return SEN_ENG_WRD
        elif is_scene_sentence == 1:
            return COS_ENG_WRD
        # 4.判断内容中是否有http和/。如果有的话，则这段英文内容应当为一个地址
        if "http" in text_lower and '/' in text_lower:
            return LNK_ENG_WRD
        # 5.判断是否以C:\xxx开头，或/xxx/xxx开头。如果是的话，则这段英文应当为一个地址。
        if re.match(r'^[A-Za-z]:\\[\w]+', text_lower):
            return LNK_ENG_WRD
        if re.match(r'^/[\w]+/[\w]+', text_lower):
            return LNK_ENG_WRD
        # 其他情况认定为代码段
        return COD_ENG_WRD

    def text_to_vec(self, text_input, stopwords: list) -> List[WordVec1Para]:
        """
        将样本转化为词向量
        :param text_input: 输入的内容
        :param stopwords: 停用词列表
        :return 转化为词向量之后的列表
        """
        # 先将文章按照'\n\n\n'划分成段落
        paragraph_list = []
        paragraphs_large = text_input.split('\n\n\n')
        for t in range(len(paragraphs_large)):
            paragraphs = paragraphs_large[t].split('\n\n')
            for t0 in range(len(paragraphs)):
                if t0 == 0:
                    paragraph_list.append((paragraphs[t0], True))
                else:
                    paragraph_list.append((paragraphs[t0], False))
        # 按照段落，生成每一段中词在wv中的索引值。忽略停用词。
        word_vector_1article = []
        for paragraph_it in range(len(paragraph_list)):
            paragraph = paragraph_list[paragraph_it][0]
            if not paragraph:
                continue
            words = lcut(paragraph)
            wv_index_1para = []
            vec_in_text_dx_list = []
            vec_len_list = []
            last_word_dx = 0
            word_dx = 0
            while True:
                word = words[word_dx]
                is_chn_word = has_chn_chr(word)
                is_eng_word = (not is_chn_word) and has_chn_or_eng_chr(word)
                if is_chn_word:
                    # 中文词：如果不是停用词，则找到对应的词向量
                    if word not in stopwords:
                        if word in self.wv:
                            wv_index = self.wv.key_to_index[word]
                        else:
                            wv_index = len(self.wv) + SPC_CHN_WRD
                        wv_index_1para.append(wv_index)
                        vec_in_text_dx_list.append(last_word_dx)
                        vec_len_list.append(len(word))
                    word_dx += 1
                    last_word_dx += len(word)
                elif is_eng_word:
                    # 英文词：当做代码段处理。但多个连续的英文词只作为一个代码段即可。
                    code_text = ""
                    code_word_list = []
                    code_word_st_dx = last_word_dx
                    while True:
                        code_text += words[word_dx]
                        code_word_list.append(words[word_dx])
                        last_word_dx += len(words[word_dx])
                        word_dx += 1
                        if word_dx >= len(words) or has_chn_chr(words[word_dx]):
                            break
                    wv_index = len(self.wv) + self.classify_eng_words(code_text, code_word_list)
                    wv_index_1para.append(wv_index)
                    vec_in_text_dx_list.append(code_word_st_dx)
                    vec_len_list.append(last_word_dx - code_word_st_dx)
                else:
                    # 如果这个词中既没有中文也没有英文，说明可能是标点符号、数字或下划线。
                    # 如果是逗号、句号、冒号、分号或换行符，则认为是一个分隔的标点符号。其他情况不计入训练样本
                    if word[0] in ['，', '。', '；', '：', '\n']:
                        wv_index = len(self.wv) + CHN_PUNC_WRD
                        wv_index_1para.append(wv_index)
                        vec_in_text_dx_list.append(last_word_dx)
                        vec_len_list.append(1)
                    elif word[0] in [',', ';', ':']:
                        wv_index = len(self.wv) + ENG_PUNC_WRD
                        wv_index_1para.append(wv_index)
                        vec_in_text_dx_list.append(last_word_dx)
                        vec_len_list.append(1)
                    word_dx += 1
                    last_word_dx += len(word)
                if word_dx >= len(words):
                    break
            if len(wv_index_1para) != 0:
                word_vec_tosave = WordVec1Para()
                word_vec_tosave.vec = wv_index_1para
                word_vec_tosave.vec_to_text_dx_list = vec_in_text_dx_list
                word_vec_tosave.vec_len_list = vec_len_list
                word_vec_tosave.paragraph_dx = paragraph_it
                word_vec_tosave.is_first_para = paragraph_list[paragraph_it][1]
                word_vector_1article.append(word_vec_tosave)
        return word_vector_1article

    # noinspection PyAttributeOutsideInit
    def get_word_vec_marked(self, articles, mark_data, stopwords):
        """
        获取标注好的文章的词向量列表
        :param articles: 文章列表
        :param mark_data: 哪些文章被标注了
        :param stopwords: 停用词列表
        :return:
        """
        __cnt = 0
        wv_list = {aid: WordVec1Article() for aid in articles}
        for aid in articles:
            title = articles[aid].title.rstrip("\n")
            text = articles[aid].text.rstrip("\n")
            wv_title = self.text_to_vec(title, stopwords)
            wv_text = self.text_to_vec(text, stopwords)
            if len(wv_title) == 0:
                self.err_aid_set.add(aid)
                continue
            wv_list[aid].title_c = wv_title[0]
            wv_list[aid].text_c = wv_text
            __cnt += 1
            if __cnt % 100 == 0:
                print('__cnt = %d, aid = %d\n' % (__cnt, aid))
        self.wcode_list = wv_list

    # noinspection PyAttributeOutsideInit
    def construct_wv(self):
        """构建用于训练的Embedding初始向量列表。"""
        # 输出的词向量一共有70维，其中前64维是中文的词编码，65-68为错误信息、库名、代码段、路径名的权重，69为标点符号，70为低频中文词
        emb_mat = np.zeros(shape=(len(self.wv) + OUTPUT_DIC_SIZE + NUM_SPC_WORDS, 70), dtype=np.float64)
        emb_mat[OUTPUT_DIC_SIZE: OUTPUT_DIC_SIZE + len(self.wv), :64] = self.wv.vectors
        emb_mat[OUTPUT_DIC_SIZE + len(self.wv) + SPC_CHN_WRD, 69] = 1
        emb_mat[OUTPUT_DIC_SIZE + len(self.wv) + ERR_ENG_WRD, 64: 68] = [1, 0, 0, 0]
        emb_mat[OUTPUT_DIC_SIZE + len(self.wv) + ERC_ENG_WRD, 64: 68] = [0.8, 0, 0.2, 0]
        emb_mat[OUTPUT_DIC_SIZE + len(self.wv) + SEN_ENG_WRD, 64: 68] = [0, 1, 0, 0]
        emb_mat[OUTPUT_DIC_SIZE + len(self.wv) + COS_ENG_WRD, 64: 68] = [0, 0.2, 0.8, 0]
        emb_mat[OUTPUT_DIC_SIZE + len(self.wv) + COD_ENG_WRD, 64: 68] = [0, 0, 1, 0]
        emb_mat[OUTPUT_DIC_SIZE + len(self.wv) + LNK_ENG_WRD, 64: 68] = [0, 0, 0, 1]
        emb_mat[OUTPUT_DIC_SIZE + len(self.wv) + CHN_PUNC_WRD, 68] = 1
        emb_mat[OUTPUT_DIC_SIZE + len(self.wv) + ENG_PUNC_WRD, 68] = 1
        self.emb_mat = emb_mat

from configs import *
from .load_data import load_base_data, load_articles, load_link
from .funcs import is_chn_chrs
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.analysis import SimpleAnalyzer
from jieba.analyse import ChineseAnalyzer
import os
import re


def split_chn_chr(str_list1):
    """拆分中文字符。如：‘这是什么what’ -> ‘[这, 是, 什, 么, what]’ """
    str_list2 = []
    for str1 in str_list1:
        st_dx = -1
        for t in range(len(str1)):
            if is_chn_chrs(str1[t]):
                if st_dx == -1:
                    str_list2.append(str1[t])
                else:
                    str_list2.append(str1[st_dx: t])
                    str_list2.append(str1[t])
                    st_dx = -1
            else:
                if st_dx == -1:
                    st_dx = t
        if st_dx != -1:
            str_list2.append(str1[st_dx:])
    return str_list2


def get_common_sstr(str_list1, str_list2):
    """获取两个字符串列表的公共子串。公共子串的加权长度必须大于等于4。字符串列表的格式如：[这,啥,呢,what]。先假定按照‘不允许字符不一致’的情况处理"""
    # todo: “允许字符不一致”的情况
    common_sstr_list = []
    table = [[0 for t0 in range(len(str_list2))] for t in range(len(str_list1))]
    start_dx_list = [[-1 for t0 in range(len(str_list2))] for t in range(len(str_list1))]
    for t in range(len(str_list1)):
        for t0 in range(len(str_list2)):
            if str_list1[t] == str_list2[t0]:
                if len(str_list1[t]) != 1:
                    weight = 2
                else:
                    weight = 1
                if t == 0 or t0 == 0:
                    table[t][t0] = weight
                    start_dx_list[t][t0] = t
                else:
                    table[t][t0] = table[t - 1][t0 - 1] + weight
                    if start_dx_list[t - 1][t0 - 1] == -1:
                        start_dx_list[t][t0] = t
                    else:
                        start_dx_list[t][t0] = start_dx_list[t - 1][t0 - 1]
                if table[t][t0] >= COMMON_SSTR_MIN_LEN:
                    common_sstr = str_list1[start_dx_list[t][t0]: t + 1]
                    common_sstr_list.append(common_sstr)
    return common_sstr_list


def judge_sublist(str_list1, str_list2):
    """判断str_list1是不是str_list2的子序列"""
    for t0 in range(len(str_list2) - len(str_list1)):
        if str_list2[t0] == str_list1[0]:
            if str_list2[t0: t0 + len(str_list1)] == str_list1:
                return True
    return False


def solve_msg_rank(all_common_sstrs, all_sstr_aid_list, sstr_dx_list):
    """
    对所有找到的多篇文章的公共子串进行排序。排序方法为 公共子串的长度×出现文章的频次×出现文章的频次
    :param all_common_sstrs: 所有找到的多篇文章的公共子串
    :param all_sstr_aid_list: 公共子串所在的文章信息
    :param sstr_dx_list: 公共子串在solve_statistic中aid_list中的索引值
    :return: 排序后的公共子串，及其所在的文章列表，所在的solve_statistic中aid_list中的索引值的列表
    """
    # 统计各个公共子串出现的频次
    unique_common_sstrs = []
    unique_aid_list = []
    unique_sstr_dx_list = []
    for t in range(len(all_common_sstrs)):
        is_found = False
        for t0 in range(len(unique_common_sstrs)):
            if all_common_sstrs[t] == unique_common_sstrs[t0]:
                unique_aid_list[t0].add(all_sstr_aid_list[t][0])
                unique_aid_list[t0].add(all_sstr_aid_list[t][1])
                unique_sstr_dx_list[t0].extend(sstr_dx_list[t])
                is_found = True
                break
        if not is_found:
            unique_common_sstrs.append(all_common_sstrs[t])
            unique_aid_list.append(set(all_sstr_aid_list[t]))
            unique_sstr_dx_list.append(sstr_dx_list[t])
    # 对统计上来的公共子串，按照长度×出现频次进行排序

    def get_weight_strlen(str_list1):
        weight_strlen = 0
        for t1 in range(len(str_list1)):
            if len(str_list1[t1]) >= 2:
                weight_strlen += 2
            else:
                weight_strlen += 1
        return weight_strlen

    score_list = dict()
    for t in range(len(unique_common_sstrs)):
        score_list[t] = get_weight_strlen(unique_common_sstrs[t]) * len(unique_aid_list[t]) * len(unique_aid_list[t])
    ranked_dx_list = sorted(score_list.items(), key=lambda x: x[1], reverse=True)
    ranked_common_sstrs = []
    ranked_aid_list = []
    ranked_sstr_dx_list = []
    cnt = 0
    for t in range(len(score_list)):
        dx = ranked_dx_list[t][0]
        is_sublist = False
        for t0 in range(len(ranked_common_sstrs)):
            if judge_sublist(unique_common_sstrs[dx], ranked_common_sstrs[t0]):
                is_sublist = True
                break
        if not is_sublist:
            ranked_common_sstrs.append(unique_common_sstrs[dx])
            ranked_aid_list.append(unique_aid_list[dx])
            ranked_sstr_dx_list.append(unique_sstr_dx_list[dx])
            cnt += 1
            if cnt >= SSTR_PAGE_LIMIT:
                break
    return ranked_common_sstrs, ranked_aid_list, ranked_sstr_dx_list


def find_origin_str(solve_msg, tokenized_str):
    """
    找到每一条处理后的报错信息对应的原始的字符串
    :param solve_msg: 原始的解决方案的字符串
    :param tokenized_str: 处理后的解决问题的子串
    :return: 这段子串对应的原始字符串
    """
    solve_start_dx = -1
    solve_cur_dx = 0
    tokenize_dx = 0
    solve_msg_lower = solve_msg.lower()  # 先对字符串最小化处理
    while True:
        strlen = len(tokenized_str[tokenize_dx])
        if solve_msg_lower[solve_cur_dx: solve_cur_dx + strlen] == tokenized_str[tokenize_dx]:
            if tokenize_dx == 0:
                solve_start_dx = solve_cur_dx
            tokenize_dx += 1
            solve_cur_dx += strlen
            if tokenize_dx == len(tokenized_str):
                return solve_msg[solve_start_dx: solve_cur_dx]
        elif not re.match(r'\w', solve_msg_lower[solve_cur_dx]):
            solve_cur_dx += 1
        else:
            solve_start_dx = -1
            solve_cur_dx += 1
            tokenize_dx = 0
        if solve_cur_dx >= len(solve_msg):
            raise RuntimeError('没有找到TokenizedStr')


class SearchMethod:

    def __init__(self):
        self.base_data = None  # 文章id与正文、解决方案等的对应关系
        self.searcher = None  # 使用whoosh引擎得到的searcher和schema
        self.schema = None

    def preparation(self):
        # load数据
        base_data = load_base_data()
        base_data = load_articles(base_data)
        base_data = load_link(base_data)
        # 1.构建whoosh搜索机制
        schema = Schema(aid=ID(unique=True, stored=True), err_msg=TEXT(ChineseAnalyzer()), title=TEXT(ChineseAnalyzer()), content=TEXT(ChineseAnalyzer()))
        indexdir = 'indexdir/'
        if not os.path.exists(indexdir):
            os.makedirs(indexdir)
        ix = create_in(indexdir, schema)
        writer = ix.writer()
        for aid in base_data:
            aid_str = '%06d' % aid
            writer.add_document(aid=aid_str, err_msg=base_data[aid].err_msg, title=base_data[aid].title, content=base_data[aid].content)
        writer.commit()
        searcher = ix.searcher()
        self.base_data = base_data
        self.searcher = searcher
        self.schema = schema

    def get_all_solves(self, err_msg, scene):
        """
        根据输入的错误信息和场景信息，找到对应的解决方案列表
        :param err_msg: 要查找的错误信息
        :param scene: 要查找的场景信息。如果不想填写场景信息的话，则此处填写空字符串
        :return: 文章ID以及解决方案的列表
        """
        # 查找这个错误以及场景对应的全部文章ID
        qp = QueryParser("err_msg", schema=self.schema)
        results_err_msg = self.searcher.search(qp.parse(err_msg), limit=None)
        aid_list_err_msg = [int(t['aid']) for t in results_err_msg]
        if scene != str():
            qp2 = MultifieldParser(["title", "content"], schema=self.schema)
            # qp2 = QueryParser("content", schema=schema)
            results_scene = self.searcher.search(qp2.parse(scene), limit=None)
            aid_list_scene = [int(t['aid']) for t in results_scene]
            aid_list = list(set(aid_list_err_msg) & set(aid_list_scene))
        else:
            aid_list = aid_list_err_msg
        # 返回这些文章对应的解决方案
        aid_list_output = list()
        solve_msg_output = list()
        for aid in aid_list:
            for solve in self.base_data[aid].solves:
                aid_list_output.append(aid)
                solve_msg_output.append(solve)
        return aid_list_output, solve_msg_output

    def solve_statistic(self, aid_list, solve_msgs):
        """
        对解决方案的信息进行汇总（至少输入2篇文章才有意义）
        :param aid_list: 输入的文章ID列表
        :param solve_msgs: 输入的解决方案信息列表
        :return:
        """
        all_common_sstrs = list()
        all_sstr_aid_list = list()
        sstr_dx_list = []
        # 先对进行标准化处理。去掉标点符号，并将英文变为小写。
        tokenized_solves = []
        analyzer = SimpleAnalyzer()
        for aid_it in range(len(aid_list)):
            tokenized_str_list = [t0.text for t0 in analyzer(solve_msgs[aid_it])]
            tokenized_solves.append(split_chn_chr(tokenized_str_list))
        # 再对拆分好的解决方案信息进行两两对比，找到不同文章的公共子串
        for aid_it in range(len(aid_list) - 1):
            for aid_it2 in range(aid_it + 1, len(aid_list)):
                common_sstr_list = get_common_sstr(tokenized_solves[aid_it], tokenized_solves[aid_it2])
                if common_sstr_list:
                    all_common_sstrs.extend(common_sstr_list)
                    all_sstr_aid_list.extend([(aid_list[aid_it], aid_list[aid_it2]) for t0 in range(len(common_sstr_list))])
                    sstr_dx_list.extend([[aid_it, aid_it2] for t0 in range(len(common_sstr_list))])
        if len(all_common_sstrs) == 0:
            return [], []
        # 再计算存在于多篇文章中的公共子串，并进行排序
        ranked_common_sstrs, ranked_aid_list, ranked_sstr_dx_list = solve_msg_rank(all_common_sstrs, all_sstr_aid_list, sstr_dx_list)
        # 找出每个公共子串对应的原字符串
        ranked_common_sstr_origin = []
        for t in range(len(ranked_common_sstrs)):
            origin_str = find_origin_str(solve_msgs[ranked_sstr_dx_list[t][0]], ranked_common_sstrs[t])
            ranked_common_sstr_origin.append(origin_str)
        return ranked_common_sstr_origin, ranked_aid_list


# def main():
#     base_data, [searcher, schema] = preparation()
#     # 找到错误信息对应的解决方案
#     err_msg = "UnicodeDecodeError:  codec can't decode byte in position"
#     scene = "文件"
#     aid_list, solve_msgs = get_all_solves(err_msg, scene, base_data, searcher, schema)
#     ranked_common_sstr_origin, ranked_aid_list = solve_statistic(aid_list, solve_msgs)
#     print('7')
#
#
# if __name__ == '__main__':
#     main()

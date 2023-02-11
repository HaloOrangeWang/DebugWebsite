from settings import *
from data_io.load_data import load_mark_data, load_articles, load_stopwords
from data_io.train_store import db_connect, insert_data
from train_simple import get_err_msg, get_solve_msg
from train_ml.word_vec import add_words, WvNormal, WvCode
from train_ml.funcs import get_all_line_no
from train_ml.err_msg import ErrMsgPipe
from train_ml.solve2 import SolvePipe
from validations.basic import ErrMsgValidation, SolveValidation # , valid_scene


def train_no_ml():
    # 从文件中读取标题列表，文章正文以及标注好的数据
    article_dict = load_articles()
    mark_data = load_mark_data()
    # 从文件中获取到错误、场景和解决方法的信息
    all_err_msgs = dict()
    all_scenes = dict()
    all_solve_lines = dict()
    all_solve_msgs = dict()
    for aid in article_dict:
        if aid in mark_data:
            err_msg = get_err_msg(article_dict[aid].title, article_dict[aid].text)
            all_err_msgs[aid] = err_msg
            if mark_data[aid].err_msg != str():
                # all_scenes[aid] = get_scene_msg(article_dict[aid].title, article_dict[aid].text, mark_data[aid].err_msg)
                all_solve_lines[aid], all_solve_msgs[aid] = get_solve_msg(article_dict[aid].text)
    # 根据已标注好的60条数据，评价错误信息判定的准确性
    err_msg_score = ErrMsgValidation.whole_sentence(all_err_msgs, mark_data)
    # scene_score, total_scene_score = valid_scene(all_scenes, mark_data)
    solve_line_score, total_solve_score = SolveValidation.line_ratio(all_solve_lines, mark_data)
    solve_msg_score, total_solve_msg_score = SolveValidation.sentence_ratio(all_solve_msgs, mark_data)
    # solve_line_score, solve_msg_score, total_solve_score = valid_solve(all_solve_lines, all_solve_msgs, mark_data)
    print('error_msg_score = %.2f / 300' % err_msg_score)
    # print('scene_score = %.2f / %d' % (scene_score, total_scene_score))
    print('solve_line_score = %.2f / %d' % (solve_line_score, total_solve_score))
    print('solve_msg_score = %.2f / %d' % (solve_msg_score, total_solve_msg_score))


def generate_no_ml():
    """根据所有的文章生成报错信息、场景信息和解决方法信息，并存放起来"""
    __cnt = 0
    # 从文件中读取标题列表，文章正文以及标注好的数据
    article_dict = load_articles()
    # 从文件中获取到错误、场景和解决方法的信息
    all_err_msgs = dict()
    # all_scenes = dict()
    all_solve_lines = dict()
    all_solve_msgs = dict()
    for aid in article_dict:
        if 100000 <= aid < 200000:
            err_msg = get_err_msg(article_dict[aid].title, article_dict[aid].text)
            all_err_msgs[aid] = err_msg
            if all_err_msgs[aid] != str():
                # all_scenes[aid] = get_scene_msg(article_dict[aid].title, article_dict[aid].text, all_err_msgs[aid])
                all_solve_lines[aid], all_solve_msgs[aid] = get_solve_msg(article_dict[aid].text)
            __cnt += 1
            if __cnt % 100 == 0:
                print('__cnt = %d, aid = %d' % (__cnt, aid))
    # 将生成好的数据存放进数据库中
    db_col = db_connect()
    insert_data(db_col, all_err_msgs, all_solve_msgs)


def train_ml():
    # 从文件中读取标题列表，文章正文以及标注好的数据，以及停用词列表
    article_dict_all = load_articles()
    mark_data = load_mark_data()
    stopwords = load_stopwords()
    add_words()

    article_dict_marked = dict()
    for aid in mark_data:
        article_dict_marked[aid] = article_dict_all[aid]
    # 获取每篇文章的正文中，每个段落对应的行标
    line_no_by_para = get_all_line_no(article_dict_marked)
    # 训练词向量（英文正常编码的方式）。并根据训练好的词向量，获取每篇标注好的文章的词向量列表，以及词向量embedding的矩阵
    wv_n = WvNormal()
    wv_n.word_to_vec(article_dict_all, stopwords)
    wv_n.get_word_vec_marked(article_dict_marked, mark_data, stopwords)
    wv_n.construct_wv()
    # 训练词向量（英文编码为<code>的方式）。并根据训练好的词向量，获取每篇标注好的文章的词向量列表，以及词向量embedding的矩阵
    wv_c = WvCode()
    wv_c.word_to_vec(article_dict_all, stopwords)
    wv_c.get_word_vec_marked(article_dict_marked, mark_data, stopwords)
    wv_c.construct_wv()
    # 错误信息的模型构建和训练
    err_pipe = ErrMsgPipe(wv_n, wv_c)
    err_pipe.setup(article_dict_marked, mark_data, line_no_by_para, [])
    err_pipe.train()
    # 解决方案信息的模型构建和训练
    solve_pipe = SolvePipe(wv_n, wv_c)
    solve_pipe.setup(article_dict_marked, mark_data, line_no_by_para, [])
    solve_pipe.train()
    # 测试错误信息和解决方案信息的训练结果
    err_pipe.test(article_dict_marked, mark_data)
    solve_pipe.test(mark_data)


def generate_ml():
    # 从文件中读取标题列表，文章正文以及标注好的数据，以及停用词列表
    article_dict_all = load_articles()
    mark_data = load_mark_data()
    stopwords = load_stopwords()
    add_words()

    article_dict_marked = dict()
    for aid in mark_data:
        article_dict_marked[aid] = article_dict_all[aid]
    # 获取每篇文章的正文中，每个段落对应的行标
    line_no_by_para = get_all_line_no(article_dict_all)
    # 训练词向量（英文正常编码的方式）。并根据训练好的词向量，获取每篇标注好的文章的词向量列表，以及词向量embedding的矩阵
    wv_n = WvNormal()
    wv_n.word_to_vec(article_dict_all, stopwords)
    wv_n.get_word_vec_marked(article_dict_all, mark_data, stopwords)
    wv_n.construct_wv()
    # 训练词向量（英文编码为<code>的方式）。并根据训练好的词向量，获取每篇标注好的文章的词向量列表，以及词向量embedding的矩阵
    wv_c = WvCode()
    wv_c.word_to_vec(article_dict_all, stopwords)
    wv_c.get_word_vec_marked(article_dict_all, mark_data, stopwords)
    wv_c.construct_wv()
    err_aid_set = wv_n.err_aid_set | wv_c.err_aid_set
    # 错误信息的模型构建和训练
    err_pipe = ErrMsgPipe(wv_n, wv_c)
    err_pipe.setup(article_dict_marked, mark_data, line_no_by_para, [])
    err_pipe.train()
    # 解决方案信息的模型构建和训练
    solve_pipe = SolvePipe(wv_n, wv_c)
    solve_pipe.setup(article_dict_all, mark_data, line_no_by_para, [])  # 解决方案的模型，第一个参数选择article_dict_all，因为要生成全部文章的handled_data
    solve_pipe.train()
    # 生成全部文章错误信息和解决方案信息的结果
    all_err_msgs = err_pipe.generate(article_dict_all, err_aid_set)
    all_solve_msgs = solve_pipe.generate(article_dict_all, err_aid_set, all_err_msgs)
    # 将生成好的数据存放进数据库中
    db_col = db_connect()
    insert_data(db_col, all_err_msgs, all_solve_msgs)


def main():
    if MODE == 'Train' and (not ML):
        train_no_ml()
    elif MODE == 'Generate' and (not ML):
        generate_no_ml()
    elif MODE == 'Train' and ML:
        train_ml()
    elif MODE == 'Generate' and ML:
        generate_ml()


if __name__ == '__main__':
    main()

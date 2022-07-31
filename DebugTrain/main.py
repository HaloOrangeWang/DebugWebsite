from settings import *
from data_io.load_data import load_mark_data, load_articles
from data_io.train_store import db_connect, insert_data
from train_simple import get_err_msg, get_scene_msg, get_solve_msg
from validations.basic import valid_err_msg, valid_scene, valid_solve


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
                all_scenes[aid] = get_scene_msg(article_dict[aid].title, article_dict[aid].text, mark_data[aid].err_msg)
                all_solve_lines[aid], all_solve_msgs[aid] = get_solve_msg(article_dict[aid].text)
    # 根据已标注好的60条数据，评价错误信息判定的准确性
    err_msg_score = valid_err_msg(all_err_msgs, mark_data)
    scene_score, total_scene_score = valid_scene(all_scenes, mark_data)
    solve_line_score, solve_msg_score, total_solve_score = valid_solve(all_solve_lines, all_solve_msgs, mark_data)
    print('error_msg_score = %.2f / 60' % err_msg_score)
    print('scene_score = %.2f / %d' % (scene_score, total_scene_score))
    print('solve_line_score = %.2f / %d, solve_msg_score = %.2f / %d' % (solve_line_score, total_solve_score, solve_msg_score, total_solve_score))


def generate_no_ml():
    """根据所有的文章生成报错信息、场景信息和解决方法信息，并存放起来"""
    __cnt = 0
    # 从文件中读取标题列表，文章正文以及标注好的数据
    article_dict = load_articles()
    # 从文件中获取到错误、场景和解决方法的信息
    all_err_msgs = dict()
    all_scenes = dict()
    all_solve_lines = dict()
    all_solve_msgs = dict()
    for aid in article_dict:
        if 100000 <= aid < 200000:
            err_msg = get_err_msg(article_dict[aid].title, article_dict[aid].text)
            all_err_msgs[aid] = err_msg
            if all_err_msgs[aid] != str():
                all_scenes[aid] = get_scene_msg(article_dict[aid].title, article_dict[aid].text, all_err_msgs[aid])
                all_solve_lines[aid], all_solve_msgs[aid] = get_solve_msg(article_dict[aid].text)
            __cnt += 1
            if __cnt % 100 == 0:
                print('__cnt = %d, aid = %d' % (__cnt, aid))
    # 将生成好的数据存放进数据库中
    db_col = db_connect()
    insert_data(db_col, all_err_msgs, all_scenes, all_solve_msgs)


def main():
    if MODE == 'Train' and (not ML):
        train_no_ml()
    elif MODE == 'Generate' and (not ML):
        generate_no_ml()


if __name__ == '__main__':
    main()

from settings import *
import pymongo


def db_connect():
    conn = pymongo.MongoClient(DB_SERVER)
    db = conn[DB_NAME]
    db_col = db[DB_COLLECTION]
    # db_col.drop()
    return db_col


def insert_data(db_col, err_msgs, scenes, solve_msgs):
    data_all = []
    for aid in err_msgs:
        if err_msgs[aid] != "":
            data_1article = {"_id": aid,
                             "err_msg": err_msgs[aid],
                             "scene": scenes[aid],
                             "solve": solve_msgs[aid]}
            data_all.append(data_1article)
    db_col.insert_many(data_all)

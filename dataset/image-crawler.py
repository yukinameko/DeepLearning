import os
import shutil
import glob
import random
import numpy as np
from icrawler.builtin import GoogleImageCrawler

def imagecrawl(searchwords, imagenum, title):
    """ 作品の画像を収集して保存する。
    収集した画像の80%をtrainに、20%をvalidationに入れる。

    Parameters
    ----------
    searchwords: 検索クエリ. str型.
    imagenum: 画像の枚数。int型.
    title: 作品タイトル。train/validation 配下のディレクトリを指定。str型.
    """
    # ディレクトリのチェック
    train_dir = './train/' + title + '/'
    dircheck(train_dir)
    valid_dir = './validation/' + title + '/'
    dircheck(valid_dir)

    max_idx = max_file_idx(train_dir, valid_dir)

    # クローラで画像のダウンロード
    crawler = GoogleImageCrawler(storage={"root_dir": "tmp"})
    crawler.crawl(
        keyword=searchwords,
        max_num=imagenum,
        file_idx_offset=max_idx
        )

    # ダウンロードしたファイルをtrain, validationに分割し移動
    image_list = glob.glob('./tmp/*')
    random.shuffle(image_list)
    train_list, valid_list = np.split(
                            np.array(image_list),
                            [int(len(image_list)*0.8)]
                            )
    train_list = list(train_list)
    valid_list = list(valid_list)

    for i in train_list:
        shutil.move(i, train_dir)
    for i in valid_list:
        shutil.move(i, valid_dir)


def dircheck(dirpath):
    """ そのディレクトリがあるかチェックして無ければ作成する """
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)


def max_file_idx(train_dir, valid_dir):
    """ offset設定のためファイルのインデックスの最大値を求める。 """
    max_idx = 0
    image_files = os.listdir(train_dir)
    image_files.extend(os.listdir(valid_dir))
    for filename in image_files:
        try:
            idx = int(os.path.splitext(filename)[0])
        except ValueError:
            continue
        if idx > max_idx:
            max_idx = idx
    return max_idx

def main():
    image_num = 100
    # imagecrawl('ご注文はうさぎですか？', imagenum=image_num, title='gochiusa')
    # imagecrawl('ご注文はうさぎですか？ 漫画', imagenum=image_num, title='gochiusa')
    # imagecrawl('ご注文はうさぎですか？ アニメ', imagenum=image_num, title='gochiusa')
    imagecrawl('ごちうさ アニメ　ココア', imagenum=image_num, title='cocoa')
    imagecrawl('ごちうさ アニメ　チノ', imagenum=image_num, title='chino')
    imagecrawl('ごちうさ アニメ　リゼ', imagenum=image_num, title='rize')
    imagecrawl('ごちうさ アニメ　シャロ', imagenum=image_num, title='syaro')
    imagecrawl('ごちうさ アニメ　千夜', imagenum=image_num, title='chiya')


if __name__ == '__main__':
    main()
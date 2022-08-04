import json
import argparse
import os
import operator


def get_all_unique_cats(data_folder, base_folder):
    json_file = os.path.join(data_folder, base_folder, 'meta_Software.json')

    pros_meta = []
    with open(json_file, 'r') as f:
        for pro_line in f:
            pro_meta = json.loads(pro_line)
            pros_meta.append(pro_meta)

    cats_count = dict()

    def normalize(cat_list):
        cat_list = cat_list[:3]
        cat_list = [cat.replace('&amp;', '&') for cat in cat_list]
        return cat_list

    for pro_meta in pros_meta:
        cat_list = pro_meta['category']
        id = pro_meta['asin']
        if len(cat_list) == 3:
            cat_list = normalize(cat_list)
            cat2 = cat_list[1]
            cat3 = cat_list[2]
            (sub_cats, c) = cats_count.get(cat2, ({}, 0))
            sub_cats[cat3] = sub_cats.get(cat3, set())
            sub_cats[cat3].add(id)
            cats_count[cat2] = (sub_cats, c + 1)

    unique_cat_file = os.path.join(data_folder, base_folder, 'unique_cat.txt')

    with open(unique_cat_file, 'w') as f:
        sorted_cats_count = sorted(cats_count.items(), key=lambda x: x[1][1], reverse=True)
        for cate, count in sorted_cats_count:
            f.write("%d -> %s\n" % (count[1], cate))

            sorted_cats_count_2 = sorted(count[0].items(), key=lambda x: len(x[1]), reverse=True)
            for cate2, count2 in sorted_cats_count_2:
                f.write("\t%d -> %s\n" % (len(count2), cate2))


parser = argparse.ArgumentParser(description='Prepare the Amazon holdout dataset')
parser.add_argument('data_folder', help='data folder path')
parser.add_argument('mode', choices=['holdout'], help='data prepare mode')

args = parser.parse_args()

data_folder = args.data_folder
mode = args.mode

base_folder = 'amazon/software'

get_all_unique_cats(data_folder, base_folder)

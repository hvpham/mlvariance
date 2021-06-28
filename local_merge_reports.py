import os
import argparse

parser = argparse.ArgumentParser(description='Merge CIFAR100 report')
parser.add_argument('result_folder', help='result folder')
parser.add_argument('mode', choices=['holdout'], help='the mode')

args = parser.parse_args()

result_folder = args.result_folder
mode = args.mode

# holdout_classes = ['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman', 'caterpillar', 'mushroom', 'porcupine', 'ray']
#holdout_classes = ['boy-girl', 'boy-man', 'girl-woman', 'man-woman', 'caterpillar', 'mushroom', 'porcupine', 'ray']
# holdout_classes = ['caterpillar', 'mushroom', 'porcupine', 'ray']

aquatic_mammals_list = ['beaver', 'dolphin', 'otter', 'seal', 'whale']
fish_list = ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']
#fish_list = ['aquarium_fish', 'flatfish', 'shark', 'trout']
flower_list = ['orchid', 'poppy', 'rose', 'sunflower', 'tulip']
fruit_and_vegetables_list = ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper']
#fruit_and_vegetables_list = ['apple', 'orange', 'pear', 'sweet_pepper']
insects_list = ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach']
#insects_list = ['bee', 'beetle', 'butterfly', 'cockroach']
medium_mammals_list = ['fox', 'porcupine', 'possum', 'raccoon', 'skunk']
#medium_mammals_list = ['fox', 'possum', 'raccoon', 'skunk']
people_list = ['baby', 'boy', 'girl', 'man', 'woman']
#people_list = ['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman']
reptiles_list = ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle']
small_mammals_list = ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']
trees_list = ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree']

holdout_class_list = []
holdout_class_list.extend(aquatic_mammals_list)
holdout_class_list.extend(fish_list)
holdout_class_list.extend(flower_list)
holdout_class_list.extend(fruit_and_vegetables_list)
holdout_class_list.extend(insects_list)
holdout_class_list.extend(medium_mammals_list)
holdout_class_list.extend(people_list)
holdout_class_list.extend(reptiles_list)
holdout_class_list.extend(small_mammals_list)
holdout_class_list.extend(trees_list)


def merge_ranks():
    overall_file_path = os.path.join(result_folder, 'cifar100', 'overall_holdout_rank.csv')
    overall_file_writer = open(overall_file_path, 'w')

    NO_COLS = 13
    NO_METRICS = 4

    overall_file_writer.write(",,")
    overall_file_writer.write(",test" + "," * (NO_METRICS * NO_COLS - 1))
    overall_file_writer.write(",train" + "," * (NO_METRICS * NO_COLS - 1))
    overall_file_writer.write(",val" + "," * (NO_METRICS * NO_COLS - 1))
    overall_file_writer.write("\n")

    overall_file_writer.write(",,")
    for i in range(3):
        overall_file_writer.write(",first_rank" + "," * (NO_COLS - 1))
        overall_file_writer.write(",median_rank" + "," * (NO_COLS - 1))
        overall_file_writer.write(",avg_top_10_rank" + "," * (NO_COLS - 1))
        overall_file_writer.write(",avg_rank" + "," * (NO_COLS - 1))
    overall_file_writer.write("\n")

    overall_file_writer.write("mode,holdout_class,ratio")
    for i in range(3 * NO_METRICS):
        overall_file_writer.write(",no_model,avg_ground_conf,std_ground_conf,avg_max_conf,std_max_conf,std_pre_labels,num_pre_labels,min_g_conf,max_g_conf,median_g_conf,min_m_conf,max_m_conf,median_m_conf")
    overall_file_writer.write("\n")

    for i in range(len(holdout_class_list)):
        holdout_class = holdout_class_list[i]
        for a in range(11):
            # for a in [0,3,6,9]:
            # for a in [1,2,4,5,7,8]:
            # for a in [0,9]:
            individual_overall_file_path = os.path.join(result_folder, 'cifar100', 'cifar100-%s_%s_%d' % (mode, holdout_class, a), 'overall_holdout_rank.csv')
            with open(individual_overall_file_path, 'r') as f:
                line = f.readline()
            overall_file_writer.write(line)
            overall_file_writer.write('\n')

    overall_file_writer.close()


def merge_auc():
    overall_file_path = os.path.join(result_folder, 'cifar100', 'overall_holdout_auc.csv')
    overall_file_writer = open(overall_file_path, 'w')

    NO_COLS = 11

    overall_file_writer.write(",,")
    overall_file_writer.write(",test" + "," * (NO_COLS - 1))
    overall_file_writer.write(",train" + "," * (NO_COLS - 1))
    overall_file_writer.write(",val" + "," * (NO_COLS - 1))
    overall_file_writer.write("\n")

    overall_file_writer.write("mode,holdout_class,ratio")
    for i in range(3):
        overall_file_writer.write(",no_model,avg_ground_conf,std_ground_conf,avg_max_conf,std_max_conf,std_pre_labels,num_pre_labels,min_g_conf,max_g_conf,median_g_conf,min_m_conf,max_m_conf,median_m_conf")
    overall_file_writer.write("\n")

    for i in range(len(holdout_class_list)):
        holdout_class = holdout_class_list[i]
        for a in range(11):
            individual_overall_file_path = os.path.join(result_folder, 'cifar100', 'cifar100-%s_%s_%d' % (mode, holdout_class, a), 'overall_holdout_auc.csv')
            with open(individual_overall_file_path, 'r') as f:
                line = f.readline()
            overall_file_writer.write(line)
            overall_file_writer.write('\n')

    overall_file_writer.close()


def merge_n_models_auc():
    overall_file_path = os.path.join(result_folder, 'cifar100', 'overall_holdout_auc_no_models.csv')
    overall_file_writer = open(overall_file_path, 'w')

    NO_SPLITS = 10

    overall_file_writer.write("mode,holdout_class,ratio")
    for i in range(1, NO_SPLITS + 1):
        overall_file_writer.write(",n_models_split_%d,no_model_%d,avg_ground_conf_%d,std_ground_conf_%d,avg_max_conf_%d,std_max_conf_%d,std_pre_labels_%d,num_pre_labels_%d" % (i, i, i, i, i, i, i, i))
    overall_file_writer.write(",single_model,min_g_conf,max_g_conf,median_g_conf,min_m_conf,max_m_conf,median_m_conf")
    overall_file_writer.write("\n")

    for i in range(len(holdout_class_list)):
        holdout_class = holdout_class_list[i]
        for a in range(11):
            individual_overall_file_path = os.path.join(result_folder, 'cifar100', 'cifar100-%s_%s_%d' % (mode, holdout_class, a), 'overall_holdout_auc_no_models.csv')
            with open(individual_overall_file_path, 'r') as f:
                line = f.readline()
            overall_file_writer.write(line)
            overall_file_writer.write('\n')

    overall_file_writer.close()


# merge_ranks()
# merge_auc()
# merge_n_models_auc()

import os
import argparse

parser = argparse.ArgumentParser(description='Merge CIFAR100 report')
parser.add_argument('result_folder', help='result folder')
parser.add_argument('mode', choices=['holdout'], help='the mode')

args = parser.parse_args()

result_folder = args.result_folder
mode = args.mode

holdout_classes = ['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman', 'caterpillar', 'mushroom', 'porcupine', 'ray']
#holdout_classes = ['caterpillar', 'mushroom', 'porcupine', 'ray']

def merge_ranks():
    overall_file_path = os.path.join(result_folder, 'cifar100', 'overall_holdout_rank.csv')
    overall_file_writer = open(overall_file_path,'w')

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

    for i in range(len(holdout_classes)):
        holdout_class = holdout_classes[i]
        for a in range(11):
        #for a in [0,3,6,9]:
        #for a in [1,2,4,5,7,8]:
        #for a in [0,9]:
            individual_overall_file_path = os.path.join(result_folder, 'cifar100', 'cifar100-%s_%s_%d' % (mode,holdout_class,a), 'overall_holdout_rank.csv')
            with open(individual_overall_file_path, 'r') as f:
                line = f.readline()
            overall_file_writer.write(line)
            overall_file_writer.write('\n')

    overall_file_writer.close()

def merge_auc():
    overall_file_path = os.path.join(result_folder, 'cifar100', 'overall_holdout_auc.csv')
    overall_file_writer = open(overall_file_path,'w')

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

    for i in range(len(holdout_classes)):
        holdout_class = holdout_classes[i]
        for a in range(11):
            individual_overall_file_path = os.path.join(result_folder, 'cifar100', 'cifar100-%s_%s_%d' % (mode,holdout_class,a), 'overall_holdout_auc.csv')
            with open(individual_overall_file_path, 'r') as f:
                line = f.readline()
            overall_file_writer.write(line)
            overall_file_writer.write('\n')

    overall_file_writer.close()

merge_ranks()
merge_auc()

        




        
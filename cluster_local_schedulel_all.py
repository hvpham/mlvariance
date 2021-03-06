import os
import shutil
import tempfile
import argparse
import math


def write_run_configs(runs, path):
    f = open(path, "w")
    for run in runs:
        f.write("%s\n%s\n%s\n" % (run[0], run[1], run[2]))
    f.close()


def schedule_train_job(data_folder, result_folder, experiment_name, runs, GPU):

    # Create project folder
    project_folder = './cluster_source_folder'
    os.makedirs(project_folder, exist_ok=True)
    shutil.copy('cluster_single_job.py', project_folder)
    shutil.copy('cluster_single_run.sh', project_folder)
    shutil.copy('cluster_single_cifar100.py', project_folder)
    shutil.copy('cluster_single_cifar10.py', project_folder)
    shutil.copy('cluster_single_detail_analysis_cifar100.py', project_folder)
    shutil.copy('cluster_single_detail_analysis_cifar100_html.py', project_folder)
    shutil.copy('cluster_single_saliency_cifar100.py', project_folder)
    shutil.copy('cluster_single_evaluate_cifar100.py', project_folder)
    shutil.copy('sort_script.txt', project_folder)
    #shutil.copy('cluster_single_test.py', project_folder)
    shutil.copy('resnet.py', project_folder)
    #shutil.copy('utils.py', project_folder)
    write_run_configs(runs, os.path.join(project_folder, 'runs'))


def check_done(path):
    if os.path.isfile(path):
        if os.stat(path).st_size > 0:
            return True
    return False


def list_cifar100_holdout_runs(local_result_folder, no_runs):
    mode = 'holdout'
    #holdout_class_list = ['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman', 'caterpillar', 'mushroom', 'porcupine', 'ray']
    #holdout_class_list = ['caterpillar', 'mushroom', 'porcupine', 'ray']

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

    runs_list = []
    for holdout_class in holdout_class_list:
        # for ratio in [0,3,6,9]:
        # for ratio in [1,2,4,5,7,8,10]:
        for ratio in range(11):
            # for ratio in [10]:
            for run_id in range(no_runs):
                outputs_file = os.path.join(local_result_folder, 'cifar100', 'cifar100-%s_%s_%d' % (mode, holdout_class, ratio), 'outputs_%s' % run_id)
                if not check_done(outputs_file):
                    script = "cluster_single_cifar100.py"
                    args = "%s %s %d %d" % (mode, holdout_class, ratio, run_id)
                    log = "cifar100_%s_%s_%d_%d" % (mode, holdout_class, ratio, run_id)
                    runs_list.append((script, args, log, True))

    return runs_list


def list_cifar100_holdout_evaluate_runs(no_runs):
    mode = 'holdout'
    holdout_classes = ['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman', 'caterpillar', 'mushroom', 'porcupine', 'ray']
    #holdout_classes = ['caterpillar', 'mushroom', 'porcupine', 'ray']

    runs_list = []
    for i in range(len(holdout_classes)):
        holdout_class = holdout_classes[i]
        for ratio in range(11):
            script = "cluster_single_evaluate_cifar100.py"
            args = "%s %s %d %d" % (mode, holdout_class, ratio, no_runs)
            log = "cifar100_evaluate_%s_%s_%d_%d" % (mode, holdout_class, ratio, no_runs)
            runs_list.append((script, args, log, False))

    return runs_list


def list_cifar100_holdout_analysis_runs(no_runs, script_name):
    mode = 'holdout'
    holdout_classes = ['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman', 'caterpillar', 'mushroom', 'porcupine', 'ray']
    holdout_targets = [6, 6, 6, 6, 6, 4, 3, 5, 1]
    #holdout_classes = ['caterpillar', 'mushroom', 'porcupine', 'ray']
    #holdout_targets = [4,3,5,1]

    runs_list = []
    for i in range(len(holdout_classes)):
        holdout_class = holdout_classes[i]
        holdout_target = holdout_targets[i]
        for ratio in range(11):
            # for ratio in [10]:
            script = script_name
            args = "%s %s %d %d %d" % (mode, holdout_class, holdout_target, ratio, no_runs)
            log = "cifar100_analyze_%s_%s_%d_%d_%d" % (mode, holdout_class, holdout_target, ratio, no_runs)
            runs_list.append((script, args, log, False))

    return runs_list


def list_cifar100_holdout_generate_map_runs(local_result_folder, no_runs):
    mode = 'holdout'
    holdout_classes = ['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman', 'caterpillar', 'mushroom', 'porcupine', 'ray']
    #holdout_classes = ['caterpillar', 'mushroom', 'porcupine', 'ray']

    runs_list = []
    for holdout_class in holdout_classes:
        # for ratio in [0,3,6,9]:
        # for ratio in [1,2,4,5,7,8,10]:
        for ratio in range(11):
            # for ratio in [10]:
            for run_id in range(no_runs):
                outputs_file = os.path.join(local_result_folder, 'cifar100', 'cifar100-%s_%s_%d' % (mode, holdout_class, ratio), 'maps_%s' % run_id)
                if not check_done(outputs_file):
                    script = "cluster_single_saliency_cifar100.py"
                    args = "%s %s %d %d" % (mode, holdout_class, ratio, run_id)
                    log = "cifar100_map_%s_%s_%d_%d" % (mode, holdout_class, ratio, run_id)
                    runs_list.append((script, args, log, True))

    return runs_list


def list_cifar10_runs(local_result_folder, no_runs):

    holdout_class = 0

    modes = ['holdout', 'holdout-dup', 'augmentation', 'augmentation-all']

    runs_list = []

    for mode in modes:
        # for ratio in [0,3,6,9]:
        for ratio in [1, 2, 4, 5, 7, 8, 10]:
            for run_id in range(no_runs):
                outputs_file = os.path.join(local_result_folder, 'cifar10', 'cifar10-%s_%d_%d' % (mode, holdout_class, ratio), 'outputs_%s' % run_id)
                if not os.path.isfile(outputs_file):
                    script = "cluster_single_cifar10.py"
                    args = "%s %d %d" % (mode, ratio, run_id)
                    log = "cifar100_%s_%d_%d_%d" % (mode, holdout_class, ratio, run_id)
                    runs_list.append((script, args, log, True))

    return runs_list


parser = argparse.ArgumentParser(description='Schedule experiment runs on the clusters with holdout CIFAR-100 and Resnet18')
parser.add_argument('local_teamdrive_folder', help='local teamdrive folder')

args = parser.parse_args()

local_teamdrive_folder = args.local_teamdrive_folder

NO_RUNS = 25

#runs_list = list_cifar10_runs(os.path.join(local_teamdrive_folder, 'result'), NO_RUNS)

runs_list = list_cifar100_holdout_runs(os.path.join(local_teamdrive_folder, 'result'), NO_RUNS)

#runs_list = list_cifar100_holdout_evaluate_runs(NO_RUNS)

#runs_list = list_cifar100_holdout_analysis_runs(NO_RUNS, "cluster_single_detail_analysis_cifar100.py")

#runs_list = list_cifar100_holdout_generate_map_runs(os.path.join(local_teamdrive_folder, 'result'), NO_RUNS)

#runs_list = list_cifar100_holdout_analysis_runs(NO_RUNS, "cluster_single_detail_analysis_cifar100_html.py")

#MAX_NO_JOBS = 64
#MAX_NO_JOBS = 16
MAX_NO_JOBS = 32

#NO_RUNS_PER_JOB = 1
NO_RUNS_PER_JOB = math.ceil(len(runs_list)/MAX_NO_JOBS)
no_jobs = math.ceil(len(runs_list)/NO_RUNS_PER_JOB)
for job_id in range(no_jobs):
    start = job_id * NO_RUNS_PER_JOB
    end = min((job_id + 1) * NO_RUNS_PER_JOB, len(runs_list))
    runs = runs_list[start:end]

    schedule_train_job("data", "result", 'modelvar-train-rn18-holdout', runs, runs[0][3])

import pandas as pd
import matplotlib.pyplot as plt
import statistics as st

OUT_FOLDER = "result/COMPAS/holdout_result_%s/"

f = open("result/holdout_result.csv","w")
f.write(",,train_accu, train_0_precision, train_0_recall, train_1_precision, train_1_recall")
f.write(",val_accu, val_0_precision, val_0_recall, val_1_precision, val_1_recall")
f.write(",test_accu, test_0_precision, test_0_recall, test_1_precision, test_1_recall")
f.write(",holdout_accu, holdout_0_precision, holdout_0_recall, holdout_1_precision, holdout_1_recall")
f.write(",normal_accu, normal_0_precision, normal_0_recall, normal_1_precision, normal_1_recall")
f.write("\n")

def analyze_experiment(cluster_index, epoch, lr, hidden, f):
    out_folder = OUT_FOLDER % ("%d_%d_%g_%d" % (cluster_index, epoch, lr, hidden))
    csv_out_file = out_folder+"result.csv"

    df = pd.read_csv(csv_out_file)
    
    stats = []
    for column_name in df:
        if column_name == 'Run':
            continue
        data = df[column_name].tolist()
        avg = st.mean(data)
        ma = max(data)
        mi = min(data)
        diff = ma-mi
        std_dev = st.stdev(data)
        stats.append((avg,ma,mi,diff,std_dev))
    
    st_names = ['avg', 'max', 'min', 'diff', 'std_dev']
    for index, st_name in enumerate(st_names):
        f.write("%d_%d_%g_%d" % (cluster_index, epoch, lr, hidden))
        f.write(",%s" % st_name)
        for stat in stats:
            f.write(",%f" % stat[index])
        f.write("\n")
    
    f.write("\n\n")


for cluster_index in range(15):
    analyze_experiment(cluster_index, 10,0.0001,64,f)    
    

#analyze_experiment(10,0.001,64,f)
#analyze_experiment(1000,0.0001,64,f)
#analyze_experiment(1000,0.0001,128,f)

f.close()
        
    


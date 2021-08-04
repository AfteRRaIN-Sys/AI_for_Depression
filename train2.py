#==================================== install fastaudio ====================================
#!pip install fastaudio

from fastai.vision.all import *
from fastaudio.core.all import *
from fastaudio.augment.all import *
from fastaudio.ci import skip_if_ci
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pathlib

#==================================== import labels csv ====================================
master_folder = pathlib.Path(__file__).parent.absolute()
path =  master_folder #path to folder which contains "train_and_validate_all.csv"
df = pd.read_csv(path+"\\train_and_validate_all.csv")
df.head()

cfg = AudioConfig.BasicMelSpectrogram(n_fft=512)
a2s = AudioToSpec.from_cfg(cfg)

#==================================== create datablock ====================================
auds = DataBlock(blocks = (AudioBlock, CategoryBlock),
                 get_x = ColReader("filename" , pref = (master_folder + '\\10sec_all_trainvalid\\')) ,
                 #pref = path to folder which contain all train n validate files
                 splitter=ColSplitter(),
                 batch_tfms = [a2s],
                 get_y = ColReader("category"))

#==================================== dataloader ====================================
dbunch = auds.dataloaders(df, bs=8)
dbunch.show_batch(figsize=(10, 5))

#==================================== create learner ====================================
learn = cnn_learner(dbunch, 
            resnet34,
            n_in=1,  # <- This is the only audio specific modification here
            loss_func=CrossEntropyLossFlat(),
            metrics=[accuracy])

#==================================== train model (transfer learning) ====================================
learn.fine_tune(5)

#==================================== show training result ====================================
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(5,5))

#==================================== start testing model ====================================
test_csv = master_folder+'\\all_test.csv' #path to .csv
trans = {
  "normal" : 0,
  "sad" : 1,
  "sobbing" : 2,
  "sob" : 2
}

df2 = pd.read_csv(test_csv)
con_mat = [[0,0,0],
           [0,0,0],
           [0,0,0]]
file_name = df2['filename'].values.tolist()
ans = df2['category'].values.tolist()
predictions = []

test_folder = master_folder+"\\10sec_all_test\\" #path to folder which contain all test file
for i in range(len(file_name)) :
  print(file_name[i])
  predict = learn.predict(test_folder+file_name[i])
  predictions.append(predict[0])
  con_mat[trans[ans[i]]][trans[predict[0]]]+=1

#==================================== show test result ====================================
con_mat = np.array(con_mat)
s = 0
c = 0
for i in range(len(con_mat)) :
  for j in range(len(con_mat[i])) :
    if i == j : c += con_mat[i][j]
    s += con_mat[i][j]
print(con_mat)
print(f"correct prediction : {c} out of {s}")
print(round(100*c/s,2))

y_true = ans
y_pred = predictions
target_names = ['normal', 'sad', 'sobbing']
labels = ['normal', 'sad', 'sobbing']

confusion_matrix(y_true, y_pred, labels=labels)
class_rep = classification_report(y_true, y_pred, labels=labels,target_names=target_names)

#==================================== save test result ====================================

f= open("class_rep.txt","w+")
f.write(class_rep)
f.close()


con_mat = pd.DataFrame(con_mat, index=target_names, columns=target_names)
sns.heatmap(con_mat/np.sum(con_mat), annot=True, fmt='.2%', cmap='Blues')
plt.plot()

results_path = 'confusion_matrix.png'
#print(results_path)
plt.savefig(results_path)
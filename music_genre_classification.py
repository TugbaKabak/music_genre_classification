import librosa.display
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression

Muzik_turleri = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7,
                 'reggae': 8, 'rock': 9}

train_muzikler = './Ses dosyaları/Train'
test_muzikler = './Ses dosyaları/Test'

train_ozellik_listesi = []
test_ozellik_listesi = []

rootdir = os.getcwd()

for subdir in os.listdir(rootdir + "/" + train_muzikler):
    path = rootdir + "/" + train_muzikler
    if os.path.isdir(path + "/" + subdir):
        for file in os.listdir(path + "/" + subdir):
            if os.path.isfile(path + "/" + subdir + "/" + file):
                x, sr = librosa.load(path + "/" + subdir + "/" + file)
                mfccs = librosa.feature.mfcc(x, sr=sr)
                ozellik_listesi = []

                for e in mfccs:
                    ozellik_listesi.append(np.mean(e))

                zero_crossings = librosa.feature.zero_crossing_rate(x)
                ozellik_listesi.append(np.mean(zero_crossings))

                spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)
                ozellik_listesi.append(np.mean(spectral_centroids))

                spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)
                ozellik_listesi.append(np.mean(spectral_rolloff))

                hop_length = 512
                chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
                ozellik_listesi.append(np.mean(chromagram))

                spec_bw = librosa.feature.spectral_bandwidth(x, sr=sr)
                ozellik_listesi.append(np.mean(spec_bw))

                rms = librosa.feature.rms(x)
                ozellik_listesi.append(np.mean(rms))

                print(len(ozellik_listesi), file)
                train_ozellik_listesi.append(ozellik_listesi)

scaler = StandardScaler()
train_x = scaler.fit_transform(np.array(train_ozellik_listesi, dtype=float))

rootdir = os.getcwd()
for subdir in os.listdir(rootdir + "/" + test_muzikler):
    path = rootdir + "/" + test_muzikler
    if os.path.isdir(path + "/" + subdir):
        for file in os.listdir(path + "/" + subdir):
            if os.path.isfile(path + "/" + subdir + "/" + file):
                x, sr = librosa.load(path + "/" + subdir + "/" + file)
                mfccs = librosa.feature.mfcc(x, sr=sr)
                ozellik_listesi = []
                for e in mfccs:
                    ozellik_listesi.append(np.mean(e))

                zero_crossings = librosa.feature.zero_crossing_rate(x)
                ozellik_listesi.append(np.mean(zero_crossings))

                spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)
                ozellik_listesi.append(np.mean(spectral_centroids))

                spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)
                ozellik_listesi.append(np.mean(spectral_rolloff))

                hop_length = 512
                chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
                ozellik_listesi.append(np.mean(chromagram))

                spec_bw = librosa.feature.spectral_bandwidth(x, sr=sr)
                ozellik_listesi.append(np.mean(spec_bw))

                rms = librosa.feature.rms(x)
                ozellik_listesi.append(np.mean(rms))

                print(len(ozellik_listesi), file)
                test_ozellik_listesi.append(ozellik_listesi)

test_x = scaler.fit_transform(np.array(test_ozellik_listesi, dtype=float))

Y = []
for x in range(700):
    if x <= 69:
        Y.extend([1])
    if x > 69 and x <= 139:
        Y.extend([2])
    if x > 139 and x <= 209:
        Y.extend([3])
    if x > 209 and x <= 279:
        Y.extend([4])
    if x > 279 and x <= 349:
        Y.extend([5])
    if x > 349 and x <= 419:
        Y.extend([6])
    if x > 419 and x <= 489:
        Y.extend([7])
    if x > 489 and x <= 559:
        Y.extend([8])
    if x > 559 and x <= 629:
        Y.extend([9])
    if x > 629 and x <= 699:
        Y.extend([10])

test_Y = []
for x in range(300):
    if x <= 29:
        test_Y.extend([1])
    if x > 29 and x <= 59:
        test_Y.extend([2])
    if x > 59 and x <= 89:
        test_Y.extend([3])
    if x > 89 and x <= 119:
        test_Y.extend([4])
    if x > 119 and x <= 149:
        test_Y.extend([5])
    if x > 149 and x <= 179:
        test_Y.extend([6])
    if x > 179 and x <= 209:
        test_Y.extend([7])
    if x > 209 and x <= 239:
        test_Y.extend([8])
    if x > 239 and x <= 269:
        test_Y.extend([9])
    if x > 269 and x <= 299:
        test_Y.extend([10])

classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

print("Random Forest ile sınıflandırma\n")
clf = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=1)
clf.fit(train_x, Y)
ex = clf.predict_proba(test_x)
print(ex)
ex2 = clf.predict(test_x)
print(ex2)
conf_matrix = confusion_matrix(y_true=test_Y, y_pred=ex2)
print(conf_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes)
disp.plot()
plt.show()

print("Naive Bayes ile sınıflandırma\n")
clf = GaussianNB()
clf.fit(train_x, Y)
ex = clf.predict_proba(test_x)
print(ex)
ex2 = clf.predict(test_x)
print(ex2)
conf_matrix = confusion_matrix(y_true=test_Y, y_pred=ex2)
print(conf_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes)
disp.plot()
plt.show()

print("KNN ile sınıflandırma\n")
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(train_x, Y)
ex = neigh.predict_proba(test_x)
print(ex)
ex2 = neigh.predict(test_x)
print(ex2)
conf_matrix = confusion_matrix(y_true=test_Y, y_pred=ex2)
print(conf_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes)
disp.plot()
plt.show()

print("SVM ile sınıflandırma\n")
clf = svm.SVC()
clf.fit(train_x, Y)
ex2 = clf.predict(test_x)
print(ex2)
conf_matrix = confusion_matrix(y_true=test_Y, y_pred=ex2)
print(conf_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes)
disp.plot()
plt.show()

print("Logistic Regression ile sınıflandırma\n")
clf = LogisticRegression(random_state=0).fit(train_x, Y)
ex = clf.predict_proba(test_x)
print(ex)
ex2 = clf.predict(test_x)
print(ex2)
conf_matrix = confusion_matrix(y_true=test_Y, y_pred=ex2)
print(conf_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes)
disp.plot()
plt.show()
import numpy as np, pandas as pd
import re
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import math

def preprocessing():
    pre2=[]
    dataset = pd.read_csv('datasetDiagnosa.csv').values.tolist()
    npds = np.array(dataset)
    cvtnp = np.ndarray.flatten(npds)
    dsetcvt = list(cvtnp)
    dsbaru =list()
    for dokumen in dsetcvt:
        dokumen = re.sub(r'(,)|(\(([^()]+)\))', '',dokumen)
        dsbaru.append(dokumen)
    cvtdsbaru = np.array(dsbaru).reshape(100,2)
    factory = StopWordRemoverFactory()
    stopwords = factory.create_stop_word_remover()
    for i in range(100):
        # casefolding
        pre11 = cvtdsbaru[i,0].lower()
        #remove stopwords
        pre12 = stopwords.remove(pre11)
        #tokenization
        pre13 = word_tokenize(pre12)
        pre2.append(pre13)
    dsfinal = list()
    for i in range(100):
        arr1 = list(np.array([pre2[i],cvtdsbaru[i,1]]))
        dsfinal.append(arr1)
    print('hasil setelah preprocessing:')
    print(dsfinal)
    return dsfinal

def split_train_test():
    dsprepro = preprocessing()
    np_dsprepro = np.array(dsprepro)
    dstrain = np_dsprepro[:93,:]
    dstest = np_dsprepro[93:100,:]
    return dstrain, dstest

def term_frequency(termtest,doktrain):
    tf = doktrain.count(termtest)
    return tf

def cari_tf():
    tfterm = list(); tfdok = dict()
    train,test = split_train_test()
    # for i in range(len(test)):
    dok_test = test[1,0]
    for k in dok_test:
        del tfterm[:]
        for j in range(len(train)):
            dok_train = train[j, 0]
            tf = term_frequency(k,dok_train)
            tfterm.append(tf)
            # print('nilai tf %s pada dokumen ke-%s : %s'%(str(k),str(j),str(tf)))
        tfdok[str(k)] = tfterm
    print('hasil TF:')
    print(tfdok)
    return tfdok, train

def cari_df(tf_antar_dok):
    df = 0
    for i in tf_antar_dok:
        if i > 0:
            df += i
    return df

def cari_idf():
    idflist = dict()
    tf_antar_dok, train = cari_tf()
    for key,value in tf_antar_dok.items():
        df = cari_df(tf_antar_dok[key])
        idf = math.log((len(train)/df),10) + 1
        idflist[str(key)] = idf
    print('hasil IDF:')
    print(idflist)
    return tf_antar_dok, idflist

def cari_tfIdf(split_train_test):
    tf_final, idf_final = cari_idf()
    weight = 0; dok_weight = list()
    key_tf = [i for i in tf_final.keys()]
    for i in range(len(tf_final[key_tf[0]])):
        for key,value in idf_final.items():
            weight += tf_final[key][i]*value
        dok_weight.append(weight)
    print(dok_weight)
    idx_relate_dok = max(dok_weight)
    dstrain,dstest = split_train_test
    relate_dok = dstrain[int(idx_relate_dok),1]
    return relate_dok


if __name__ == '__main__':
    dok_sesuai = cari_tfIdf(split_train_test())
    print('jadi penyakit yang sesuai dengan diagnosa adalah: %s'%(dok_sesuai))
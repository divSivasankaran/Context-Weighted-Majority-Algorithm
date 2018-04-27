# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from scipy.spatial import distance
from sklearn import preprocessing
import numpy as np
import os
import VGGFace as FaceVerifier
import pickle

f_features = pickle.load(open("../input/vgg_features_pie_full.pickle", "rb"))
v_features = pickle.load(open("../input/voice_features_full.pickle", "rb"))
pubfig_features = pickle.load(open("../input/vgg_features_pubfig_cropped.pickle", "rb"))

pubfig_trained_files = []

e = [0, 18, 50, 82, 114,
     146,
     178,
     210,
     242,
     274,
     306,
     338,
     370,
     402,
     434,
     466,
     498,
     289
     ]
p = [0, 2,
     6,
     10,
     14,
     18,
     22,
     26,
     30,
     34,
     38,
     42,
     46,
     50,
     54,
     58,
     62,
     66,
     70,
     74,
     78,
     82,
     86,
     90,
     94,
     98,
     102,
     106,
     110,
     114,
     118,
     122,
     126,
     289]
f = [0, 16,
     17,
     18,
     19,
     48,
     49,
     50,
     51,
     80,
     81,
     82,
     83,
     112,
     113,
     114,
     115,
     289]


def getpubfigfiles(data, mode, context):
    lst = []
    files = []
    ids = []
    t = 0
    imposter = 0
    genuine = 0
    cset = dict()
    print("data", len(data))
    for line in data:
        ID, f, c, train, claim, nl = line.strip().split(",")
        if nl not in cset.keys():
            cset[nl] = 0
        cset[nl] += 1
        if train == "1":
            t += 1
        # else:
        #            if claim == ID:
        #                genuine += 1
        #            else:
        #                imposter += 1
        if mode == "train":
            if int(nl) not in context or train != "1":
                continue
            lst.append(line)
            files.append(pubfig_features[ID][f])
            ids.append(ID)
            pubfig_trained_files.append(ID + f)
        else:
            #            if train!="0":
            #                continue
            if claim == ID:
                genuine += 1
            else:
                imposter += 1
            if ID + f not in pubfig_trained_files:
                lst.append(line)
                files.append(pubfig_features[ID][f])
                ids.append(claim)
    print "train, imposter, genuine: ", t, imposter, genuine
    return lst, np.array(files), ids


def getfiles(data, mode, modality, context, tk=-1):
    lst = []
    files = []
    ids = []
    cset = set()
    print(len(data))
    for line in data:
        line = line.strip()
        ID, f, v, c, k, train, claim = line.split(",")
        if modality == "Voice":
            c = v.split(".")[0].split("_")[-2]
            if mode == "train":
                if c not in context or train != "1":
                    continue
                if tk != -1 and tk != int(k):
                    continue
                lst.append(line)
                files.append(v_features[ID][v])
                ids.append(ID)
            else:
                if tk != -1 and tk != int(k):
                    continue
                lst.append(line)
                files.append(v_features[ID][v])
                ids.append(claim)
        else:
            c = f.split(".")[0].split("_")[-2]
            if mode == "train":
                if c not in context:
                    cset.add(c)
                    continue
                if tk != -1 and tk != int(k):
                    continue
                if train == "1":
                    lst.append(line)
                    files.append(f_features[ID][f])
                    ids.append(ID)
            else:
                if train != "0":
                    continue
                if tk != -1 and tk != int(k):
                    continue
                lst.append(line)
                files.append(f_features[ID][f])
                ids.append(claim)
    return lst, np.array(files), ids


def cosine_distance(a, b):
    return distance.cosine(a, b) / 2


def main(mode, dataset, k=-1):
    data = []
    if dataset == "pie":
        #       infile = open("Samples_Paired_pie_full.csv","r")
        infile = open("../input/Samples_Paired_5_Fold.csv", "r")
        infile.readline()
        data = infile.readlines()
        infile.close()
    else:
        #       infile = open("samples_paired_pubfig_attr.csv","r")
        infile = open("../input/pubfig_epa.csv", "r")
        infile.readline()
        data = infile.readlines()
        infile.close()
    FV = FaceVerifier.FaceVerifier()
    FV.init()
    # POSE
    lst = []
    train_features = []
    ids = []

    #   lst,train_features,ids = getfiles(data,"train","Face",["2","3"])
    if dataset == "pie":
        lst, train_features, ids = getfiles(data, "train", "Face", ["2", "3", "4"], k)
    else:
        #       lst,train_features,ids = getpubfigfiles(data,"train",["1","4","5","7"])
        #       lst,train_features,ids = getpubfigfiles(data,"train",["0","19","46","61","69","86"])
        lst, train_features, ids = getpubfigfiles(data, "train", p)

    print("Training ", len(train_features))
    if mode == "mean":
        p_clf = FV.generateMeanTemplate(train_features, ids)
    if mode == "svm":
        p_clf = FV.generateTemplate_SVM(train_features, ids)
    if mode == "pca":
        p_W, p_m, p_clf = FV.generateTemplate_PCA(train_features, ids)
    if mode == "nb":
        p_clf = FV.generateTemplate_NB(train_features, ids)

    # EXP
    if dataset == "pie":
        lst, train_features, ids = getfiles(data, "train", "Face", ["0", "1", "5"], k)
    else:
        #       lst,train_features,ids = getpubfigfiles(data,"train",["7","4","0","5"])
        #       lst,train_features,ids = getpubfigfiles(data,"train",["0","2","16","18","19","20","21","23","47","50","58","62","67","69","76","77","96"])
        lst, train_features, ids = getpubfigfiles(data, "train", e)  # ["0","34","40"])
    print("Training ", len(train_features))

    if mode == "mean":
        e_clf = FV.generateMeanTemplate(train_features, ids)
    if mode == "svm":
        e_clf = FV.generateTemplate_SVM(train_features, ids)
    if mode == "pca":
        e_W, e_m, e_clf = FV.generateTemplate_PCA(train_features, ids)
    if mode == "nb":
        e_clf = FV.generateTemplate_NB(train_features, ids)

        # BOTH
    if dataset == "pie":
        lst, train_features, ids = getfiles(data, "train", "Face", ["0", "1", "2", "3", "4", "5"], k)
    else:
        #       lst,train_features,ids = getpubfigfiles(data,"train",["4","3","2","5"])
        #       lst,train_features,ids = getpubfigfiles(data,"train",["0","3","4","19","42","52","69"])
        lst, train_features, ids = getpubfigfiles(data, "train", f)  # ["0","32","34","40"])

    print("Training ", len(train_features))

    if mode == "mean":
        a_clf = FV.generateMeanTemplate(train_features, ids)
    if mode == "svm":
        a_clf = FV.generateTemplate_SVM(train_features, ids)
    if mode == "pca":
        a_W, a_m, a_clf = FV.generateTemplate_PCA(train_features, ids)
    if mode == "nb":
        a_clf = FV.generateTemplate_NB(train_features, ids)

        # TESTING POSE
    threshold = 0.1
    print("threshold", threshold)
    if dataset == "pie":
        lst, test_features, ids = getfiles(data, "test", "Face", ["0", "1", "2", "3", "4", "5"], k)
    else:
        l, t, i = getpubfigfiles(data, "test", f)  # ["0,","1","2","3","5","6","7"])
        if mode == "mean":
            id_to_delete = (set(a_clf.keys()) & set(p_clf.keys()) & set(e_clf.keys())) ^ set(i)
            print(id_to_delete)
        else:
            id_to_delete = (set(a_clf.classes_) & set(p_clf.classes_) & set(e_clf.classes_)) ^ set(i)
            print(id_to_delete)
        print("Testing before ", len(t))
        lst = []
        test_features = []
        ids = []
        for j in range(0, len(i)):
            if i[j] not in id_to_delete:
                lst.append(l[j])
                test_features.append(t[j])
                ids.append(i[j])

    print("Testing ", len(test_features))
    p_result = []
    e_result = []
    a_result = []

    if mode == "pca":
        p_test_features = []
        e_test_features = []
        a_test_features = []
        for i in range(0, len(ids)):
            p_test_features.append(np.dot(p_W, test_features[i] - p_m))
            e_test_features.append(np.dot(e_W, test_features[i] - e_m))
            a_test_features.append(np.dot(a_W, test_features[i] - a_m))
        for i in range(0, len(ids)):
            p_result.append(cosine_distance(p_clf[ids[i]], p_test_features[i]))
            e_result.append(cosine_distance(e_clf[ids[i]], e_test_features[i]))
            a_result.append(cosine_distance(a_clf[ids[i]], a_test_features[i]))

    if mode == "mean":
        for i in range(0, len(ids)):
            p_result.append(cosine_distance(p_clf[ids[i]], test_features[i]))
            e_result.append(cosine_distance(e_clf[ids[i]], test_features[i]))
            a_result.append(cosine_distance(a_clf[ids[i]], test_features[i]))
    if mode == "svm" or mode == "nb":
        p_classes = list(p_clf.classes_)
        e_classes = list(e_clf.classes_)
        a_classes = list(a_clf.classes_)
        p_result_ = p_clf.predict_proba(test_features)
        e_result_ = e_clf.predict_proba(test_features)
        a_result_ = a_clf.predict_proba(test_features)
        for i in range(0, len(ids)):
            p_result.append(p_result_[i][p_classes.index(ids[i])])
            e_result.append(e_result_[i][e_classes.index(ids[i])])
            a_result.append(a_result_[i][a_classes.index(ids[i])])

    print("result ", len(p_result))
    p_training_error = 0
    e_training_error = 0
    a_training_error = 0
    filename = os.getcwd() + "/../input/F_" + dataset + "_" + mode + ".csv"
    #   test = a_clf.predict(test_features)
    #   filename = os.getcwd() + "/input/Data_"+dataset+"_"+mode+str(k)+".csv"
    fout = open(filename, "w")
    o_list = []
    for i in range(0, len(ids)):
        orig_id = (lst[i].split(",")[0])
        o_list.append(orig_id)
        #       print(orig_id,test[i],ids[i])
        if dataset == "pubfig":
            context = lst[i].strip().split(",")[-1]
        else:
            context = lst[i].split(",")[1].split("_")[2]
        # print(p_result[i],orig_id,ids[i])
        true_d = int(ids[i] == orig_id)
        if mode == "svm" or mode == "nb":
            p_result[i] = 1 - p_result[i]
            e_result[i] = 1 - e_result[i]
            a_result[i] = 1 - a_result[i]

        # FAR
        if orig_id != ids[i] and p_result[i] <= threshold:
            p_training_error += 1
        # FRR
        if orig_id == ids[i] and p_result[i] >= threshold:
            p_training_error += 1

        if orig_id != ids[i] and e_result[i] <= threshold:
            e_training_error += 1
            # FRR
        if orig_id == ids[i] and e_result[i] >= threshold:
            e_training_error += 1

        if orig_id != ids[i] and a_result[i] <= threshold:
            a_training_error += 1
        # FRR
        if orig_id == ids[i] and a_result[i] >= threshold:
            a_training_error += 1
        fout.write((",").join(
            [str(orig_id), str(ids[i]), str(context), str(p_result[i]), str(e_result[i]), str(a_result[i]),
             str(true_d)]))
        fout.write("\n")
    fout.close()
    # newPath = os.getcwd() + "/../../FusionMethods/input/" + filename
    # copyfile(filename,newPath)


    print("pose error: ", p_training_error)
    print("exp error: ", e_training_error)
    print("all error: ", a_training_error)
    if mode == "svm" or mode == "nb":
        print("a_clf score ", a_clf.score(test_features, o_list))
        print("e_clf score ", e_clf.score(test_features, o_list))
        print("p_clf score ", p_clf.score(test_features, o_list))


def exp_FaceVoice(mode, dataset, k=-1):
    data = []
    if dataset == "pie":
        infile = open("../input/Samples_Paired_pie_full.csv", "r")
        infile.readline()
        data = infile.readlines()
        infile.close()
    else:
        infile = open("../input/samples_paired_pubfig.csv", "r")
        infile.readline()
        data = infile.readlines()
        infile.close()
    FV = FaceVerifier.FaceVerifier()
    FV.init()
    # POSE
    lst = []
    train_features = []
    ids = []

    #   lst,train_features,ids = getfiles(data,"train","Face",["2","3"])
    if dataset == "pie":
        lst, train_features, ids = getfiles(data, "train", "Face", ["2", "3", "4"], k)
    else:
        lst, train_features, ids = getpubfigfiles(data, "train", ["0", "4", "6", "7"])

    print("Training ", len(train_features))
    if mode == "mean":
        p_clf = FV.generateMeanTemplate(train_features, ids)
    if mode == "svm":
        p_clf = FV.generateTemplate_SVM(train_features, ids)
    if mode == "pca":
        p_W, p_m, p_clf = FV.generateTemplate_PCA(train_features, ids)

    # EXP
    if dataset == "pie":
        lst, train_features, ids = getfiles(data, "train", "Face", ["0", "1", "5"], k)
    else:
        lst, train_features, ids = getpubfigfiles(data, "train", ["0", "2", "3", "4"])
    print("Training ", len(train_features))

    if mode == "mean":
        e_clf = FV.generateMeanTemplate(train_features, ids)
    if mode == "svm":
        e_clf = FV.generateTemplate_SVM(train_features, ids)
    if mode == "pca":
        e_W, e_m, e_clf = FV.generateTemplate_PCA(train_features, ids)

        # VOICE
    if dataset == "pie":
        lst, train_features, ids = getfiles(data, "train", "Voice", ["0", "1"], k)
    else:
        lst, train_features, ids = getpubfigfiles(data, "train", ["0", "1", "5", "4"])

    print("Voice Training ", train_features.shape)
    # a_clf = FV.generateTemplate_SVM(train_features, ids)


    #   if mode == "svm":
    #       a_clf = FV.generateTemplate_SVM(train_features,ids)
    #   if mode == "mean":
    #       a_clf = FV.generateMeanTemplate(train_features,ids)
    #   if mode == "pca":
    #       a_W,a_m,a_clf = FV.generateTemplate_PCA(train_features,ids)
    # TESTING POSE
    threshold = 0.1
    print("threshold", threshold)
    if dataset == "pie":
        lst, test_features, ids = getfiles(data, "test", "Face", ["0", "1", "2", "3", "4", "5"])
    else:
        lst, test_features, ids = getpubfigfiles(data, "test", ["0,", "1", "2", "3", "5", "6", "7"])

    voice_test_features = []
    for line in lst:
        ID, f, v, c, k, train, claim = line.split(",")
        voice_test_features.append(v_features[ID][v])

    print("voice testing", len(voice_test_features), len(voice_test_features[0]))
    print("Testing ", len(test_features))
    p_result = []
    e_result = []
    a_result = []

    if mode == "mean":
        for i in range(0, len(ids)):
            p_result.append(cosine_distance(p_clf[ids[i]], test_features[i]))
            e_result.append(cosine_distance(e_clf[ids[i]], test_features[i]))
            #           a_result.append(cosine_distance(a_clf[ids[i]],voice_test_features[i]))
    if mode == "svm":
        p_classes = list(p_clf.classes_)
        e_classes = list(e_clf.classes_)
        p_result_ = p_clf.predict_proba(test_features)
        e_result_ = e_clf.predict_proba(test_features)
        #       a_classes = list(a_clf.classes_)
        #       a_result_ = a_clf.predict_proba(voice_test_features)
        for i in range(0, len(ids)):
            p_result.append(p_result_[i][p_classes.index(ids[i])])
            e_result.append(e_result_[i][e_classes.index(ids[i])])
            #           a_result.append(a_result_[i][a_classes.index(ids[i])])
    a_classes = list(a_clf.classes_)
    a_result_ = a_clf.predict_proba(voice_test_features)
    for i in range(0, len(ids)):
        a_result.append(a_result_[i][a_classes.index(ids[i])])

    # #Normalize Scores
    print("before scaling ", min(a_result), max(a_result))
    a = preprocessing.MinMaxScaler()
    a_result = a.fit(a_result).transform(a_result)
    print("after scaling ", min(a_result), max(a_result))

    print("before scaling ", min(p_result), max(p_result))
    p = preprocessing.MinMaxScaler()
    p_result = p.fit(p_result).transform(p_result)
    print("after scaling ", min(p_result), max(p_result))
    print("before scaling ", min(e_result), max(e_result))
    e = preprocessing.MinMaxScaler()
    e_result = e.fit(e_result).transform(e_result)
    print("after scaling ", min(e_result), max(e_result))
    print("result ", len(p_result))
    p_training_error = 0
    e_training_error = 0
    a_training_error = 0
    filename = os.getcwd() + "../input/FV_" + dataset + "_" + mode + ".csv"
    fout = open(filename, "w")
    o_list = []
    for i in range(0, len(ids)):
        orig_id = (lst[i].split(",")[0])
        o_list.append(orig_id)
        #       if dataset == "pubfig":
        #           context = lst[i].split(",")[2]
        #       else:
        context = lst[i].split(",")[3]
        # print(p_result[i],orig_id,ids[i])
        true_d = int(ids[i] == orig_id)
        if mode == "svm" or mode == "nb":
            p_result[i] = 1 - p_result[i]
            e_result[i] = 1 - e_result[i]
        a_result[i] = 1 - a_result[i]

        # FAR
        if orig_id != ids[i] and p_result[i] <= threshold:
            p_training_error += 1
        # FRR
        if orig_id == ids[i] and p_result[i] >= threshold:
            p_training_error += 1

        if orig_id != ids[i] and e_result[i] <= threshold:
            e_training_error += 1
            # FRR
        if orig_id == ids[i] and e_result[i] >= threshold:
            e_training_error += 1

        if orig_id != ids[i] and a_result[i] <= threshold:
            a_training_error += 1
        # FRR
        if orig_id == ids[i] and a_result[i] >= threshold:
            a_training_error += 1
        fout.write((",").join(
            [str(orig_id), str(ids[i]), str(context), str(p_result[i]), str(e_result[i]), str(a_result[i]),
             str(true_d)]))
        fout.write("\n")
    fout.close()

    print("pose error: ", p_training_error)
    print("exp error: ", e_training_error)
    print("all error: ", a_training_error)
    if mode == "svm":
        print("a_clf score ", p_clf.score(test_features, o_list))
        print("e_clf score ", e_clf.score(test_features, o_list))
    print("p_clf score ", a_clf.score(voice_test_features, o_list))


main(mode="mean", dataset="pubfig")

# exp_FaceVoice(mode = "mean", dataset = "pie")


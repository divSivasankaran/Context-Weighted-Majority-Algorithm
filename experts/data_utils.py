import pickle
import numpy as np
import configparser

config = configparser.ConfigParser()
config.read("../config.ini")

#### CURRENTLY ONLY SUPPORTS THE DATASETS USED FOR THE EXPERIMENTS IN THE THESIS - PIE, SITW, PUBFIG

def load_features_of(dataset):
    file = None
    features = None
    if dataset == "pie":
        file = config['DEFAULT']['pie_file']
    elif dataset == "sitw":
        file = config['DEFAULT']['sitw_file']
    elif dataset == "pubfig":
        file = config['DEFAULT']['pubfig_file']
    if file != None:
        print("Loading pickle:", file)
        features = pickle.load(open(file,"rb"))
    return features

def process_input(dataset, line , __train, __context, __k, __full_features):
    k = -1
    train = -1
    c = -1
    if dataset in ["pie", "sitw"]:
        ID, f, v, c, k, train, claim = line.strip().split(",")
        if dataset == "pie":
            file = f
        else:
            file = v
        c = file.split(".")[0].split("_")[-2]
    elif dataset == "pubfig":
        ID, file, co, train, claim, c = line.strip().split(",")

    if __train == bool(int(train)) and ( (int(k) == __k) or __k == -1):
        if __train == True and int(c) in __context:
            return True, ID, __full_features[ID][file]
        elif __train == False:
            return True, claim, __full_features[ID][file]

    return False, -1, None




def load_data(dataset = None, in_file = None, train = True, context = None, k=-1):
    lst = []
    features = []
    ids = []
    if dataset == None or in_file == None or (context == None and train ==True):
        print("Error! Please pass valid arguments to load_data")
        return

    full_features = load_features_of(dataset)
    print("Loading dataset from ",in_file)
    infile = open(in_file, "r")
    infile.readline()
    data = infile.readlines()
    infile.close()

    for line in data:
        include, ID, fl = process_input(dataset, line, train, context, k, full_features)
        if include == True:
            lst.append(line)
            ids.append(ID)
            features.append(fl)

    return lst, np.array(features), ids
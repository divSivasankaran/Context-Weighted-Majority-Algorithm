import data_utils
import ast
import numpy as np
from scipy.spatial import distance
from sklearn import model_selection, svm
from sklearn.naive_bayes import  MultinomialNB
import multiprocessing
import FisherFace as ff

class Expert(object):
    def __init__(self):
        self.method = "mean"
        self.clf = None
        self.templates = None
        self.W = None
        self.train_method = {
            "svm": self.generateTemplate_SVM,
            "mean": self.generateTemplate_MEAN,
            "lda": self.generateTemplate_LDA,
            "pca": self.generateTemplate_PCA,
            "naivebayes": self.generateTemplate_NB
        }
        # self.eval_method = {
        #     "svm": self.eval_SVM,
        #     "mean": self.eval_MEAN
        # }
        self.labels_dict = dict()
        self.reverse_labels = dict()
        self.distance = "cosine"
        self.threshold = 0.1

    def svc_param_selection(self, X, y, nfolds=5):
        param_grid = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 0.1, 1],
                      'C': [0.01, 0.1, 1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100, 1000]}
        cores = multiprocessing.cpu_count()
        print(cores)
        gridsearch = model_selection.GridSearchCV(svm.SVC(C=1), param_grid, n_jobs=cores, cv=nfolds)  # ,verbose = 5)
        gridsearch.fit(X, y)
        gridsearch.best_params_
        return gridsearch.best_params_

    def generateTemplate_SVM(self, features, ids):
        param = self.svc_param_selection(features, ids)
        print("Best SVM Parameters after grid-search :" , param)
        if param["kernel"] == "rbf":
            self.clf = svm.SVC(probability=True, decision_function_shape='ovr', C=param["C"], kernel=param["kernel"],
                          gamma=param["gamma"])
        else:
            self.clf = svm.SVC(probability=True, decision_function_shape='ovr', C=param["C"], kernel=param["kernel"])
        self.clf.fit(features, ids)
        return self.clf

    def generateTemplate_NB(self, features, ids):
        self.clf = MultinomialNB()
        self.clf.fit(features, ids)
        return self.clf

    def getPCAFeatures(self, faces_train, id_train, r):
        W, LL, m = ff.myPCA(faces_train)
        W_e = W[:, :r + 1]
        y = np.empty([id_train.shape[0], r])
        print(r, W_e.shape, y.shape, m.shape)
        # calculate PCA feature for each training image
        for i in range(0, id_train.shape[0]):
            f = faces_train[:, i] - m
            y[i] = np.dot(np.transpose(W_e), f)

        return W_e, y, m


    def generateTemplate_PCA(self, faces_train, id_train):
        K = 30
        self.W, X, self.mean_samples = self.getPCAFeatures(faces_train, id_train, K)

        self.templates = self.generateMeanTemplate(X, id_train)
        return self.W, self.mean_samples, self.templates

    def getLDAFeatures(self, faces_train, id_train, r=90, use_pca=True):
        if use_pca == True:
            W_1, b, m = self.getPCAFeatures(faces_train, id_train, r)
            X = np.transpose(b)
            W_f, Centers, classLabels = ff.myLDA(X, id_train)
            # calculate LDA feature for each training image
            q = np.transpose(np.dot(np.transpose(W_f), X))

            return np.dot(np.transpose(W_f), np.transpose(W_1)), q, m
        else:
            m = np.mean(faces_train, 1)
            W_f, Centers, classLabels = ff.myLDA(faces_train, id_train)
            # calculate LDA feature for each training image
            q = np.transpose(np.dot(np.transpose(W_f), faces_train))

            return np.dot(np.transpose(W_f)), q, m

    def generateTemplate_LDA(self, faces_train, id_train, r=91, use_pca=True):
        self.W, q, self.mean_samples = self.getLDAFeatures(faces_train, id_train, r, use_pca)

        self.templates = self.generateMeanTemplate(q, id_train)
        return self.W, self.mean_samples, self.templates

    def generateTemplate_MEAN(self, faces, ids):
        self.mean_samples = np.mean(faces,1)
        y = dict()
        for i in range(0, faces.shape[0]):
            if ids[i] not in y.keys():
                y[ids[i]] = []
            y[ids[i]].append(faces[i])
        # create template for each person in the training set
        z = dict()

        for key in y.keys():
            t = np.mean(np.array(y[key]), 0)
            z[key] = t
        self.templates = z
        return self.templates


    def train(self, samples,labels,method = None):
        if method != None:
            self.method = method
        samples = np.array(samples)
        labels = np.array(labels)
        self.train_method[self.method](samples, labels)

    # def predict(self, samples,labels):
    #     return self.eval_method[self.method](samples, labels)

    def accuracy(self, pred, actual):
        res = np.zeros(len(actual))
        res = [1 for i in range(0, len(actual)) if pred[i] == actual[i]]
        return np.mean(res)

    def loss(self, pred, actual):
        res = np.zeros(len(actual))
        res = [1 for i in range(0, len(actual)) if pred[i] != actual[i]]
        return np.sum(res)

    def get_distance(self, X,Y):
        return distance.cdist(X,Y,self.distance).diagonal()

def run(experiment = "SCWMA_UNIMODAL"):
    cfg = data_utils.config
    infile = cfg[experiment]["infile"]
    n_experts = cfg.getint(experiment,"n_experts")
    contexts = ast.literal_eval(cfg[experiment]["contexts"])
    methods = ast.literal_eval(cfg[experiment]["methods"])
    datasets = ast.literal_eval(cfg[experiment]["datasets"])

    ###### TRAIN THE EXPERTS WITH THE CORRESPONDING METHODS FOR THE GIVEN DATASET EXTRACTED TO CREATE INVARIANCE TO THE GIVEN SET OF CONTEXT DIMENSIONS
    experts = [Expert() for i in range(n_experts)]
    for i in range(0, n_experts):
        lst, train_features, train_ids = data_utils.load_data(dataset=datasets[i], context=contexts[i], in_file=infile)
        print("Training Expert %d with %d samples using %s in the following contexts" % (i + 1, len(train_features), methods[i]),contexts[i])
        experts[i].train(samples = train_features,labels= train_ids,method= methods[i])

    ###### GENERATING THE PREDICTIONS FOR EACH EXPERT
    # for i in range(0, n_experts):
    #     lst, test_features, test_ids = data_utils.load_data(dataset=datasets[i],train = False, in_file=infile)
    #     accuracy, loss = experts[i].predict(samples=test_features, labels=test_ids)
    #     print("Loss of Expert %d is %f"%(i+1,loss))


if __name__=="__main__":
    run()

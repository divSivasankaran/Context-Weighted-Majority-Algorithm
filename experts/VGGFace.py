# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 12:22:22 2017

@author: divya
"""

# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
from scipy.spatial import distance
from sklearn import preprocessing
import multiprocessing
import FisherFace as ff
from sklearn import grid_search, svm
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys

caffe_root = 'C:/caffe/'  # set caffe_root_path
sys.path.insert(0, caffe_root + 'python')

import caffe


# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

class FaceVerifier(object):
    def init(self, use_VGGFace=True, use_PCA=False):
        # self.loadVGGFace()
        self.batch_size = 50
        # self.loadVGGFace(cpu_only = True)
        self.use_pca = use_PCA
        self.threshold = 0.4

    def loadVGGFace(self, cpu_only=True):
        if cpu_only:
            caffe.set_mode_cpu()  # Using onyl cpu
        else:
            caffe.set_device(0)  # if we have multiple GPUs, pick the first one
            caffe.set_mode_gpu()

        self.model_def = caffe_root + 'models/VGGFace/VGG_FACE_deploy.prototxt'
        self.model_weights = caffe_root + 'models/VGGFace/VGG_FACE.caffemodel'

        self.net = caffe.Net(self.model_def,  # defines the structure of the model
                             self.model_weights,  # contains the trained weights
                             caffe.TEST)  # use test mode (e.g., don't perform dropout)

        mean = np.array([129.1863, 104.7624, 93.5940])
        print("mean:", mean)

        # create transformer for the input called 'data'
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})

        self.transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
        self.transformer.set_mean('data', mean)  # subtract the dataset-mean value in each channel
        self.transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
        self.transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
        # set the size of the input (we can skip this if we're happy
        #  with the default; we can also change it later, e.g., for different batch sizes)
        self.net.blobs['data'].reshape(self.batch_size,  # batch size
                                       3,  # 3-channel (BGR) images
                                       224, 224)  # image size is 227x227

    def get_caffe_feature(self, image_path):
        image = caffe.io.load_image(image_path)
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[0] = transformed_image
        self.net.forward()
        return preprocessing.normalize(self.net.blobs['fc6'].data, norm='l2')[0]

    def get_caffe_features(self, facesT):
        length = len(facesT)
        t = int(length / self.batch_size)
        full_features = []
        print("number of batches ", self.batch_size, " ", t)
        for j in range(0, t + 1):
            if j == t:
                faces = facesT[j * self.batch_size:]
            else:
                faces = facesT[j * self.batch_size:(j + 1) * self.batch_size]
            new_length = min(self.batch_size, len(faces))
            print("new_length", new_length)
            for i in range(0, new_length):
                image = caffe.io.load_image(faces[i])
                transformed_image = self.transformer.preprocess('data', image)
                self.net.blobs['data'].data[i] = transformed_image
            print("forwarding net ", self.net.blobs['data'].data.shape)
            self.net.forward()
            features = np.array(preprocessing.normalize(self.net.blobs['fc6'].data, norm='l2'))
            if j == t:
                features = features[:new_length]
            if j == 0:
                full_features = features
            else:
                full_features = np.concatenate((full_features, features), axis=0)
            print("feature size: ", full_features.shape)
        return full_features

    def get_distance(self, image_path1, image_path2):
        return distance.cosine(self.get_caffe_feature(image_path1), self.get_caffe_feature(image_path2))

    def verify(self, image_path1, image_path2):
        if image_path1 == None or image_path2 == None:
            return False

        if self.get_distance(image_path1, image_path2) <= self.threshold:
            return True
        else:
            return False

    def verifyClaim(self, Id=-1, feature=''):
        if feature == '' or Id == -1:
            return False, 1
        if self.use_pca == True and self.pca_w != None:
            feature = np.dot(np.transpose(self.pca_w), feature - self.pca_m)
        dist = distance.cosine(self.template[Id], feature)
        return dist <= self.threshold, dist

    def train(self, faces, ids):
        features = self.get_caffe_features(faces)
        ids = np.array(ids)
        if self.use_pca == True:
            print("starting pca with features", features.shape)
            self.pca_w, self.pca_m, self.template = self.generateTemplate_PCA(np.transpose(features), ids)
            print("pca: done")
        else:
            self.template = self.generateMeanTemplate(features, ids)
            self.pca_w = None
            self.pca_m = None

            # return features

    def test(self, faces, ids, outfile=None):
        print("testing started")
        if outfile == None:
            outfile = open("result.csv", "w")
        features = self.get_caffe_features(faces)
        features = np.transpose(features)
        for i in range(0, len(faces)):
            res, dist = self.verifyClaim(Id=ids[i], feature=features[:, i])
            outfile.write(",".join(str(x) for x in [faces[i], ids[i], res, dist, "\n"]))
        print("test done")

    def svc_param_selection(self, X, y, nfolds=5):
        param_grid = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 0.1, 1],
                      'C': [0.01, 0.1, 1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100, 1000]}
        # Cs = [0.001, 0.01, 0.1, 1, 10]
        # gammas = [0.001, 0.01, 0.1, 1]
        # param_grid = {'C': Cs, 'gamma' : gammas}
        cores = multiprocessing.cpu_count()
        print(cores)
        gridsearch = grid_search.GridSearchCV(svm.SVC(C=1), param_grid, n_jobs=cores, cv=nfolds)  # ,verbose = 5)
        gridsearch.fit(X, y)
        gridsearch.best_params_
        return gridsearch.best_params_

    def generateTemplate_SVM(self, features, ids):
        param = self.svc_param_selection(features, ids)
        print(param)
        clf = []
        if param["kernel"] == "rbf":
            clf = svm.SVC(probability=True, decision_function_shape='ovr', C=param["C"], kernel=param["kernel"],
                          gamma=param["gamma"])
        else:
            clf = svm.SVC(probability=True, decision_function_shape='ovr', C=param["C"], kernel=param["kernel"])
        clf.fit(features, ids)
        return clf  # OneVsRestClassifier(clf.fit(features, ids))

    def generateTemplate_NB(self, features, ids):
        clf = svm.SVC(probability=True, decision_function_shape='ovr', C=1, kernel='rbf', gamma=0.1)
        #        clf =  KNeighborsClassifier(5)
        clf.fit(features, ids)
        return clf  # OneVsRestClassifier(clf.fit(features, ids))

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
        W_e, X, m = self.getPCAFeatures(faces_train, id_train, K)

        z = self.generateMeanTemplate(X, id_train)
        return W_e, m, z

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
        W, q, m = self.getLDAFeatures(faces_train, id_train, r, use_pca)

        z = self.generateMeanTemplate(q, id_train)
        return W, m, z

    def generateMeanTemplate(self, faces, ids):
        # m = np.mean(faces,1)
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
        return z
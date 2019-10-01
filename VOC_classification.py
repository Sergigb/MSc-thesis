import os
import sys
import argparse

import torch
from torchvision import models
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import average_precision_score
from sklearn import preprocessing

from model import CNN

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('-k', type=int)
parser.add_argument('-cnn', type=str)
parser.add_argument('--mixture_model', action='store_true')
parser.add_argument('--n_topics', type=int, defaul=40)
args = parser.parse_args()

model_path = args.model_path
model_name = model_path.replace("/", "")

print(model_path)
print(model_name)

images_train_root = '../datasets/VOCdevkit/VOC2007-train/JPEGImages/'
images_val_root = '../datasets/VOCdevkit/VOC2007-val/JPEGImages/'

feat_layer = 4  # feature layer, check list(model.alexnet.children())
feat_root = 'data/features/feats-' + str(feat_layer) + "-" + model_name + '/'

if not os.path.isdir('data/features'):
    os.mkdir('data/features')

if not os.path.isdir(feat_root):  # extract features
    os.mkdir(feat_root)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    model = CNN(args.n_topics, args.k, out_dim=256, mixture_model=args.mixture_model, cnn=args.cnn)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    global lin_output

    def lin_ret_hook(self, input, output):
        global lin_output
        lin_output = output
        return None

    if args.cnn == 'alexnet':
        model.cnn.classifier[feat_layer].register_forward_hook(lin_ret_hook) # in case of alexnet
    else:
        model.cnn.avgpool.register_forward_hook(lin_ret_hook) # in case of resnet

    if torch.cuda.is_available():
        model.cuda()

    train_images = [f for f in os.listdir(images_train_root)]
    val_images = [f for f in os.listdir(images_val_root)]

    progress = 0
    for files, root in zip([train_images, val_images], [images_train_root, images_val_root]):
        for filename in files:
            image_path = os.path.join(root, filename)
            im = Image.open(image_path)
            im = transform(im)
            im = im.unsqueeze(0)
            if torch.cuda.is_available():
                im = im.cuda()

            _ = model(im)

            np.save(os.path.join(feat_root, filename), lin_output.cpu().detach().numpy())

            progress += 1
            sys.stdout.write("\rCompleted:  " + str(progress) + "/" + str(len(train_images) + len(val_images)))
            sys.stdout.flush()
print("")


svm_path = 'data/svm-' + model_name + '/'
gt_path = '../datasets/VOCdevkit/VOC2007-train/ImageSets/Main/'
gt_path_test = '../datasets/VOCdevkit/VOC2007-val/ImageSets/Main/'
scaler_path = 'data/scaler'
scaler_fname = 'scaler-layer-' + str(feat_layer) + model_name + '.pkl'

if not os.path.exists(svm_path):
    os.mkdir(svm_path)

classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
cs = [13, 14, 15, 16, 17, 18]  # List of margins for SVM

mAP2 = 0

for cl in classes:
    with open(gt_path + cl + '_train.txt') as f:
        content = f.readlines()
    aux = np.load(feat_root + content[0].split(' ')[0] + '.jpg.npy')
    X = np.zeros((len(content), (aux.flatten()).shape[0]), dtype=np.float32)
    y = np.zeros(len(content))
    idx = 0

    for sample in content:
        data = sample.split(' ')
        if data[1] == '': data[1] = '1'
        X[idx, :] = np.load(feat_root + data[0] + '.jpg.npy').flatten()
        y[idx] = max(0, int(data[1]))
        idx += 1

    with open(gt_path + cl + '_val.txt') as f:
        content = f.readlines()
    aux = np.load(feat_root + content[0].split(' ')[0] + '.jpg.npy')
    XX = np.zeros((len(content), (aux.flatten()).shape[0]), dtype=np.float32)
    yy = np.zeros(len(content))
    idx = 0

    for sample in content:
        data = sample.split(' ')
        if data[1] == '': data[1] = '1'
        XX[idx, :] = np.load(feat_root + data[0] + '.jpg.npy').flatten()
        yy[idx] = max(0, int(data[1]))
        idx += 1

    bestAP = 0
    bestC = -1

    scaler = preprocessing.StandardScaler().fit(X)
    if not os.path.isdir(scaler_path):
        os.mkdir(scaler_path)
    joblib.dump(scaler, scaler_path + scaler_fname)
    X_scaled = scaler.transform(X)
    XX_scaled = scaler.transform(XX)

    for c in cs:
        clf = svm.LinearSVC(C=pow(0.5, c))
        clf.fit(X_scaled, y)
        yy_ = clf.decision_function(XX_scaled)
        AP = average_precision_score(yy, yy_)
        if AP > bestAP:
            bestAP = AP
            bestC = pow(0.5, c)

    print " Best validation AP (class: "+cl+") :"+str(bestAP)+" found for C="+str(bestC)
    mAP2=mAP2+bestAP
    X_all = np.concatenate((X, XX), axis=0)
    scaler = preprocessing.StandardScaler().fit(X_all)
    X_all = scaler.transform(X_all)
    joblib.dump(scaler, scaler_path + scaler_fname)
    y_all = np.concatenate((y, yy))
    clf = svm.LinearSVC(C=bestC)
    clf.fit(X_all, y_all)
    joblib.dump(clf, svm_path + 'clf-'+cl+'-layer-'+str(feat_layer)+'.pkl')
  #  print "  ... model saved as " + svm_path + 'clf-'+cl+'-layer-'+str(feat_layer)+'.pkl'

print "\nValidation mAP: "+str(mAP2/float(len(classes)))+" (this is an underestimate, you must run VOC_eval.m for \
      mAP taking into account don't care objects)"

# Testing

mAP2=0

for cl in classes:
    with open(gt_path_test+cl+'_test.txt') as f:
        content = f.readlines()
    print "Testing one vs. rest SVC for class "+cl+" for "+str(len(content))+" test samples"
    aux = np.load(feat_root+content[0].split(' ')[0]+'.jpg.npy')
    X = np.zeros((len(content),(aux.flatten()).shape[0]), dtype=np.float32)
    y = np.zeros(len(content))
    idx = 0
    for sample in content:
        data = sample.split(' ')
        if data[1] == '': data[1] = '1'
        X[idx,:] = np.load(feat_root+data[0]+'.jpg.npy').flatten()
        y[idx]   = max(0,int(data[1]))
        idx += 1

#    print "  ... loading model from " + svm_path + 'clf-'+cl+'-layer-'+str(feat_layer)+'.pkl'
    clf = joblib.load(svm_path + 'clf-'+cl+'-layer-'+str(feat_layer)+'.pkl')
    scaler = joblib.load(scaler_path + scaler_fname)
    X = scaler.transform(X)

    y_ = clf.decision_function(X)
    AP = average_precision_score(y, y_)
    print "  ... Test AP: "+str(AP)
    mAP2 += AP

    # fr = open(res_root+'RES_cls_test_'+cl+'.txt','w+')
    # idx = 0
    # for sample in content:
    #     data = sample.split(' ')
    #     fr.write(str(data[0])+' '+str(y_[idx])+'\n')
    #     idx = idx+1
    # fr.close()

print("\nTest mAP: "+str(mAP2/float(len(classes)))+" (this is an underestimate, you must run VOC_eval.m for mAP \
       taking into account don't care objects)", 'green')



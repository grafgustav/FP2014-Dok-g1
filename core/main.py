import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import Image
import matplotlib
# ACHTUNG: Die vlfeat Python Bindungs werden nur fuer Linux unterstuetzt und 
# muessen nicht unbedingt eingebunden werden --> siehe unten
import vlfeat
import cPickle as pickle
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from curses.textpad import rectangle
from CalcHelper import Calculator
from sets import Set

'''
Created on 27.06.2014

@author: abelst00
'''

def project():
    
    #Constants
    step_size = 10
    cell_size = 6
    n_centroids = 50
    
    # Setting Files up.
    File_String = "2700270"
    
    
    file = open("../resources/GT/%s.gtp" % File_String)
    document_image_filename = "../resources/pages/%s.png" % File_String
    
    
    # loads the GT and stores it in a numpy array
    GT = []
    for line in file:
        t = line.split()
        GT.append([int(t[0]),int(t[1]),int(t[2]),int(t[3])])         
    GT = np.array(GT)
    
    # calculate codebook, labels n stuff
    image = Image.open(document_image_filename)
    im_arr = np.asarray(image, dtype='float32')
    frames, desc = vlfeat.vl_dsift(im_arr, step=step_size, size=cell_size) 
    frames = frames.T
    desc = desc.T
    codebook, labels = kmeans2(desc, n_centroids, iter=20, minit='points')  
    
    # Draw plot
    
#     draw_descriptor_cells = True
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im_arr, cmap=cm.get_cmap('Greys_r'))
    ax.hold(True)
    ax.autoscale(enable=False)
    for i in GT:
        rect = Rectangle((i[0], i[1]), (i[2]-i[0]), (i[3]-i[1]), alpha=1, lw=1, color="red", fill=False)
        ax.add_patch(rect) 
#   plt.show()
    
    
    
    # calculate bag of feature representations of each word marked by the ground truth
    calc = Calculator(codebook, labels, frames)
    bagOfFeatures = []
    for g in GT:
        bagOfFeatures.append(calc.getHistogramOfWord(g))
        
    IFS = {}
    for i in range(n_centroids):
        IFS[i]=[]
        
    i = 0
    for word in bagOfFeatures:
        j = 0
        for k in word:
            if k>0:
                IFS[j].append(i)
            j += 1      
        i += 1
    
    for i in IFS:
        print IFS[i]
        
   #"the" als Testword definiert
   
#     for k in np.arange(len(GT)):
#         test = GT[k]
#      
#         testHisto = calc.getHistogramOfWord(test)
#      
# #         print testHisto
# #         print testHisto.shape
#      
#         newSet = []
#         j = 0
#         for i in testHisto:
#             if i > 0:
#                 newSet = newSet + IFS[j]
#             j += 1
#      
#         newSet = np.array(newSet)
#         newSet = np.unique(newSet)
#         print "InvertedFileStrukture" + str(newSet.shape)
    
    test = GT[4]
     
    testHisto = calc.getHistogramOfWord(test)
    testHisto = np.array([testHisto])    
     
    bagOfFeatures = np.array(bagOfFeatures)
    print bagOfFeatures.shape
    print testHisto.shape
     
    vecDist = scipy.spatial.distance.cdist(testHisto, bagOfFeatures, metric="cityblock")
    vecDist = np.argsort(vecDist)
    print vecDist[0,1:10]

if __name__ == '__main__':
    project()
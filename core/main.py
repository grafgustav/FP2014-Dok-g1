import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import Image
import matplotlib
import pickle
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
    
    print "Initializing program..."
    
    #Constants
    step_size = 5
    cell_size = 10
    n_centroids = 1000
    
    testword_number = 113
    
    # Setting Files up.
    File_String = "2700270"
    
    
    print "Loading Ground Truth..."
    
    file = open("../resources/GT/%s.gtp" % File_String)
    document_image_filename = "../resources/pages/%s.png" % File_String
    
    # loads the GT and stores it in a numpy array
    GT = []
    words = []
    for line in file:
        t = line.split()
        GT.append([int(t[0]),int(t[1]),int(t[2]),int(t[3])])
        words.append(t[4])
    GT = np.array(GT)
    words = np.array(words)
    
    print "Calculating Codebook..."
    print GT
    
  
    
    
    # calculate codebook, labels n stuff
    image = Image.open(document_image_filename)
    im_arr = np.asarray(image, dtype='float32')
#     frames, desc = vlfeat.vl_dsift(im_arr, step=step_size, size=cell_size) 
#     frames = frames.T
#     desc = desc.T
#     pickle.dump( frames, open( "frames5-10-1000cen.p", "wb" ) )
#     pickle.dump( desc, open( "desc5-10-1000cen.p", "wb" ) )
#     codebook, labels = kmeans2(desc, n_centroids, iter=20, minit='points')  
#     pickle.dump( codebook, open( "codebook5-10-1000cen.p", "wb" ) )
#     pickle.dump( labels, open( "labels5-10-1000cen.p", "wb" ) )
    frames = pickle.load(open('frames5-10-1000cen.p', 'rb')) 
    desc = pickle.load(open('desc5-10-1000cen.p', 'rb'))
    codebook = pickle.load(open('codebook5-10-1000cen.p', 'rb')) 
    labels = pickle.load(open('labels5-10-1000cen.p', 'rb'))
    print "Calculating bag-of-feature representation for every GT word..."
    
    # calculate bag of feature representations of each word marked by the ground truth
    calc = Calculator(codebook, labels, frames)
#     bagOfFeatures = []
#     for g in GT:
#         bagOfFeatures.append(calc.getHistogramOfWord(g))
#             
#     pickle.dump( bagOfFeatures, open( "bagOfFeatures5-10-1000cen.p", "wb" ) )
    bagOfFeatures = pickle.load(open('bagOfFeatures5-10-1000cen.p','rb'))
#     bagOfWords = []
#     for g in GT:
#         bagOfWords.append(g)
#     pickle.dump( bagOfWords, open( "bagOfWords5-10-1000cen.p", "wb" ) )
    bagOfWords = pickle.load(open('bagOfWords5-10-1000cen.p','rb'))
#     print len(bagOfWords)
#     print bagOfFeatures
#     print bagOfFeatures.shape
        
    print "Filling Inverted File Structure..."
        
    IFS = {}
    IFSS = {}
    for i in range(n_centroids):
        IFS[i] = []
        IFSS[i] = []
        
    i = 0
    for word in bagOfFeatures:
        j = 0
        for k in word:
            if k>0:
                IFS[j].append(i)
                IFSS[j].append(bagOfWords[i])
            j += 1      
        i += 1

        
#     "the" als Testword definiert
    print "Initializing test word..."
     
    test = GT[testword_number]
  
    testHisto = calc.getHistogramOfWord(test)
    testSpatial = calc.getSpatialPyramidVector(test)

    print "Finding candidates for camparison..."
  
    newSet = []
    newSetSpatial = []
    j = 0
    for i in testHisto:
        if i > 0:
            newSet = newSet + IFS[j]
            newSetSpatial = newSetSpatial + IFSS[j]
        j += 1
        
    newSetSpatial = np.array(newSetSpatial)
    newSetSpatial = np.unique(newSetSpatial)
  
    newSet = np.array(newSet)
    newSet = np.unique(newSet)
#     newSet = newSet[:]
#     print "Inverted FileStrukture %s"  % newSet
    
    testHisto = np.array([testHisto])    
     
    bagOfFeatures = np.array(bagOfFeatures)
    bagOfFeatures = bagOfFeatures[newSet]
    
#     print bagOfFeatures.shape
    
    bagOfWords = np.array(bagOfWords)    
    bagOfWords = bagOfWords[newSet]
    
    print "bag of words shape %s" % str(bagOfWords.shape)
#     print bagOfWords
        
#     spatialVectors = []
#     for word in bagOfWords:
#         spatialVectors.append(calc.getSpatialPyramidVector(word))
#     spatialVectors = np.array(spatialVectors)
#     pickle.dump( spatialVectors, open( "spatialVectors5-10-1000cen.p", "wb" ) )
    spatialVectors = pickle.load(open('spatialVectors5-10-1000cen.p','rb'))
#     print bagOfFeatures.shape
    print "Calculating distances..."
#      
#     vecDist = scipy.spatial.distance.cdist(testHisto, bagOfFeatures, metric="cityblock")
#     vecDist = np.argsort(vecDist)
#     vecDist = vecDist[0,1:10]
    
    print "Calculating distances using the spatial pyramid..."
    
    testSpatial = np.array([testSpatial])
    
#     print testSpatial
    
    print "shape spacialvectors %s" % str(spatialVectors.shape)
#     print "shape testspatial %s" % str(testSpatial.shape)
      
    vecDist = scipy.spatial.distance.cdist(spatialVectors, spatialVectors, metric="euclidean")
    vecDist = np.argsort(vecDist)
     
#     vecDist = pickle.load(open('vecDist.p','rb'))
#     vecDist = vecDist[0,1:10]
#     pickle.dump( vecDist, open( "vecDist5-10-1000cen.p", "wb" ) )
#     print vecDist.shape
    
    print "Drawing results..."
    
    precision = np.zeros(len(GT))
    recall = np.zeros(len(GT))
    
    print len(GT)
    print vecDist.shape
    print precision.shape
    print recall.shape
    print vecDist
#     for i in range(len(GT)):
    precision_erg = 0.0
    print vecDist[testword_number,0:10]
    print words[vecDist[testword_number,0:10]]
    for i in range(len(GT)):
        if(ber_pre(words,i)!=0):
            precision[i], recall[i] = calculate_accuary(i, vecDist[i,0:ber_pre(words,i)], words, GT,  im_arr)
            precision_erg = precision_erg + precision[i]
    precision_erg = float(precision_erg) / float(len(GT))
    print precision_erg
        
    
#     print percentages
    print "averrage precision: %s , averrage recall: %s " % (np.around(np.mean(precision_erg), 2), np.around(np.mean(recall), 2))

def draw_plot(test_word, words_found, words_should_be, im_arr):
    
    '''
draws a plot for the results. blue rectangle is for test word. red rectangle is for similar word found.
:param: test_word: an array with 2 koordinates, stored like the GT.
words_found: an matrix with all similiar words. each line contains 2 koordinates, stored like the GT.
im_arr: the im_arr, cretated trough np.asarray()
'''
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im_arr, cmap=cm.get_cmap('Greys_r'))
    ax.hold(True)
    ax.autoscale(enable=False)
    for i in words_found:
        rect = Rectangle((i[0], i[1]), (i[2]-i[0]), (i[3]-i[1]), alpha=1, lw=1, color="red", fill=False)
        ax.add_patch(rect)
    for i in words_should_be:
        rect_1 = Rectangle((i[0], i[1]), (i[2]-i[0]), (i[3]-i[1]), alpha=1, lw=1, color="green", fill=False)
        ax.add_patch(rect_1)
    rect_2 = Rectangle((test_word[0], test_word[1]), (test_word[2]-test_word[0]), (test_word[3]-test_word[1]), alpha=1, lw=1, color="blue", fill=False)
    ax.add_patch(rect_2)
    
    plt.show()
    
    
    plt.show()
def ber_pre (words, testword_number):
    string = words[testword_number]   
    #       Berechenung von averrage precision und averrage recall
    words_should_be = []
    k = 0
    for i in words:
        if string==i :
            words_should_be.append(k)
        k += 1
    rw=0   
    words_should_be = np.array(words_should_be)
    if(len(words_should_be)==1):
       rw= 0 
    else:
        rw=len(words_should_be)
    return rw
def calculate_accuary(test_word_number, words_found_number, words, GT, im_arr):
    
    
    string = words[test_word_number]   
    #       Berechenung von averrage precision und averrage recall
    words_should_be = []
    k = 0
    for i in words:
        if string==i :
            words_should_be.append(k)
        k += 1   
    words_should_be = np.array(words_should_be)
    words_array = words[words_found_number]

    words_right_count = 0
    # words_right_count wie oft wir richtig getrofen haben
    for i in words_array:
        if i==string:
            words_right_count +=1
    # Anzahl alle richtige Treffer durch den Anzahl von 
    precision = float(words_right_count)/float(len(words_array))
    if len(words_should_be)>0:
        recall = float(len(words_should_be))/float(words_right_count)
    else:
        recall = 1
    
    print "Testword: \t\t [%s]" % string
    print "Testword (Nr) \t\t %s" % test_word_number
    if len(words_should_be)>0:
        print "Words should: \t\t %s" % words[words_should_be]
        print "Words should (Nr):\t %s" % words_should_be
    print "Words found: \t\t %s" % words_array
    print "Words found (Nr):\t %s" % words_found_number
#     print "Word [%s] \t Count of found words: %s, count of correct words: %s (%s%%)" % (string, len(words_array),words_right_count, percentage)
#     print " "
#     draw_plot( GT[test_word_number], GT[words_found_number], GT[words_should_be], im_arr)
    print precision, recall
    return precision, recall

if __name__ == '__main__':
    project()
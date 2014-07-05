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
    
    print "Initializing program..."
    
    #Constants
    step_size = 20
    cell_size = 2
    n_centroids = 5
    
    testword_number = 9
    
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
    
    # calculate codebook, labels n stuff
    image = Image.open(document_image_filename)
    im_arr = np.asarray(image, dtype='float32')
    frames, desc = vlfeat.vl_dsift(im_arr, step=step_size, size=cell_size) 
    frames = frames.T
    desc = desc.T
    codebook, labels = kmeans2(desc, n_centroids, iter=20, minit='points')  
     
    print "Calculating bag-of-feature representation for every GT word..."
    
    # calculate bag of feature representations of each word marked by the ground truth
    calc = Calculator(codebook, labels, frames)
    bagOfFeatures = []
    for g in GT:
        bagOfFeatures.append(calc.getHistogramOfWord(g))
        
    
    bagOfWords = []
    for g in GT:
        bagOfWords.append(g)
        
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
    
    bagOfWords = np.array(bagOfWords)    
    bagOfWords = bagOfWords[newSet]
    
    spatialVectors = []
    for word in bagOfWords:
        spatialVectors.append(calc.getSpatialPyramidVector(word))
    spatialVectors = np.array(spatialVectors)
    
#     print bagOfFeatures.shape
    print "Calculating distances..."
#      
#     vecDist = scipy.spatial.distance.cdist(testHisto, bagOfFeatures, metric="cityblock")
#     vecDist = np.argsort(vecDist)
#     vecDist = vecDist[0,1:10]
    
    print "Calculating distances using the spatial pyramid..."
    
    testSpatial = np.array([testSpatial])
     
    vecDist = scipy.spatial.distance.cdist(testSpatial, spatialVectors, metric="cityblock")
    vecDist = np.argsort(vecDist)
    vecDist = vecDist[0,1:10]
    
    
    print "Drawing results..."
    
    percentages = []
    
    for i in range(len(GT)):
        percentages.append(calculate_accuary(i, vecDist, words, GT, im_arr))
        
    percentages = np.array(percentages)
    print "Durchschnittliche Trefferrate: %s" % np.mean(percentages)

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
    
def calculate_accuary(test_word_number, words_found_number, words, GT, im_arr):
    
    words_should_be = []
    string = words[test_word_number]
    
    k = 0
    for i in words:
        if string==i and k!=test_word_number:
            words_should_be.append(k)
        k += 1
        
    words_should_be = np.array(words_should_be)
    words_array = words[words_found_number]

    words_right_count = 0
    
    for i in words_array:
        if i==string:
            words_right_count +=1
    
    percentage = 0
    if words_right_count > 0:
        percentage=((float(100)/float(len(words_array)))*float(words_right_count))
            
#     print "Testword: \t\t [%s]" % string
#     print "Testword (Nr) \t\t %s" % test_word_number
#     print "Words should: \t\t %s" % words[words_should_be]
#     print "Words should (Nr):\t %s" % words_should_be
#     print "Words found: \t\t %s" % words_array
#     print "Words found (Nr):\t %s" % words_found_number
    print "Word [%s] \t Count of found words: %s, count of correct words: %s (%s%%)" % (string, len(words_array),words_right_count, percentage)
    
#     draw_plot(GT[test_word_number], GT[words_found_number], GT[words_should_be], im_arr)
    return percentage

if __name__ == '__main__':
    project()
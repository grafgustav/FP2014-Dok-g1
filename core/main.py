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

'''
Created on 27.06.2014

@author: abelst00
'''

def bla():

    GT = []
    file = open('../resources/GT/2700270.gtp')
    for line in file:
        t = line.split()
        GT.append([int(t[0]),int(t[1]),int(t[2]),int(t[3])])
        
    
    GT = np.array(GT)
#     print GT
    
    document_image_filename = '../resources/2700270.png'
    image = Image.open(document_image_filename)
    im_arr = np.asarray(image, dtype='float32')
    
    step_size = 15
    cell_size = 5
    frames, desc = vlfeat.vl_dsift(im_arr, step=step_size, size=cell_size)
    
    frames = frames.T
    desc = desc.T
    print desc.shape
    
    print frames
    print desc
    print "frames:" + str(frames.shape)
    n_centroids = 50
    codebook, labels = kmeans2(desc, n_centroids, iter=20, minit='points')
    
    print labels.shape
#     for i in labels:
#         print i
    
    print codebook
    print codebook.shape
    
#     draw_descriptor_cells = True
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im_arr, cmap=cm.get_cmap('Greys_r'))
    ax.hold(True)
    ax.autoscale(enable=False)
    for i in GT:
#     colormap = cm.get_cmap('jet')
#     desc_len = cell_size * 4
#     for (x, y), label in zip(frames, labels):
#         color = colormap(label / float(n_centroids))
#         circle = Circle((x, y), radius=1, fc=color, ec=color, alpha=1)
#         print ((i[0], i[1]), (i[2]-i[0]), (i[3]-i[1]))
        rect = Rectangle((i[0], i[1]), (i[2]-i[0]), (i[3]-i[1]), alpha=1, lw=1, color="red", fill=False)
        

#         ax.add_patch(circle)
#         if draw_descriptor_cells:
#             for p_factor in [0.25, 0.5, 0.75]:
#                 offset_dyn = desc_len * (0.5 - p_factor)
#                 offset_stat = desc_len * 0.5
#                 line_h = Line2D((x - offset_stat, x + offset_stat), (y - offset_dyn, y - offset_dyn), alpha=0.08, lw=1)
#                 line_v = Line2D((x - offset_dyn , x - offset_dyn), (y - offset_stat, y + offset_stat), alpha=0.08, lw=1)
#                 ax.add_line(line_h)
#                 ax.add_line(line_v)
        ax.add_patch(rect)
    
#     plt.show()
    
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
    

if __name__ == '__main__':
    bla()
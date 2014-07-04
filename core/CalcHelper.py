import numpy as np

class Calculator(object):
    
    def __init__(self, centroids, des_label, des_coords):
        self.centroids = centroids
        self.des_label = des_label
        self.des_coords = des_coords

    def getHistogramOfWord(self, word):
        '''
        Returns historgram of a word. 
        :param: word: the word the histogram shall be calculated for
        :type: numpy array
        '''
        
        histo = np.zeros(len(self.centroids))
        
        i = 0
        for point in self.des_coords:
            if (point[0] >= word[0]) & (point[0] <= word[2]) & (point[1] >= word[1]) & (point[1] <= word[3]):
                 histo[self.des_label[i]] += 1
                 
            i += 1
            
        return histo
    
    def getSpatialPyramidVector(self, word):
        '''
        Returns the spacial pyramid of a word segment
        The  word gets separated as shown in the presentation slides (Complete, Left, Right)
        :param word: The word the spatial pyramid gets calculated to
        :type numpy array with shape = (1, 3xVOC_SIZE)
        '''
        
        result = np.zeros(len(self.centroids) * 3)
        # add complete histogram to result
        j = 0
        for i in self.getHistogramOfWord(word):
            result[j]  = i
            j += 1
        
        leftBorders = [word[0], word[1], round(word[2]/2.0), word[3]]
        
        for i in self.getHistogramOfWord(leftBorders):
            result[j] = i 
            j += 1
        
        rightBorders = [round(word[2]/2.0), word[1], word[2], word[3]]
        
        for i in self.getHistogramOfWord(rightBorders):
            result[j] = i 
            j += 1
            
        return result
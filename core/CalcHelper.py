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
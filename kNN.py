import numpy as np
import operator

def create_data_set():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0,0], [0,0.1]])
    labels = ['A','A','B','B']
    return group, labels
    
def classify0(in_x, data_set, labels, k):
    data_set_size = data_set.shape[0]
    diff_mat = np.tile(in_x, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2 
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sortd_dist_indicies = distances.argsort()
    class_count = {}
    
    for i in range(k):
        vote_i_label = labels[sortd_dist_indicies[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
        
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]
    
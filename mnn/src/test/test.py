# from sklearn.cluster import KMeans
# import numpy as np
# X = np.array([[32], [32], [32], [16], [16], [16], [16], \
#     [16], [32], [8], [8], [8], [8], [4], [4], \
#         [8], [8], [8], [4], [4]])
# kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
# print(kmeans.labels_)
from profile import read_profile_data
from utils import utils
import logging

class Student:
    def __init__(self, name):
        self.name = name
        self.age = 0



if __name__ == "__main__":
    # a = dict()
    # a[1] = 1
    # a[2] = 2
    # for k, v in a.items():
    #     print(k,v)
    # name_list, name_op_dict = read_profile_data.read_net_info("../models/model1/model1-info.txt")
    # for name in name_list:
    #     str_list = []
    #     op = name_op_dict[name]
    #     str_list.append(name)
    #     str_list.append(" | ")
    #     for op_parent in op.parents:
    #         str_list.append(op_parent.name)
    #     for op_child in op.children:
    #         str_list.append(op_child.name)
    #     print(str_list)
    
    s1 = Student('a')
    s2 = Student('b')
    print(s1.age)
    sl = []
    sl.append(s1)
    s1.age = 1
    s1.time = 2
    utils.get_logger().info(sl[0].age)
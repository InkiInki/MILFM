"""
@author: Inki
@contact: inki.yinji@gmail.com
@version: Created in 2020 1120, last modified in 2020 1120.
@note: The refer link: https://blog.csdn.net/weixin_44575152/article/details/109595872
"""
import os
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from Prototype import MIL
from I2I import dis_euclidean
from I2B import max_similarity
from FunctionTool import print_progress_bar, load_file


class MilFm(MIL):
    """
    The class of MILFM.
    @param:
        loops:
            The number of repetitions for kMeans.
        k:
            The number of clustering centers for kMeans.
        gamma:
            The gamma for i2i distance, include rbf and rbf2 (Gaussian distance).
    """

    def __init__(self, path, has_ins_label=True, loops=5, k=10, k_c=50, gamma=0.1):
        """
        The constructor.
        """
        super(MilFm, self).__init__(path, has_ins_label)
        self.loops = loops
        self.k = k
        self.k_c = k_c
        self.gamma = gamma
        self.mapping_path = ''
        self.full_mapping = []
        self.tr_idx = []
        self.te_idx = []
        self.__initialize_milfm()
        self.__full_mapping()

    def __initialize_milfm(self):
        """
        The initialize of MilFm.
        """
        self.mapping_path = 'D:/Data/TempData/Mapping/MilDm/' + self.data_name + '_max_rbf2_' + str(self.gamma) + '.csv'

    def __full_mapping(self):
        """
        Mapping bags by using all instances.
        @Note:
            The size of data set instance space will greatly affect the running time.
        """

        self.full_mapping = np.zeros((self.num_bags, self.num_ins))
        if not os.path.exists(self.mapping_path) or os.path.getsize(self.mapping_path) == 0:
            print("Full mapping starting...")
            open(self.mapping_path, 'a').close()

            for i in range(self.num_bags):
                print_progress_bar(i, self.num_bags)
                # print("%d-th bag mapping..." % i)
                for j in range(self.num_ins):
                    self.full_mapping[i, j] = max_similarity(self.bags[i, 0][:, :self.dimensions],
                                                             self.ins[j], 'rbf2', self.gamma)
            pd.DataFrame.to_csv(pd.DataFrame(self.full_mapping), self.mapping_path,
                                index=False, header=False, float_format='%.6f')
            print("Full mapping end...")
        else:
            temp_data = load_file(self.mapping_path)
            for i in range(self.num_bags):
                self.full_mapping[i] = [float(value) for value in temp_data[i].strip().split(',')]

    def get_mapping(self):
        """
        Get mapping.
        """

        self.tr_idx, self.te_idx = self.get_index(self.k)
        temp_positive_label = np.max(self.bags_label)
        for loop_k in range(self.k):

            # Step 1. Collect positive instances and clustering negative instances.
            temp_tr_idx = self.tr_idx[loop_k]
            temp_positive_ins_idx = []
            temp_negative_ins_idx = []
            temp_negative_ins = []
            for tr_idx in temp_tr_idx:
                temp_label = self.bags_label[tr_idx]
                for ins_i in range(self.bags_size[tr_idx]):
                    temp_idx = self.ins_idx[tr_idx] + ins_i
                    if temp_label == temp_positive_label:
                        temp_positive_ins_idx.append(temp_idx)
                    else:
                        temp_negative_ins_idx.append(temp_idx)
                        temp_negative_ins.append(self.ins[temp_idx].tolist())
            temp_negative_ins = np.array(temp_negative_ins)
            temp_kmeans = MiniBatchKMeans(self.k_c)
            temp_negative_center_ins_idx = []
            for loop in range(self.loops):
                temp_kmeans.fit(temp_negative_ins)
                temp_dis = []
                for idx_neg_ins in range(len(temp_negative_ins)):
                    temp_dis.append(dis_euclidean(temp_negative_ins[idx_neg_ins], temp_kmeans.labels_[idx_neg_ins]))
                temp_sorted_dis_idx = np.argsort(temp_dis)
                temp_negative_center_ins_idx += np.array(temp_negative_ins_idx)[temp_sorted_dis_idx].tolist()
            temp_negative_center_ins_idx = set(temp_negative_center_ins_idx)

            # Step 2. Mapping.
            temp_idx = temp_positive_ins_idx + list(temp_negative_center_ins_idx)
            temp_mapping = self.full_mapping[:, temp_idx]
            yield temp_mapping[temp_tr_idx], self.bags_label[temp_tr_idx],\
                  temp_mapping[self.te_idx[loop_k]], self.bags_label[self.te_idx[loop_k]], None

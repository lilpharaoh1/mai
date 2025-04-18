import os
import sys
sys.path.insert(0, os.getcwd()) # hacky fix

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.use14corefonts'] = True

from TrajectoryAidedLearning.Utils.utils import *
from TrajectoryAidedLearning.DataTools.plotting_utils import *


colors = [
        'gray',
        'red',
        'blue',
        'green',
    ]

names = [u"Progress \u2191", u"Step Reward \u2191", u"Velocity \u2191", u"Overtakes \u2191", u"A2A Collisions \u2193", u"Finishing Place \u2191"]


# # Classic One
# data_means = np.array([
#               [[0.1989, 0.3651, 4.7315, 0.0556, 12.7111, 1.9996],
#                [0.4909, 0.4298, 4.9480, 0.3982, 16.5556, 1.7524],
#                [0.5655, 0.3844, 4.4461, 0.1022, 14.5778, 1.9644],
#                [0.5388, 0.4122, 4.7301, 0.5382, 17.4889, 1.7969]],
               
#               [[0.1990, 0.3705, 4.7412, 0.1002, 16.0350, 1.9996],
#                [0.5331, 0.4360, 4.9729, 0.1792, 16.3300, 1.7741],
#                [0.5440, 0.3788, 4.4301, 0.0862, 15.6800, 1.9750],
#                [0.5027, 0.4018, 4.6441, 0.5909, 17.4650, 1.8533]],
#              ])
# data_stds = np.array([
#               [[0.0421, 0.0021, 0.0286, 0.0281, 6.0825, 0.0013],
#                [0.0474, 0.0136, 0.1170, 0.0452, 2.4851, 0.1513],
#                [0.0133, 0.0243, 0.2630, 0.0227, 3.0397, 0.0150],
#                [0.0310, 0.0146, 0.1934, 0.0450, 1.8260, 0.0395]],
               
#               [[0.0799, 0.0092, 0.0273, 0.1269, 15.1183, 0.0012],
#                [0.0878, 0.0237, 0.2165, 0.1707, 5.7790, 0.1059],
#                [0.0378, 0.0650, 0.7115, 0.0499, 6.4408, 0.0210],
#                [0.0541, 0.0401, 0.4662, 0.0704, 5.0403, 0.0951]],
#              ])
# ylims = [[0, None], [0, None], [0, None], [0, None], [0, None], [1, 2]]


# # Classic Three
# data_means = np.array([
#               [[0.1938, 0.3584, 4.6895, 0.0000, 10.4000, 4.0000],
#                [0.4932, 0.4011, 4.4279, 0.0053, 29.4667, 4.0000],
#                [0.4753, 0.3725, 4.3842, 0.0013, 24.8222, 4.0000],
#                [0.4710, 0.3363, 4.2555, 1.6627, 6.4889, 4.0000]],
               
#               [[0.1867, 0.3632, 4.6931, 0.0027, 15.8900, 3.9997],
#                [0.4368, 0.4298, 4.6419, 0.0500, 27.8450, 3.9958],
#                [0.3959, 0.3602, 4.3619, 0.0031, 23.9800, 3.9994],
#                [0.4761, 0.3356, 4.2118, 1.6245, 7.6300, 3.9994]],
#              ])
# data_stds = np.array([
#               [[0.0323, 0.0023, 0.0367, 0.0000, 6.0325, 0.0000],
#                [0.0961, 0.0226, 0.2108, 0.0082, 6.5401, 0.0000],
#                [0.0180, 0.0308, 0.3037, 0.0038, 3.8775, 0.0000],
#                [0.0168, 0.0222, 0.2511, 0.1279, 2.4660, 0.0000]],
               
#               [[0.0764, 0.0089, 0.0315, 0.0099, 15.6568, 0.0019],
#                [0.1870, 0.0432, 0.4436, 0.0916, 15.9838, 0.0102],
#                [0.0529, 0.0579, 0.7017, 0.0079, 10.8907, 0.0026],
#                [0.0354, 0.0600, 0.6581, 0.1035, 4.7547, 0.0026]],
#              ])
# ylims = [[0, None], [0, None], [0, None], [0, None], [0, None], [1, 4]]


# # Disparity Extender One
# data_means = np.array([
#               [[0.1902, 0.3730, 4.7825, 0.0249, 14.1333, 2.0000],
#                [0.5055, 0.4542, 4.9148, 0.3236, 21.0667, 1.7831],
#                [0.4769, 0.4232, 4.674, 0.1636, 16.5333, 1.9747],
#                [0.6228, 0.3810, 4.5425, 0.2383, 7.3778, 1.9049]],
               
#               [[0.1914, 0.3792, 4.7633, 0.0376, 18.9200, 1.9995],
#                [0.4055, 0.4374, 4.7633, 0.1516, 27.0350, 1.9077],
#                [0.4038, 0.4050, 4.5064, 0.1054, 17.6850, 1.9862],
#                [0.5490, 0.3625, 4.4294, 0.3023, 10.5050, 1.9335]],
#              ])
# data_stds = np.array([
#               [[0.0709, 0.0037, 0.0368, 0.0265, 12.1253, 0.0000],
#                [0.0391, 0.0211, 0.1970, 0.0374, 5.1208, 0.0202],
#                [0.0378, 0.0211, 0.2579, 0.0422, 2.4203, 0.0073],
#                [0.0295, 0.0162, 0.2247, 0.0543, 1.5533, 0.0543]],
               
#               [[0.1121, 0.0085, 0.0875, 0.0428, 18.9877, 0.0013],
#                [0.1698, 0.0448, 0.4428, 0.1140, 11.5588, 0.0875],
#                [0.0632, 0.0530, 0.6439, 0.0816, 5.2086, 0.0179],
#                [0.0791, 0.0634, 0.6643, 0.1466, 8.1674, 0.0567]],
#              ])
# ylims = [[0, None], [0, None], [0, None], [0, None], [0, None], [1, 2]]

# Disparity Extender Three
data_means = np.array([
              [[0.1896, 0.3577, 4.6678, 0.0000, 10.1778, 4.0000],
               [0.5466, 0.4336, 4.7290, 0.2147, 24.9111, 3.8572],
               [0.7670, 0.4232, 4.5650, 0.2373, 8.6667, 3.8507],
               [0.8084, 0.3950, 4.4189, 0.1577, 3.6222, 3.9969]],
               
              [[0.1761, 0.3603, 4.6388, 0.0007, 16.6200, 4.0000],
               [0.4682, 0.4268, 4.7193, 0.1726, 26.2850, 3.9053],
               [0.6594, 0.4069, 4.4523, 0.1149, 14.9700, 3.9223],
               [0.6967, 0.3837, 4.3379, 0.5598, 10.8200, 3.9909]],
             ])
data_stds = np.array([
              [[0.0625, 0.0017, 0.0414, 0.0000, 8.8591, 0.0000],
               [0.0972, 0.0191, 0.2040, 0.1697, 7.5319, 0.1050],
               [0.0808, 0.0259, 0.2809, 0.1779, 5.1545, 0.1243],
               [0.0373, 0.0234, 0.3079, 0.0423, 2.0121, 0.0053]],
               
              [[0.0952, 0.0058, 0.0847, 0.0027, 17.0914, 0.0000],
               [0.2052, 0.0417, 0.4312, 0.1696, 14.4271, 0.0923],
               [0.1578, 0.0670, 0.7820, 0.1311, 7.5101, 0.0910],
               [0.1098, 0.0524, 0.7228, 0.3953, 8.9981, 0.0135]],
             ])
ylims = [[0, None], [0, None], [0, None], [0, None], [0, None], [1, 4]]

data_means = np.transpose(data_means, (0, 2, 1))
data_stds = np.transpose(data_stds, (0, 2, 1))


fig, axs = plt.subplots(2, 6, figsize=(18, 8), sharex=True)
fig.subplots_adjust(hspace=0.3, wspace=0.3)

for row in range(2):
    for col in range(6):
        ax = axs[row, col]
        
        ax.bar(
            range(len(data_means[row, col])), 
            data_means[row, col], 
            yerr=data_stds[row, col], 
            capsize=5,
            color=colors
        )

        ax.set_ylim(ylims[col])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_title(names[col])

std_img_saving("Data/Vehicles/dispext_222_barplot", grid=False)
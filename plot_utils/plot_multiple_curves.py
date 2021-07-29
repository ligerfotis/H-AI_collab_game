from math import sqrt
from statistics import stdev, mean
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

dir = "results/figures/"


file_1 = 'results/tmp/max_games_mode_continuous_O_a_70K_every10_Shafti_descending_maze_v2_expert_online_40games_1/test_scores.csv'
file_2 = 'results/tmp/max_games_mode_continuous_O_a_70K_every10_Shafti_descending_maze_v2_expert_online_40games_2/test_scores.csv'
file_3 = 'results/tmp/max_games_mode_continuous_O_a_70K_every10_Shafti_descending_maze_v2_expert_online_40games_3/test_scores.csv'
file_4 = 'results/tmp/max_games_mode_continuous_O_O_a_70K_every10_Shafti_descending_maze_v2_expert_online_70games_1/test_scores.csv'
file_5 = 'results/tmp/max_games_mode_continuous_O_O_a_70K_every10_Shafti_descending_maze_v2_expert_online_70games_2/test_scores.csv'
file_6 = 'results/tmp/max_games_mode_continuous_O_O_a_70K_every10_Shafti_descending_maze_v2_expert_online_70games_3/test_scores.csv'

legend_names = ["O-O-a 70K | Train-Games:40 | GU/Test Sessions:3 | online",
                "O-O-a 70K | Train-Games:70 | GU/Test Sessions:6 | online"]

# colors = ["g", "b", "r", "m", "navy", "darkorange"]
# colors_light = ["limegreen", "cornflowerblue", "lightcoral", "violet", "slateblue", "wheat"]

fill = True
start = 0
end = 205


def plot(filename_list, legend_names, figure_file=None):
    """
    Plot the mean and sem of list groups
    :param filename_list: list of lists (groups) to average from
    :param legend_names:
    :param figure_file:
    :return:
    """
    fig, ax = plt.subplots()
    axes = plt.gca()
    axes.set_ylim([start, end])
    # colors = sns.color_palette("husl", 5)
    with sns.axes_style("darkgrid"):
        for c, group_files in enumerate(filename_list):
            group_data_lists = []
            # read the data of each list in the group
            for file_name in group_files:
                my_data = np.genfromtxt(file_name, delimiter=',')
                group_data_lists.append(my_data)

            data, means, stds, x_axis = [], [], [], []
            for i in range(0, len(group_data_lists[0]), 10):
                for file_to_merge in group_data_lists:
                    data.extend(file_to_merge[i:i + 10])
                if i >= 10:
                    means.append(mean(data))
                    stds.append(stdev(data) / sqrt(len(data)))
                    x_axis.append(i/10-1)
                    data = []

            means, stds, x_axis = np.asarray(means), np.asarray(stds), np.asarray(x_axis)
            # meanst = np.array(means.ix[i].values[3:-1], dtype=np.float64)
            # sdt = np.array(stds.ix[i].values[3:-1], dtype=np.float64)
            if fill:
                ax.plot(x_axis, means, label=legend_names[c], marker='*')
                ax.fill_between(x_axis, means - stds, means + stds, alpha=0.5)
            else:
                plt.errorbar(x_axis, means, stds, label=legend_names[c], marker='*')
        ax.yaxis.set_ticks(np.arange(start, end, 10))
        ax.set_ylabel('Score')
        ax.set_xlabel('Testing Sessions')
        plt.grid()
        plt.legend(loc='lower right')
        plt.title("Testing Score Mean and SEM")
        plt.savefig(figure_file)
        # plt.show()


#
# plot([[file_1f, file_2f, file_3f, file_1t, file_2t, file_3t],
#       [file_1, file_2]], legend_names, figure_file=dir + "old_vs_new.png")

# plot([[file_3], [file_4]], legend_names, figure_file=dir + "40s_vs_30s_v2.png")

plot([[file_1, file_2, file_3], [file_4, file_5, file_6]], legend_names, figure_file=dir + "20 random games 70K 40 vs 70 games.png")

# plot([[file_1f, file_2f, file_3f, file_1t, file_2t, file_3t], [file_1, file_2], [file_3]], legend_names, figure_file=dir + "maze_versions.png")
# plot([[file_4],[file_1c], [file_1p]], legend_names, figure_file=dir + "expert_vs_novel_users_40games.png")

# plot([[file_name_1, file_name_2, file_name_3, file_name_1t, file_name_2t, file_name_3t],
#       [file_name_4, file_name_5, file_name_6, file_name_4t, file_name_5t, file_name_6t],
#       [file_name_7, file_name_8, file_name_9, file_name_7t, file_name_8t, file_name_9t],
#       [file_name_10, file_name_11, file_name_12, file_name_10t, file_name_11t, file_name_12t]],
#     legend_names,
#      figure_file=dir + "154K vs 28K 2 users.png")

# plot([filename_list_sparse1[0]], [legend_names_sparse1[0]], figure_file=dir +"offline_sparse_1")
# plot([filename_list_sparse1[1]], [legend_names_sparse1[1]], figure_file=dir +"online_sparse_1")
# plot([filename_list_sparse2[0]], [legend_names_sparse2[0]], figure_file=dir +"offline_sparse_2")
# plot([filename_list_sparse2[1]], [legend_names_sparse2[1]], figure_file=dir +"online_sparse_2")
#
# plot([filename_list_sparse1[0], filename_list_sparse2[0]], [legend_names_sparse1[0],legend_names_sparse1[0]], figure_file=dir +"offline_all rewards")
# plot([filename_list_sparse1[1], filename_list_sparse2[1]], [legend_names_sparse1[1], legend_names_sparse2[1]], figure_file=dir +"online_all rewards")
#
# plot(filename_list_sparse1, legend_names_sparse1, figure_file=dir +"Learning_Comparison_sparse_1")
# plot(filename_list_sparse2, legend_names_sparse2, figure_file=dir +"Learning_Comparison_sparse_2")
# plot(filename_list_sparse1 + filename_list_sparse2, legend_names_sparse1 + legend_names_sparse2,
#      figure_file=dir +"Learning_Comparison_all rewards")

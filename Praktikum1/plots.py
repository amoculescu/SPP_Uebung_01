import matplotlib.pyplot as plt

title = ["dotproduct", "quicksort", "heated_plate"]

data_dot = [0.0119740710, 0.0087391510, 0.0052726810, 0.0051024770, 0.0053829720]
data_quick = [0.01284963099897141, 0.1082519239998874, 0.9523564190021716]
data_heated = [21.7603, 11.2188, 7.56801, 4.95866, 4.36131]

x_dot = [1, 2, 4, 8, 16]
x_quick = [5, 6, 7]
x_heated = [1, 2, 4, 8, 16]

data = [data_dot, data_quick, data_heated]
x = [x_dot, x_quick, x_heated]
xlabels = ["threads", "Array length (log)", "threads"]

for mode in range(len(title)):
    plt.plot(x[mode], data[mode], '-o')
    plt.title("Evaluation of the " + title[mode] + " task")
    plt.xlabel(xlabels[mode])
    plt.ylabel("time in sec")
    plt.grid()
    plt.savefig("plots/plot_" + title[mode] + ".pdf", kbox_inches='tight')
    plt.show()

import matplotlib.pyplot as plt

# ================ Dotproduct ================
title = "dotproduct"

data_dot_parallel = [0.0119740710, 0.0087391510, 0.0052726810, 0.0051024770, 0.0053829720]
data_dot_sequential = [0.0120985640, 0.0120985640]
x_dot_parallel = [1, 2, 4, 8, 16]
x_dot_sequential = [1, 16]
plt.plot(x_dot_parallel, data_dot_parallel, '-o', label='parallel')
plt.plot(x_dot_sequential, data_dot_sequential, '-', label='sequential')
plt.title("Evaluation of the " + title)
plt.xlabel("threads")
plt.ylabel("time in sec")
plt.legend()
plt.grid()
plt.savefig("plots/plot_" + title + ".png", kbox_inches='tight')
plt.show()

# ================ Quicksort ================
title = "quicksort"

data_quick_parallel = [0.01284963099897141, 0.1082519239998874, 0.9523564190021716]
data_quick_sequential = [0.01209410600040428, 0.09453706600015721, 1.081194949000746]
x_quick = [5, 6, 7]

plt.plot(x_quick, data_quick_parallel, '-o', label='parallel')
plt.plot(x_quick, data_quick_sequential, '-o', label='sequential')
plt.title("Evaluation of the " + title)
plt.xlabel("Array length (log)")
plt.ylabel("time in sec")
plt.legend()
plt.grid()
plt.savefig("plots/plot_" + title + ".png", kbox_inches='tight')
plt.show()

# ================ Heatedplate ================
title = "heated_plate"

data_heated = [13.4109, 6.83496, 3.58897, 2.13486, 1.71605, 7.92361]

x_heated = [1, 2, 4, 8, 16, 32]

plt.plot(x_heated, data_heated, '-o')
plt.title("Evaluation of the " + title)
plt.xlabel("threads")
plt.ylabel("time in sec")
plt.grid()
plt.savefig("plots/plot_" + title + ".png", kbox_inches='tight')
plt.show()

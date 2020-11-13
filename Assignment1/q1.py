"""
Author@VaibhaviDharashivkar
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

"""
Importing data
"""


def importing_data(filename):
    MFCCs_10 = []
    MFCCs_17 = []
    Species = []
    data = np.genfromtxt(filename, dtype=None, delimiter=',', skip_header=1, unpack=True)
    for d in data:
        for var in range(3):
            if var == 0:
                MFCCs_10.append(d[var])
            elif var == 1:
                MFCCs_17.append(d[var])
            else:
                Species.append(d[var])
    return MFCCs_10, MFCCs_17, Species


"""
A scatter plot of the ‘raw’ samples (both features, separate color for each class)
"""


def scatterplot(filename, MFCCs_10, MFCCs_17, Species):
    MFCC_10 = np.asarray(MFCCs_10)
    MFCC_17 = np.asarray(MFCCs_17)
    Specie = np.asarray(Species)
    colors = []
    for i in Specie:
        if i == b'HylaMinuta':
            colors.append('red')
        else:
            colors.append('blue')

    plt.style.use('seaborn')
    plt.axvline(0, c=(0.5, 0.5, 0.5), ls='--')
    plt.axhline(0, c=(0.5, 0.5, 0.5), ls='--')
    plt.scatter(MFCC_10, MFCC_17, s=100, alpha=0.7, color=colors, edgecolors='black')
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.xlabel('MFCCs_10')
    plt.ylabel('MFCCs_17')
    plt.title(filename.split('.')[0], fontweight="bold", fontsize=20)
    red_patch = mpatches.Patch(color='red', label='HylaMinuta')
    blue_patch = mpatches.Patch(color='blue', label='HypsiboasCinerascens')
    plt.legend(handles=[red_patch, blue_patch], loc=0)
    plt.tight_layout()
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(filename.split('.')[0] + 'scatterplot.png', dpi=100)


"""
- For each frog class in the file:
    - 2 histograms (1 per feature/attribute)
"""


def histogramplot(filename, MFCCs_10, MFCCs_17, Species):
    count = 0
    for i in Species:
        if i == b'HylaMinuta':
            count += 1
    MFCCs_10_x1 = np.asarray(MFCCs_10[:count])
    MFCCs_10_y1 = np.asarray(MFCCs_10[count:])
    MFCCs_17_x2 = np.asarray(MFCCs_17[:count])
    MFCCs_17_y2 = np.asarray(MFCCs_17[count:])

    f, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=True)
    f.suptitle(filename.split('.')[0] + " Dataset", fontweight="bold", fontsize=14)
    ax1, ax2, ax3, ax4 = axes.flatten()
    ax1.hist(MFCCs_10_x1, edgecolor='black', align='mid', color='#81ADC8')
    ax1.set_xlabel('MFCCs_10 for HylaMinuta')
    ax1.set_ylabel('Frequence')
    ax1.grid(axis='y', alpha=0.75)
    ax2.hist(MFCCs_17_x2, edgecolor='black', align='mid', color='#CD4631')
    ax2.set_xlabel('MFCCs_17 for HylaMinuta')
    ax2.set_ylabel('Frequence')
    ax2.grid(axis='y', alpha=0.75)
    ax3.hist(MFCCs_10_y1, edgecolor='black', align='mid', color='#81ADC8')
    ax3.set_xlabel('MFCCs_10 for HypsiboasCinerascens')
    ax3.set_ylabel('Frequence')
    ax3.grid(axis='y', alpha=0.75)
    ax4.hist(MFCCs_17_y2, edgecolor='black', align='mid', color='#CD4631')
    ax4.set_xlabel('MFCCs_17 for HypsiboasCinerascens')
    ax4.set_ylabel('Frequence')
    ax4.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(filename.split('.')[0] + 'Histogramplots.png', dpi=100)
    # plt.close(f)


"""
- For each frog class in the file:
     - 2 line graphs (1 per feature/attribute - after sorting feature values)
"""


def linegrapgplot(filename, MFCCs_10, MFCCs_17, Species):
    count = 0
    for i in Species:
        if i == b'HylaMinuta':
            count += 1
    MFCCs_10_x1 = np.asarray(MFCCs_10[:count])
    MFCCs_10_y1 = np.asarray(MFCCs_10[count:])
    MFCCs_17_x2 = np.asarray(MFCCs_17[:count])
    MFCCs_17_y2 = np.asarray(MFCCs_17[count:])
    MFCCs_10_x1.sort()
    MFCCs_10_y1.sort()
    MFCCs_17_x2.sort()
    MFCCs_17_y2.sort()

    f, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 10), sharex=True)
    f.suptitle(filename.split('.')[0] + " Dataset", fontweight="bold", fontsize=14)
    ax1, ax2 = axes.flatten()
    ax1.plot(MFCCs_10_x1)
    ax1.plot(MFCCs_10_y1, '--')
    ax1.set_xlabel("MFCCs_10")
    ax1.set_ylabel("Frequency")
    ax1.legend(['HylaMinuta', 'HypsiboasCinerascens'], loc=2)
    ax2.plot(MFCCs_17_x2)
    ax2.plot(MFCCs_17_y2, '--')
    ax2.set_xlabel("MFCCs_17")
    ax2.set_ylabel("Frequency")
    ax2.legend(['HylaMinuta', 'HypsiboasCinerascens'], loc=2)
    plt.tight_layout()
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(filename.split('.')[0] +'linegraph.png', dpi=100)


"""
- Plotting Feature Distributions
    - A boxplot showing the distribution of features for both classes (For each class, 1 box+whiskers per feature; 4 boxes total)
"""


def boxplot(filename, MFCCs_10, MFCCs_17, Species):
    count = 0
    for i in Species:
        if i == b'HylaMinuta':
            count += 1
    MFCCs_10_x1 = np.asarray(MFCCs_10[:count])
    MFCCs_10_y1 = np.asarray(MFCCs_10[count:])
    MFCCs_17_x2 = np.asarray(MFCCs_17[:count])
    MFCCs_17_y2 = np.asarray(MFCCs_17[count:])

    plt.style.use('seaborn')
    plt.boxplot([MFCCs_10_x1, MFCCs_17_x2, MFCCs_10_y1, MFCCs_17_y2], labels=("MFCCs_10 \nHylaMinuta", "MFCCs_17 \nHylaMinuta", "MFCCs_10 \nHypsiboasC\ninerascens", "MFCCs_17 \nHypsiboasC\ninerascens"))
    plt.ylabel("Frequency")
    plt.title(filename.split('.')[0] + " Dataset")
    plt.tight_layout()
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(filename.split('.')[0] +'boxplot.png', dpi=100)


"""
- Plotting Feature Distributions
    - Bar graph with error bars (For each class, 1 error bar per feature; 4 errors bars total)
"""


def errorbarplot(filename, MFCCs_10, MFCCs_17, Species):
    count = 0
    for i in Species:
        if i == b'HylaMinuta':
            count += 1
    MFCCs_10_x1 = np.asarray(MFCCs_10[:count])
    MFCCs_10_y1 = np.asarray(MFCCs_10[count:])
    MFCCs_17_x2 = np.asarray(MFCCs_17[:count])
    MFCCs_17_y2 = np.asarray(MFCCs_17[count:])
    MFCCs_10_x1.sort()
    MFCCs_10_y1.sort()
    MFCCs_17_x2.sort()
    MFCCs_17_y2.sort()

    data = np.arange(len([MFCCs_10_x1, MFCCs_10_y1, MFCCs_17_x2, MFCCs_17_y2]))
    mean = [np.mean(MFCCs_10_x1), np.mean(MFCCs_17_x2), np.mean(MFCCs_10_y1), np.mean(MFCCs_17_y2)]
    sd = [np.std(MFCCs_10_x1), np.std(MFCCs_17_x2), np.std(MFCCs_10_y1), np.std(MFCCs_17_y2)]
    plt.bar(data, mean, yerr=sd, ecolor="red", capsize=10)
    plt.xticks(data, ["MFCCs_10 \nHylaMinuta", "MFCCs_17 \nHylaMinuta", "MFCCs_10 \nHypsiboasC\ninerascens", "MFCCs_17 \nHypsiboasC\ninerascens"])
    plt.title(filename.split('.')[0] + " Dataset")
    plt.tight_layout()
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(filename.split('.')[0] +'Errorbarplot.png', dpi=100)


"""
Descriptive Statistics. Separately for each data set, use numpy to compute 1) the mean (expected value), 
2) covariance matrix, and 3) standard deviation for each individual feature.
"""


def statisticValues(filename, MFCCs_10, MFCCs_17, Species):
    print(filename.split('.')[0]+" Dataset")
    MFCC_10 = np.asarray(MFCCs_10)
    MFCC_17 = np.asarray(MFCCs_17)
    MFCCs_10_mean = np.mean(MFCC_10)
    print("MFCCs_10 mean=", MFCCs_10_mean)
    MFCCs_17_mean = np.mean(MFCC_17)
    print("MFCCs_17 mean=", MFCCs_17_mean)

    cov_MFCCs_10 = np.cov(MFCC_10)
    print("co-variance of MFCCs_10=", cov_MFCCs_10)
    cov_MFCCs_17 = np.cov(MFCC_17)
    print("co-variance of MFCCs_17=", cov_MFCCs_17)

    sd_MFCCs_10 = np.std(MFCC_10)
    print("standard deviation of MFCCs_10=", sd_MFCCs_10)
    sd_MFCCs_17 = np.std(MFCC_17)
    print("standard deviation of MFCCs_17=", sd_MFCCs_17)


if __name__ == '__main__':
    Mfcc_10, Mfcc_17, Species = importing_data('Frogs.csv')
    scatterplot('Frogs.csv',Mfcc_10, Mfcc_17, Species)
    histogramplot('Frogs.csv',Mfcc_10, Mfcc_17, Species)
    linegrapgplot("Frogs.csv",Mfcc_10, Mfcc_17, Species)
    boxplot('Frogs.csv',Mfcc_10, Mfcc_17, Species)
    errorbarplot('Frogs.csv',Mfcc_10, Mfcc_17, Species)
    statisticValues('Frogs.csv',Mfcc_10, Mfcc_17, Species)
    Mfcc_10, Mfcc_17, Species = importing_data('Frogs-subsample.csv')
    scatterplot('Frogs-subsample.csv',Mfcc_10, Mfcc_17, Species)
    histogramplot('Frogs-subsample.csv',Mfcc_10, Mfcc_17, Species)
    linegrapgplot('Frogs-subsample.csv',Mfcc_10, Mfcc_17, Species)
    boxplot('Frogs-subsample.csv',Mfcc_10, Mfcc_17, Species)
    errorbarplot('Frogs-subsample.csv',Mfcc_10, Mfcc_17, Species)
    statisticValues('Frogs-subsample.csv',Mfcc_10, Mfcc_17, Species)

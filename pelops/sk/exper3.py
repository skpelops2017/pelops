%matplotlib inline
import matplotlib.pyplot as plt
import pickle

with open('experiments.pkl', 'rb') as hdl: 
    results = pickle.load(hdl)

for idx, dct in enumerate(results):
    figure = plt.figure()
    ax = plt.subplot(111)
    ax.plot(dct['gdata'].transpose())
    plot_name = "ResNet50 Features\n%s" % dct['labels'][idx+1].replace("; B", ";\nb")
    plt.title('{}\n({} CMC curves with {} experiments / curve)'.format(plot_name, dct['CMC_CNT'], dct['EXP_CNT']))
    ax.legend(('-stddev','avg','+stddev'), bbox_to_anchor=(1, -0.05), fancybox=True, shadow=True, ncol=5)
    plt.savefig("/sk/data/transforms/transpose_{}-cmc-resnet.png".format(idx+1), bbox_inches='tight', pad_inches=0.25)

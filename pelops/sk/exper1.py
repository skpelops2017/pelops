import pelops.datasets.slice as slice
import pelops.features.resnet50 as resnet
import pelops.datasets.featuredataset as feature_dataset
import pelops.experiment_api.experiment as exp
import pelops.analysis.analysis as analysis
import matplotlib.pyplot as plt

labels = [
    '',
    'No resample; Background val 0; Anchor at (0, 0)',
    'No resample; Background val 0; Anchor at center',
    'NEIGHBOR resample at largest integer multiple; Background val 0; Anchor at center',
    'BILINEAR resample at largest integer multiple; Background val 0; Anchor at center',
    'BICUBIC resample at largest integer multiple; Background val 0; Anchor at center',
    'LANCZOS resample at largest integer multiple; Background val 0; Anchor at center',
    'ANTIALIAS resample at largest integer multiple; Background val 0; Anchor at center'
]

ITEMS_PER_CAMERA = 10
Y_RANDOM = 1024
CAMERAS = 2
DROPPED = 0
CMC_CNT = 100
EXPERIMENTS = 100

for i in range(1, 8):

    # Load SLiCE chips and generate HOG features

    DATA_DIR = '/Users/schuylerx/dev/data/transforms/transform_%d' % i
    HDF_FILE = '/Users/schuylerx/dev/data/transforms/transform_%d.resnet.hdf5' % 1
    print(DATA_DIR)
    data_set = slice.SliceDataset(DATA_DIR, debug=True)
    res_features = resnet.ResNet50FeatureProducer(data_set)
    feature_dataset.FeatureDataset.save(HDF_FILE, *res_features.return_features())

    slice_hog_features = feature_dataset.FeatureDataset(HDF_FILE)
    experiment_gen = exp.ExperimentGenerator(slice_hog_features, CAMERAS, ITEMS_PER_CAMERA, DROPPED, Y_RANDOM)
    experiment_hldr = analysis.repeat_pre_cmc(slice_hog_features, experiment_gen, NUMCMC=CMC_CNT, EXPPERCMC=EXPERIMENTS)
    stats, gdata = analysis.make_cmc_stats(experiment_hldr, ITEMS_PER_CAMERA)

    # Plot experiment results

    figure = plt.figure()
    ax = plt.subplot(111)
    ax.plot(gdata.transpose())
    plot_name = "ResNet50 Features\n%s" % labels[i]
    plt.title('{}\n({} CMC curves with {} experiments / curve)'.format(plot_name, CMC_CNT, EXPERIMENTS))
    ax.legend(('-stddev','avg','+stddev'), bbox_to_anchor=(1, -0.05), fancybox=True, shadow=True, ncol=5)
    plt.savefig(HDF_FILE.replace('.hdf5', '.cmc.png'))

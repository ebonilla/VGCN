# Sensible defaults

# Optimization
NUM_EPOCHS = 200
INITIAL_LEARNING_RATE = 0.01
OPTIMIZER_NAME = "Adam"
ALTERNATE_OPTIMIZATION = False
GCN_OPT_STEPS = 100
ADJ_OPT_STEPS = 100
NUM_CLUSTERS = 2

# prior
ONE_SMOOTHING_FACTOR = 0.99  # smoothing factor for 1s (1=no smoothing)
CONSTANT = None  # `None` defaults to using given adjacency matrix.
REGULARIZER_L1_SCALE = None
REGULARIZER_L2_SCALE = None
ZERO_SMOOTHING_FACTOR = 1e-5  # smoothing factor for 0s (0=no smoothing)
#
# noisy flips edges such that 0->1 and 1->0
# missing removes edges such that 1->0
# adding adds edges such that 0->1
adj_corruption_methods = ["noisy", "missing", "adding"]
PERC_CORRUPTION = 0.1  # as a fraction of the number of edges in the graph
ADJ_CORRUPTION_METHOD = None
PRIOR_TYPES = ["smoothing", "feature", "knn", "free_lowdim"]
DEFAULT_PRIOR_TYPE = "smoothing"
KNN_METRICS = ["cosine", "minkowski"]
DEFAULT_KNN_METRIC = "cosine"
DEFAULT_KNN_K = 10
INIT_SIZE = 4
INIT_VAL = 1e-5

# Posterior
TEMPERATURE_PRIOR = TEMPERATURE_POSTERIOR = 0.1
MC_SAMPLES_TRAIN = 3
MC_SAMPLES_TEST = 3
POSTERIOR_TYPES = ["free", "amortized", "lowrank", "kronecker", "free_lowdim"]
DEFAULT_POSTERIOR_TYPE = "free"  # q(A_ij = 1) = sigmoid(g(z_i, z_j))
RELAXED = True
LOGIT_SHIFT = 0.0  # for lowrank posterior
LATENT_DIM = 64  # for lowrank posterior
KL_FACTOR = 1.0  # beta factor for KL divergence
USE_METIS = True

# Monitoring
EXPERIMENTAL = True
SAVE_CHECKPOINT_STEPS = 1000
SAVE_SUMMARIES_STEPS = 100
CHECKPOINT_DIR = None  # '/tmp/gcn-latent-net'
RESULTS_DIR = None
LOG_EVERY_N_ITER = 50

# Datasets, Training splits
DATASET_NAME = "cora"
RANDOM_SPLIT = False
SEED = None
SEED_NP = None
SEED_VAL = 1  # same default value as LDS
USE_HALF_VAL_TO_TRAIN = False  # like the LDS paper, if true, half the data in the validation set are used for training
SPLIT_SIZES = [0.9, 0.75]  # corresponds to training 10%, validation 22.5%, test 67.5%
BALANCED_SPLIT = False
SAMPLES_PER_CLASS = 20
#
TRAIN_RANGE = (0, 140)
VALIDATION_RANGE = (200, 500)
TEST_RANGE = (500, 1500)

# GCN parameters
DEGREE = None  # `None` defaults to GCN renormalization trick.
DROPOUT_RATE = 0.5
L2_REG_SCALE_GCN = 5e-4
LAYER_CLS_NAME = "dense"

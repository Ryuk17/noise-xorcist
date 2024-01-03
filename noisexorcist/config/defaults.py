from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of wavs during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of wavs for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Samplerate for wav
_C.INPUT.SAMPLE_RATE = 16000
# Frame length for wav
_C.INPUT.FRAME_LEN = 128
# Hop length for wav
_C.INPUT.HOP_LEN = 128
# FFT points for wav
_C.INPUT.NFFT = 256


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8

# For set re-weight
_C.DATALOADER.SET_WEIGHT = []

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #

_C.SOLVER = CN()

# AUTOMATIC MIXED PRECISION
_C.SOLVER.AMP = CN({"ENABLED": False})

# Optimizer
_C.SOLVER.OPT = "Adam"

_C.SOLVER.MAX_EPOCH = 120

_C.SOLVER.BASE_LR = 3e-4

# This LR is applied to the last classification layer if
# you want to 10x higher than BASE_LR.
_C.SOLVER.HEADS_LR_FACTOR = 1.

_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.NESTEROV = False

_C.SOLVER.WEIGHT_DECAY = 0.0005
# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0005

# The previous detection code used a 2x higher LR and 0 WD for bias.
# This is not useful (at least for recent models). You should avoid
# changing these and they exists only to reproduce previous model
# training if desired.
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.WEIGHT_DECAY_BIAS = _C.SOLVER.WEIGHT_DECAY

# Multi-step learning rate options
_C.SOLVER.SCHED = "MultiStepLR"

_C.SOLVER.DELAY_EPOCHS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = [30, 55]

# Cosine annealing learning rate options
_C.SOLVER.ETA_MIN_LR = 1e-7

# Warmup options
_C.SOLVER.WARMUP_FACTOR = 0.1
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"


_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.LOG_PERIOD = 100

# Number of wav samples per batch
# This is global, so if we have 8 GPUs and BATCH_SIZE = 16, each GPU will
# see 2 wavs per batch
_C.SOLVER.BATCH_SIZE = 16


# Backbone freeze iters
_C.SOLVER.FREEZE_ITERS = 0

_C.SOLVER.CHECKPOINT_PERIOD = 20

# Number of wavs per batch across all machines.
# This is global, so if we have 8 GPUs and BATCH_SIZE = 256, each GPU will
# see 32 wavs per batch
_C.SOLVER.BATCH_SIZE = 64

# Gradient clipping
_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
# Maximum absolute value used for clipping gradients
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 5.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

# ---------------------------------------------------------------------------- #
# Test options
# ---------------------------------------------------------------------------- #
# see 2 wavs per batch
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 8
_C.TEST.WEIGHT = ""


# ---------------------------------------------------------------------------- #
# REID LOSSES options
# ---------------------------------------------------------------------------- #
_C.MODEL.LOSSES = CN()
_C.MODEL.LOSSES.NAME = ("CrossEntropyLoss",)

# Cross Entropy Loss options
_C.MODEL.LOSSES.CE = CN()
# if epsilon == 0, it means no label smooth regularization,
# if epsilon == -1, it means adaptive label smooth regularization
_C.MODEL.LOSSES.CE.EPSILON = 0.0
_C.MODEL.LOSSES.CE.ALPHA = 0.2
_C.MODEL.LOSSES.CE.SCALE = 1.0

# Weighted Speech Distortion Loss options
_C.MODEL.LOSSES.WSD = CN()
_C.MODEL.LOSSES.WSD.ALPHA = 0.4
_C.MODEL.LOSSES.WSD.SCALE = 1.0

# -----------------------------------------------------------------------------
# KNOWLEDGE DISTILLATION
# -----------------------------------------------------------------------------

_C.KD = CN()
_C.KD.MODEL_CONFIG = []
_C.KD.MODEL_WEIGHTS = []
_C.KD.EMA = CN({"ENABLED": False})
_C.KD.EMA.MOMENTUM = 0.999


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "./runs/"


# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.CUDNN_BENCHMARK = False

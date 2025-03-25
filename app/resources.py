from os import path

import numpy as np

resources_path = path.join(path.dirname(path.dirname(__file__)), "resources")


# -------------------------------------------------------- DATA --------------------------------------------------------


# TRPL - BT (from experimental measurement) - X/Y1/Y2/Y3/...
BT_TRPL_DATA_PATH = path.join(resources_path, "data/BT_TRPL_data.txt")
BT_TRPL_DATA = np.loadtxt(BT_TRPL_DATA_PATH, unpack=True)

# TRPL - BTD (simulated with noise) - X/Y1/Y2/Y3/...
BTD_TRPL_DATA_PATH = path.join(resources_path, "data/BTD_TRPL_data.txt")
BTD_TRPL_DATA = np.loadtxt(BTD_TRPL_DATA_PATH, unpack=True)

# TRMC - BTD (simulated) - X/Y1/Y2/Y3/...
BTD_TRMC_DATA_PATH = path.join(resources_path, "data/BTD_TRMC_data.txt")
BTD_TRMC_DATA = np.loadtxt(BTD_TRMC_DATA_PATH, unpack=True, skiprows=1)

# TRMC - BTD (from Brenes 2017) - X1/Y1/X2/Y2/...
BTD_TRMC_DATA_PATH_2 = path.join(resources_path, "data/BTD_TRMC_data_2.txt")
BTD_TRMC_DATA_2 = np.loadtxt(BTD_TRMC_DATA_PATH_2, unpack=True, skiprows=1)


# ------------------------------------------------------- IMAGES -------------------------------------------------------


LOGO_TEXT_PATH = path.join(resources_path, "medias/logo_text.svg")
LOGO_PATH = path.join(resources_path, "medias/logo.svg")
MODELS_PATH = path.join(resources_path, "medias/models.svg")
ICON_PATH = path.join(resources_path, "medias/icon.png")
OPT_GUESS_PATH = path.join(resources_path, "medias/Optimisation_guess.png")
BT_MODEL_PATH = path.join(resources_path, "medias/BT_model.svg")
BTD_MODEL_PATH = path.join(resources_path, "medias/BTD_model.svg")


# ------------------------------------------------------- OTHERS -------------------------------------------------------


FITTING_MODE = "Fitting"
ANALYSIS_MODE = "Grid fitting"
APP_MODES = (FITTING_MODE, ANALYSIS_MODE)
TUTORIAL_PATH = open(path.join(resources_path, "medias/tutorial.mp4"), "rb").read()

import os.path
from os import path

from models import BTDModelTRMC, BTDModelTRPL, BTModelTRMC, BTModelTRPL
from utility.data import generate_download_link

resources_path = path.join(path.dirname(path.dirname(__file__)), "resources")
CSS_STYLE_PATH = os.path.join(resources_path, "style.css")

# -------------------------------------------------------- DATA --------------------------------------------------------

BT_TRPL_DATA = BTModelTRPL().generate_decays(noise=0.02)
BT_TRMC_DATA = BTModelTRMC().generate_decays(noise=0.02)
BTD_TRPL_DATA = BTDModelTRPL().generate_decays(noise=0.02)
BTD_TRMC_DATA = BTDModelTRMC().generate_decays(noise=0.02)
BT_header = ["Time (ns)"] + ["Intensity %i" % i for i in range(1, len(BT_TRPL_DATA[2]) + 1)]
BTD_header = ["Time (ns)"] + ["Intensity %i" % i for i in range(1, len(BTD_TRPL_DATA[2]) + 1)]
BT_TRPL_LINK = generate_download_link(BT_TRPL_DATA[:2], BT_header, text="TRPL data set 1 (BT model)")
BT_TRMC_LINK = generate_download_link(BT_TRMC_DATA[:2], BT_header, text="TRMC data set 1 (BT model)")
BTD_TRPL_LINK = generate_download_link(BTD_TRPL_DATA[:2], BTD_header, text="TRPL data set 1 (BTD model)")
BTD_TRMC_LINK = generate_download_link(BTD_TRMC_DATA[:2], BTD_header, text="TRMC data set 2 (BTD model)")

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
ANALYSIS_MODE = "Grid Fitting"
APP_MODES = (FITTING_MODE, ANALYSIS_MODE)
TUTORIAL_PATH = open(path.join(resources_path, "medias/tutorial.mp4"), "rb").read()

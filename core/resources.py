import numpy as np
from os import path

basepath = path.dirname(__file__)

# Test files
BT_TRPL_np_file = path.join(basepath, '../resources/data/BT_TRPL_nonprocessed.txt')
BT_TRPL_np = np.loadtxt(BT_TRPL_np_file, unpack=True)
BT_TRPL_p_file = path.join(basepath, '../resources/data/BT_TRPL_processed.txt')
BT_TRPL_p = np.loadtxt(BT_TRPL_p_file, unpack=True)
BTD_TRPL_p_file = path.join(basepath, '../resources/data/BTD_TRPL_processed.txt')
BTD_TRPL_p = np.loadtxt(BTD_TRPL_p_file, unpack=True)
BTD_TRMC_file = path.join(basepath, '../resources/data/BTD_TRMC.txt')
BTD_TRMC = np.loadtxt(BTD_TRMC_file, unpack=True, skiprows=1)
BTD_TRMC_perfect_file = path.join(basepath, '../resources/data/BTD_TRMC_perfect.txt')
BTD_TRMC_perfect = np.loadtxt(BTD_TRMC_perfect_file, unpack=True, skiprows=1)
incomplete_file = path.join(basepath, '../resources/data/incomplete_data.csv')

# Images
logo_text_filename = path.join(basepath, '../resources/medias/logo_text.svg')
logo_filename = path.join(basepath, '../resources/medias/logo.svg')
models_filename = path.join(basepath, '../resources/medias/models.svg')
icon_filename = path.join(basepath, '../resources/medias/icon.png')
opt_guess = path.join(basepath, '../resources/medias/Optimiation_guess.png')
btmodel_filename = path.join(basepath, '../resources/medias/BT_model.svg')
btdmodel_filename = path.join(basepath, '../resources/medias/BTD_model.svg')

# Others
fitting_mode = 'Fitting'
analysis_mode = 'Grid fitting'
app_modes = (fitting_mode, analysis_mode)
tutorial_video = open(path.join(basepath, '../resources/medias/tutorial.mp4'), 'rb').read()

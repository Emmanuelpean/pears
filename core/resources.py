import numpy as np
from os import path

basepath = path.dirname(__file__)

# Test files
test_file1 = np.loadtxt(path.join(basepath, '../resources/data/test_file.txt'), unpack=True)
test_file2 = np.loadtxt(path.join(basepath, '../resources/data/brenes.txt'), unpack=True)

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

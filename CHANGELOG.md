#### Version 2.0.1 - May 2025
##### 💻 Visual Updates
* Updated the interface to be closer to *Raft*.

### Version 2.0.0 - March 2025

##### ✨ TRMC Data Support
* *Pears* now supports fitting time-resolved microwave photoconductivity (TRMC) data.
* Added a "Quantity" input in the sidebar to select the type of data uploaded (TRMC or TRPL).
* Added TRMC example files. 

##### 💻 User Experience Enhancements and Visual Updates
* The photoexcited carrier concentration input now dynamically adjusts to match the number of decays in the uploaded file.
* Updated the example files data.
* Improved About sections.
* Refreshed the main logo.
* The main logo is now hidden when displaying fit results for a cleaner interface.
* Updated the colour theme to align with the *Pears* logo.

##### 🌟 Improved Fitting Results
* The corresponding TRMC data are displayed when TRPL data are fitted, and vice versa. 
* The TRPL/TRMC decays after 1 excitation pulse and after stabilisation are now displayed in the carrier accumulation section.
* The fit parameters are now displayed in a table format.
* Fit parameters can now be downloaded as a CSV.
* Added help bubbles for every section of the fitting results.

##### 👨‍💻 Code Quality & Performance Improvement
* Refactored the script for improved readability and maintainability. 
* Achieved 100% test coverage to ensure result reliability.

##### 🐛 Performance, Bug Fixes & Miscellaneous
* Fixed a bug that caused an error when the period input was incorrect.
* Resolved an issue that prevented the log scale on the plots from displaying correctly.


#### Version 1.1.3 - May 2022

##### 💻 User Experience Enhancements
* The Auger rate constant in the Bimolecular-Trapping-Auger model is now fixed to 0 by default.

##### 🐛 Performance, Bug Fixes & Miscellaneous
* Resolved an issue preventing successful data loading.

#### Version 1.1.2 - April 2022

##### 🐛 Performance, Bug Fixes & Miscellaneous
* Addressed bugs related to file uploads introduced in the previous update.

#### Version 1.1.1 - March 2022

##### 💻 User Experience Enhancements
* Added a "Pre-process" feature to shift data to zero and normalise it.
* Files with headers can now be uploaded.

### Version 1.1.0 - November 2021

##### 🗃 New Data Upload Options
  * Added support for a new file format (X1/Y1/X2/Y2).
  * Expanded compatibility to accept a wider range of data formats.

##### 🌟 Improved Fitting
  * Introduced the I₀ intensity parameter in the fitting procedure.
  * Added the process contribution calculation after fitting.
  * Added carrier accumulation effect calculations after fitting.
  * Added a parallel plot visualisation for grid fitting analysis.

##### 💻 User Experience Enhancements
  * Added help bubbles for easier navigation.
  * Enhanced model descriptions.
  * Fits and Grid Fit analysis results now persist on-screen until the run button is pressed again (unless the mode or data changes).
  * Included a video tutorial.

#### 🍐 General Updates
  * Added a disclaimer.
  * Introduced a website icon.

##### 🐛 Performance, Bug Fixes & Miscellaneous
  * Fixed a bug that prevented changing fixed parameter values.


### Version 1.0.0 - October 2021
##### ✨ Initial release

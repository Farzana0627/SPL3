Proposed Method Revising Perceptual Linear Prediction with Wavelet

Revising Perceptual Linear Prediction comes with some modifications of Perceptual Linear Prediction proposed by Hermansky. The bark filter bank of PLP is replaced by a modified Mel filter bank that uses 257 filters (Number of spectral coefficients) instead of 24-40 filters used in conventional Mel filter bank. The equal loudness weighing of PLP is replaced by initial preemphasis step done in calculation of MFCC features. Also, the center frequencies of the first and last Bark filter have a distance of ≈ 1 Bark to the boundaries of the frequency range [0; Nyquist frequency]. According to Hermansky, two additional filters with center frequencies 0 and Nyquist frequency should be there for a signal. However, computation of the filters would reach too far into undefined. Therefore, the first and last filter bank output values are duplicated. Notwithstanding this notion, in RPLP, duplication of these center frequencies is discarded to avoid over-emphasis. 
As discussed earlier, it is evident that Fourier transform cannot retain the time localized information in a signal whereas wavelet transform can both integrate the frequency and time resolution. Therefore, wavelet features can be extracted for speech recognition purpose to get rid of this problem. Incorporation of wavelet features with the RPLP method leads to the naming of our proposed method Revising Perceptual Linear Prediction with Wavelet (RPLPwavelet). Following are the comparative figures containing PLP, RPLP and RPLPwavelet processing.

Dataset Description

Speech audio files containing uttered individual Bengali digits from 0 (Shunno) to 9 (Noy) are used for experiment in this work. The files were recorded in various environments with or without noise by both male and female of age group 15 to 30. The environments include classroom, home, slightly crowded streets, office rooms with both silence and minimal amount of noise. Then the audio files were converted to .wav format because .wav files are lossless and uncompressed. Total 1000 sound files were recorded for each digit class having 100 files each. Then these data were used to generate synthetic data using SMOTE of python ‘imblearn’ library for imbalanced dataset.  Training dataset contained features for 5000 sound samples in total with 500 samples per digit class. However, the test samples contain authentic sound samples of both male and female. These are also collected from different environments with much variation of noise level in them.

Implementation 

Feature Extraction

The sound features for the methods PLP, RPLP and RPLPwavelet are extracted using Matlab codes. Rastamat sound processing toolbox is used in this work for calculating the PLP cepstral coefficients. 

Feature Classification

Acoustic features are obtained from application of PLP, RPLP and RPLPwavelet. Since CNN (Convolutional Neural Network) demands large dataset to train, artificial data are synthesized to be used for training the neural network model along with the authentic ones. Before training the model, the data need to be normalized. In this work, normalization with mean=zero and standard deviation=1 is performed. Keras library from Python is used to train the CNN model here that uses Tensorflow at its backend. Due to gpu support of tensorflow, training the models is very time efficient now. A dataset containing 500 samples per digit class is trained with two convolutional layers containing kernel size 128 and two dense layers with kernel size 200 in its first layer are applied. In the first three layes activation function ‘relu’ is used, and ‘softmax’ is used in the last layer. Different strategies and combinations are followed for obtaining the best results of each of the method. The generated models are saved as .json format for future loading and testing with different test datasets.



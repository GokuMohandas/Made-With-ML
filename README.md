<img src="https://raw.githubusercontent.com/GokuMohandas/practicalAI/master/images/logo.png" width=150>

### Running the notebooks
1. Access the notebooks at https://goku.me/practicalAI or in the `notebooks` directory in this repo.
2. Replace `https://github.com/` with `https://colab.research.google.com/github/` in the notebook URL or use this [Chrome extension](https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo) to do it with one click.
3. Sign into your Google account.
4. Click the `COPY TO DRIVE` button on the toolbar. This will open the notebook on a new tab.

<img src="https://raw.githubusercontent.com/GokuMohandas/practicalAI/master/images/copy_to_drive.png">

5. Rename this new notebook by removing the `Copy of` part in the title.
6. Run the code, make changes, etc. and it's all automatically saved to you personal Google Drive.


---


### Contributing to notebooks
1. Make your changes and download the Google colab notebook as a .ipynb file.

<img src="https://raw.githubusercontent.com/GokuMohandas/practicalAI/master/images/download_ipynb.png">

2. Go to https://github.com/GokuMohandas/practicalAI/tree/master/notebooks
3. Click on upload files.

<img src="https://raw.githubusercontent.com/GokuMohandas/practicalAI/master/images/upload.png">

4. Upload the .ipynb file.
5. Write a detailed commit title and message.
6. Name your branch as appropriately

<img src="https://raw.githubusercontent.com/GokuMohandas/practicalAI/master/images/commit.png">


---


### TODO
- [x] Notebook basics
- [x] Python
- [x] NumPy
- [x] Pandas
- [x] Linear regression
- [x] Logistic regression
- [x] Random forests
- [x] PyTorch
- [x] Multi-layer perceptron
- [x] Data and models
- [x] Object oriented ML
- [x] Convolutional neural networks
- [x] Embeddings
- [x] Recurrent neural networks
- [ ] CNN image classification and segmentation
- [ ] Advanced RNNs (conditioned hidden state, attention, char-level embeddings from CNN)
- [ ] Highway networks, residual, etc.
- [ ] Kmeans clustering
- [ ] Topic modeling
- [ ] AE, DAE, VAE, CVAE
- [ ] GANs
- [ ] Recommendation systems (matrix factorization, ALS, SGD)
- [ ] Transfer learning (language modeling), ELMO
- [ ] Multitask learning
- [ ] Transformers

|              Notebook              |                           ToDo                           |
|------------------------------------|----------------------------------------------------------|
| 00_Notebooks                       | <ul> <li>- [x] text cells </li> <li>- [x] code cells </li> <li>- [x] saving notebook </li> </ul> |
| 01_Python                          | <ul> <li>- [x] lists, tuples, dicts </li> <li>- [x] functions </li> <li>- [x] classes </li> </ul> |
| 02_NumPy                           | <ul> <li>- [x] indexing </li> <li>- [x] arithmetic </li> <li>- [x] advanced </li> </ul> |
| 03_Pandas                          | <ul> <li>- [x] loading data </li> <li>- [x] exploratory analysis </li> <li>- [x] preprocessing </li> <li>- [x] feature engineering </li> <li>- [x] saving data </li> </ul> |
| 04_Linear_Regression               | <ul> <li>- [x] overview </li> <li>- [x] training </li> <li>- [x] data </li> <li>- [x] scikit </li> <li>- [x] interpretability </li> <li>- [x] regularization </li> <li>- [x] categorical </li> <li>- [ ] polynomial </li> <li>- [ ] normal equation </li> </ul> |
| 05_Logistic_Regression             | <ul> <li>- [x] overview </li> <li>- [x] training </li> <li>- [x] data </li> <li>- [x] scikit </li> <li>- [x] metrics </li> <li>- [x] metrics </li> <li>- [x] interpretability </li> <li>- [x] cross validation </li> <li>- [ ] interaction terms </li> <li>- [ ] odds ratio </li> <li>- [ ] coordinate descent </li> </ul>|
| 06_Random_Forests                  | <ul> <li>- [x] decision tree </li> <li>- [x] training </li> <li>- [x] data </li> <li>- [x] scikit </li> <li>- [x] interpretability </li> <li>- [x] random forests </li> <li>- [x] interpretability </li> <li>- [x] grid search </li> <li>- [ ] regression example </li> <li>- [ ] gini vs. entropy </li> </ul> |
| 07_PyTorch                         | <ul> <li>- [x] tensors </li> <li>- [x] indexing </li> <li>- [x] gradients </li> <li>- [x] CUDA </li> <li>- [ ] customized function </li> </ul> |
| 08_Multilayer_Perceptron           | <ul> <li>- [x] overview </li> <li>- [x] training </li> <li>- [x] data </li> <li>- [x] linear model </li> <li>- [x] non-linear model </li> <li>- [x] tensorboardd </li> <li>- [x] activation functions </li> <li>- [x] initialize weights </li> <li>- [x] overfitting </li> <li>- [x] dropout </li> <li>- [ ] interpretability </li> <li>- [ ] dropconnect </li> <li>- [ ] PReLU </li> </ul> |
| 09_Data_and_Models                 | <ul> <li>- [x] data </li> <li>- [x] quality </li> <li>- [x] quantity </li> <li>- [x] modeling </li> </ul> |
| 10_Object_Oriented_ML              | <ul> <li>- [x] overview </li> <li>- [x] set up </li> <li>- [x] data </li> <li>- [x] Vocabulary </li> <li>- [x] Vectorizer </li> <li>- [x] Dataset </li> <li>- [x] Model </li> <li>- [x] Trainer </li> <li>- [x] Inference </li> <li>- [ ] tqdm </li> </ul>|
| 11_Convolutional_Neural_Networks   |                                                          |
| 12_CNNs_for_Text                   |                                                          |
| 13_Embeddings                      |                                                          |
| 14_Recurrent_Neural_Networks       |                                                          |
| 15_Advanced_RNNs                   |                                                          |
| 16_Highway_and_Residual_Networks   |                                                          |
| 17_Time_Series_Analysis            |                                                          |
| 18_Kmeans_Clustering               |                                                          |
| 19_Topic_Modeling                  |                                                          |
| 20_Auto_Encoders                   |                                                          |
| 21_Generative_Adversarial_Networks |                                                          |
| 22_Recommendation_Systems          |                                                          |
| 23_Transfer_Learning               |                                                          |
| 24_Multitask_Learning              |                                                          |
| 25_Low_Shot_Learning               |                                                          |
| 26_Reinforcement_Learning          |                                                          |


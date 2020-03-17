from sklearn.preprocessing import QuantileTransformer
import numpy as np
from sklearn.decomposition import PCA
import pickle

data = pickle.load(open('C:/Users/Fuzhi/OneDrive/Share_folder/data/data.pck','rb'))


def do_pca(n_components, data):

    #X = QuantileTransformer.fit_transform(data)
    pca = PCA(n_components)
    X_pca = pca.fit_transform(data)
    variance_score = sum(pca.explained_variance_ratio_[0:6])
    return pca, X_pca, variance_score

pca = PCA(100)
X_pca_train = pca.fit_transform(data['current'][data['train_ix']])[0:6]
var = pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_[0:6])

pca = PCA(100)
X_pca_test = pca.fit_transform(data['current'][data['test_ix']])[0:6]


pca = PCA(100)
X_pca_val = pca.fit_transform(data['current'][data['val']])[0:6]

scan_rate = np.reshape(data['v'][0:6], (-1, 1))
concentration = np.reshape(data['c'][0:6], (-1, 1))

m = np.append(X_pca_train, scan_rate, axis=1)
m.shape

train_x = [X_pca_train, scan_rate, concentration]

train_x[X_pca_train

print(train_x)

k0_scale = QuantileTransformer(data['k0'])
kc_scale = QuantileTransformer(data['kc'])
d_scale = QuantileTransformer(data['d'])

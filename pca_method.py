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

#train data
pca = PCA(n_components= 6)
X_pca_train = pca.fit_transform(data['current'][data['train_ix']])
#var = pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_[0:6])

scan_rate_train = np.reshape(data['v'][data['train_ix']], (-1, 1))

concentration_train = np.reshape(data['c'][data['train_ix']], (-1,1))

train_x = np.append(np.append(X_pca_train, scan_rate_train, axis=1), concentration_train, axis=1)

print(train_x)

#validation data

X_pca_val = pca.transform(data['current'][data['val']])

scan_rate_val = np.reshape(data['v'][data['val']], (-1, 1))

concentration_val = np.reshape(data['c'][data['val']], (-1,1))

val_x = np.append(np.append(X_pca_val, scan_rate_val, axis=1), concentration_val, axis=1)


#test data

X_pca_test = pca.transform(data['current'][data['test_ix']])

scan_rate_test = np.reshape(data['v'][data['test_ix']], (-1, 1))

concentration_test = np.reshape(data['c'][data['test_ix']], (-1, 1))

test_x = np.append(np.append(X_pca_test, scan_rate_test, axis= 1), concentration_test, axis = 1)


#prediction

scaling = QuantileTransformer(n_quantiles=1000, random_state=0)


k0_train = np.reshape(data['k0'][data['train_ix']], (-1, 1))
kc_train = np.reshape(data['kc'][data['train_ix']], (-1, 1))
D_train = np.reshape(data['d'][data['train_ix']], (-1, 1))

y= np.array([k0_train,kc_train,D_train])


train_y = scaling.fit_transform(D_train)

k0_test = np.reshape(data['k0'][data['test_ix']], (-1, 1))
kc_test = np.reshape(data['kc'][data['test_ix']], (-1, 1))
D_test = np.reshape(data['d'][data['test_ix']], (-1, 1))

test_y = scaling.transform(D_test)


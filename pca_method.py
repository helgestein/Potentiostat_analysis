from sklearn.preprocessing import QuantileTransformer
import numpy as np
from sklearn.decomposition import PCA
import pickle

data = pickle.load(open(r'C:\Users\Helge\OneDrive\Documents\Literatur\Share_Literatur\data\data.pck','rb'))


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


k0_train = data['k0'][data['train_ix']]
kc_train = data['kc'][data['train_ix']]
D_train = data['d'][data['train_ix']]

y= np.array([k0_train,kc_train,D_train])
train_y = scaling.fit_transform(y.T)

k0_test = data['k0'][data['test_ix']]
kc_test = data['kc'][data['test_ix']]
D_test = data['d'][data['test_ix']]

yt = np.array([k0_test,kc_test,D_test])
test_y = scaling.transform(yt.T)


k0_val = data['k0'][data['val']]
kc_val = data['kc'][data['val']]
D_val = data['d'][data['val']]

yv = np.array([k0_test,kc_test,D_test])
val_y = scaling.transform(yv.T)




from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
MultiRFRegression = MultiOutputRegressor(RandomForestRegressor(n_estimators= 50, max_depth=30,random_state=2))
MultiRFRegression.fit(train_x, train_y)
y_predict = MultiRFRegression.predict(test_x)


import xgboost as xgb
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror',
                          colsample_bytree = 0.3,
                          learning_rate = 0.1,
                          max_depth = 30,
                          alpha = 10,
                          n_estimators = 50,
                          verbose=-1)
xg_reg.fit(train_x,train_y)
pred_y = xg_reg.predict(test_x)

plt.hist2d(test_y[:,0],pred_y)
plt.show()

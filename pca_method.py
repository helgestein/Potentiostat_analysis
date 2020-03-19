from sklearn.preprocessing import QuantileTransformer
import numpy as np
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
import xgboost as xgb
import random

data = pickle.load(open('C:/Users/Fuzhi/OneDrive/Share_folder/data/data.pck', 'rb'))


def do_pca(n_components, data):
    # X = QuantileTransformer.fit_transform(data)
    pca = PCA(n_components)
    X_pca = pca.fit_transform(data)
    variance_score = sum(pca.explained_variance_ratio_[0:6])
    return pca, X_pca, variance_score


# train data
pca = PCA(n_components=6)
X_pca_train = pca.fit_transform(data['current'][data['train_ix']])
# var = pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_[0:6])

scan_rate_train = np.reshape(data['v'][data['train_ix']], (-1, 1))

concentration_train = np.reshape(data['c'][data['train_ix']], (-1, 1))

train_x = np.append(np.append(X_pca_train, scan_rate_train, axis=1), concentration_train, axis=1)

# validation data

X_pca_val = pca.transform(data['current'][data['val']])

scan_rate_val = np.reshape(data['v'][data['val']], (-1, 1))

concentration_val = np.reshape(data['c'][data['val']], (-1, 1))

val_x = np.append(np.append(X_pca_val, scan_rate_val, axis=1), concentration_val, axis=1)

# test data

X_pca_test = pca.transform(data['current'][data['test_ix']])

scan_rate_test = np.reshape(data['v'][data['test_ix']], (-1, 1))

concentration_test = np.reshape(data['c'][data['test_ix']], (-1, 1))

test_x = np.append(np.append(X_pca_test, scan_rate_test, axis=1), concentration_test, axis=1)

# prediction

scaling = QuantileTransformer(n_quantiles=10, random_state=0)

k0_train = data['k0'][data['train_ix']]
kc_train = data['kc'][data['train_ix']]
D_train = data['d'][data['train_ix']]

y = np.array([k0_train, kc_train, D_train])
train_y = scaling.fit_transform(y.T)

k0_test = data['k0'][data['test_ix']]
kc_test = data['kc'][data['test_ix']]
kc_test_reduced = np.where((kc_test > 0.79) & (kc_test < 0.91))
D_test = data['d'][data['test_ix']]

yt = np.array([k0_test, kc_test, D_test])
test_y = scaling.transform(yt.T)


# validation
k0_val = data['k0'][data['val']]
kc_val = data['kc'][data['val']]
D_val = data['d'][data['val']]

yv = np.array([k0_test, kc_test, D_test])
val_y = scaling.transform(yv.T)

# multi random forest regression model
MultiRFRegression = MultiOutputRegressor(RandomForestRegressor(n_estimators=50, max_depth= 45, min_samples_split= 3, random_state=3))
MultiRFRegression.fit(train_x, train_y)
#MultiRFRegression.feature_importances_
y_predict = MultiRFRegression.predict(test_x)

for i in range(3):


    print(r2_score(test_y[:, i], y_predict[:, i]))
    #print(r2_score(train_x[:, i], train_y[:, i]))
    print(mse(test_y[:, i], y_predict[:, i]))


    plt.figure(figsize=(5,5))
    plt.hist2d(test_y[:,i], y_predict[:,i], bins= 10)
    #plt.hist2d(train_x[:,i], train_y[:,i], bins= 10)
    #plt.hist2d(test_y[:, 0], test_y[:, 1], edgecolor='k', label="Data")
    #plt.hist2d(y_predict[:, 0], y_predict[:, 1], edgecolor='k', label="Multi RF score=%.2f" % r2_score(test_y, y_predict))
    plt.xlim([0, 1.5])
    plt.ylim([0, 1.5])
    plt.axis('equal')
    #plt.xlabel("target 1")
    #plt.ylabel("target 2")
    plt.show()


# getting the index and investigating regarding the data'value
test_indices =np.where((test_y[:, 1] > 0.79) & (test_y[:, 1] < 0.91))
global_indices = np.random.choice(data['test_ix'][test_indices],100,replace=False)

# evaluating the data correspond to their indices
kc_new = test_y[:,1][test_indices]
k0_new = test_y[:,0][test_indices]
d_new = test_y[:,2][test_indices]


#plotting the new data
current_new = []
potential_new = []
for i in range(10):
    current_new.append(data['current'][global_indices[i]])
    potential_new.append(data['potential'][global_indices[i]])
'''
fig, ax = plt.subplots(3, 3)
ax = ax.flatten()
for i in range(9):
    ax[i].plot(potential_new[i],current_new[i])
    ax[i].axis('off')
plt.show()
'''
# getting the index and investigating regarding the data'value
indices_pred =np.where((y_predict[:, 1] > 0.79) & (y_predict[:, 1] < 0.91))

# evaluating the data correspond to their indices
kc_new = y_predict[:,1][indices_pred]
k0_new = y_predict[:,0][indices_pred]
d_new = y_predict[:,2][indices_pred]

global_pred_indices = np.random.choice(data['test_ix'][indices_pred] ,size = 100, replace=False)
#plotting the new data
current_pred_new = []
potential_pred_new = []
for i in range(10):
    current_pred_new.append(data['current'][global_pred_indices[i]])
    potential_pred_new.append(data['potential'][global_pred_indices[i]])


fig, ax = plt.subplots(3, 3)
ax = ax.flatten()
for i in range(9):
    ax[i].plot(potential_pred_new[i], current_pred_new[i])
    ax[i].plot(potential_new[i], current_new[i])
    ax[i].axis('off')
plt.show()

# train data
pca = PCA(n_components=6)
X_test = pca.fit_transform(data['current'][data['test_ix']])
# var = pca.explained_variance_ratio_
print(sum(pca.explained_variance_ratio_[0:6]))
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
v =np.where((test_y[:, 1] > 0.79) & (test_y[:, 1] < 0.91))
v_new = np.transpose(v)

c = random.choice(v_new,100)

print(len(c))
# evaluating the data correspond to their indices
kc_new = test_y[:,1][v]
k0_new = test_y[:,0][v]
d_new = test_y[:,2][v]


#plotting the new data
current_new = []
potential_new = []
for i in range(100):
    current_new.append(data['current'][data['test_ix']][c[i]])
    potential_new.append(data['potential'][data['test_ix']][c[i]])


for i in range(10):
    plt.plot(potential_new[i], current_new[i])
    plt.show()

fig, ax = plt.subplots(10, 10)
ax = ax.flatten()
for i in range(100):
    ax[i].plot(potential_new[i],current_new[i])
plt.show()

'''
ax[col,row].plot(potential,current)
for row in ax:
    for col in row:
        col.plot(potential_new, current_new)


plt.show()
'''

from sklearn.preprocessing import QuantileTransformer
import numpy as np
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from simulate_cyclic_voltammetry import sim_cv, run

data = pickle.load(open(r'C:\Users\Helge\Documents\data\data.pck', 'rb'))

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
MultiRFRegression = MultiOutputRegressor(RandomForestRegressor(n_estimators=50,
                                                               max_depth=45,
                                                               min_samples_split=3,
                                                               random_state=3))
MultiRFRegression.fit(train_x, train_y)
# MultiRFRegression.feature_importances_
y_predict = MultiRFRegression.predict(test_x)


for i in range(3):
    print(r2_score(test_y[:, i], y_predict[:, i]))
    # print(mse(test_y[:, i], y_predict[:, i]))
    plt.figure(figsize=(5,5))
    plt.hist2d(test_y[:,i], y_predict[:,i], bins= 10)
    plt.xlim([0, 1.5])
    plt.ylim([0, 1.5])
    plt.axis('equal')
    plt.xlabel('Real')
    plt.ylabel('Prediced')
    plt.show()

# getting the index and investigating regarding the data'value
test_indices = np.where((test_y[:, 1] > 0.79) & (test_y[:, 1] < 0.91))[0]
#Issue 1: global indices are w.r.t. data
#global_test_indices = np.random.choice(test_indices, 100, replace=False)
global_test_indices = np.random.choice(data['test_ix'][test_indices], 100, replace=False)

# evaluating the data correspond to their indices
#isue 1.1: this way we are actually using the global indices
#this is a less error prone method I believe
kc_test_new = data['kc'][global_test_indices]
k0_test_new = data['k0'][global_test_indices]
d_test_new = data['d'][global_test_indices]
#these were missing for the simulation in line 120:
v_test_new = data['v'][global_test_indices]
c_test_new = data['c'][global_test_indices]

# creating current and potential of the corresponding indices of kc
#current_test = []
#potential_test = []
current_test = data['current'][data['test_ix']]
potential_test = data['potential'][data['test_ix']]
#issue 2: here you mixed up global and test indices
#global_current_test = current_test[global_test_indices]
#global_potential_test = potential_test[global_test_indices]
global_current_test = data['current'][data['test_ix']][test_indices]
global_potential_test = data['current'][data['test_ix']][test_indices]


### New simulation of Current and Potential
result_test = []
for i in range(len(k0_test_new)):
    print(i)
    #issue 3: you are running run with a fixed sweep rate which is not the case
    #looking at line 19 you can see that this is an input parameter just likie D
    #which here is also constant though it should not
    #result_test.append(run(1E-3, k0_test_new[i], kc_test_new[i], 1.0, d_test_new[i]))
    result_test.append(run(v_test_new[i], k0_test_new[i], kc_test_new[i], c_test_new[i], d_test_new[i]))

#I have not looked further but I see issues continuing
# getting the index and investigating regarding the prediction data'value
predict_indices = np.where((y_predict[:, 1] > 0.79) & (y_predict[:, 1] < 0.91))[0]
global_predict_indices = np.random.choice(predict_indices, 100, replace=False)

# evaluating the data correspond to their indices
kc_predict_new = y_predict[:,1][global_predict_indices]#see issue 1 and 1.1
k0_predict_new = y_predict[:,0][global_predict_indices]#see issue 1 and 1.1
d_predict_new = y_predict[:,2][global_predict_indices]#see issue 1 and 1.1

result_predict = []
for i in range(len(kc_predict_new)):
    print(i)
    #see issue 3
    result_predict.append(run(1E-3, k0_predict_new[i], kc_predict_new[i], 1.0, d_predict_new[i]))

fig, ax= plt.subplots(10, 10)
ax = ax.flatten()
for i in range(100):
    np.nan_to_num(result_predict[i][1], copy=False)
    np.nan_to_num(result_test[i][1], copy=False)
    ax[i].plot(result_test[i][0], result_test[i][1] / (np.max(np.abs(result_test[i][1]))))
    ax[i].plot(result_predict[i][0], result_predict[i][1]/np.max(np.abs(result_predict[i][1])))
    #ax[i].plot(potential_test[i], current_test[i]/np.max(np.abs(current_test[i])))
    ax[i].axis('off')
plt.show()

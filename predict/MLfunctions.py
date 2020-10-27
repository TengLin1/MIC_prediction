#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os


class LR:
    def __init__(self, path):
        self.path=path

    def LogisticRegression(self,input_file, ratio, Cval, function, col, arg, output, col_n):
        from sklearn import linear_model
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import scale

        if not os.path.isfile(self.path+"/"+input_file):
            print("No input file")
            return
        else:
            print("Reading input file")
            feature_inp= pd.read_csv(self.path+"/"+input_file,index_col=0)
            print("input file is imported")

        # feature_inp=feature_inp.dropna()

        X = feature_inp.iloc[:,list(range(15, feature_inp.shape[1]))]#15+24+500  feature_inp.shape[1]
        y = feature_inp.iloc[:,col]
        columns = X.columns

        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=ratio, random_state=42)
        sum = 0
        print("Performing Logistic Regression")
        if function ==1:
            regr = linear_model.LinearRegression()
            # Train the model using the training sets
            regr.fit(X_train, y_train)

            # Make predictions using the testing set
            y_pred = regr.predict(X_test)
            y_c = y_pred.astype('int')
            for i in range(len(y_pred)):
                if y_pred[i] >= y_test[i] / 2 and y_pred[i] <= y_test[i] * 2:
                    y_c[i] = 1
                    sum = sum + 1
                else:
                    y_c[i] = 0
            report = list(zip(y_test, y_pred, y_c))
            report = pd.DataFrame(report, columns=['test', 'pred', 'correct or not'])
            report.to_csv(self.path + output + "/" + "LR.csv", sep='\t')
        else:

            logreg = linear_model.LogisticRegression()
            # print(np.array(str(y_train).split()).shape[0])
            logreg.fit(X_train,  1000*y_train)
            y_pred = logreg.predict(X_test)
            # conf = confusion_matrix(y_test, y_pred, sample_weight=None)
            # labels = unique_labels(y_test, y_pred)
            # inp = precision_recall_fscore_support(y_test, y_pred, labels=labels, average=None)
            # res_conf = conf.ravel().tolist()
            # res_inp = np.asarray(inp).ravel().tolist()
            # y_test = np.asfarray(y_test, float)
            # y_train = np.asfarray(y_train, float)

            # report = res_conf + res_inp

            y_c = y_pred.astype('int')
            for i in range(len(y_pred)):
                if y_pred[i]/1000 >= y_test[i] / 2 and y_pred[i]/1000 <= y_test[i] * 2:
                    y_c[i] = 1
                    sum = sum + 1
                else:
                    y_c[i] = 0
            report = list(zip(y_test, y_pred/1000, y_c))
            report = pd.DataFrame(report, columns=['test', 'pred', 'correct or not'])
            report.to_csv(self.path + output + "/" + "LR_class.csv", sep='\t')
        print("Done!")
        return sum/len(y_pred)

class RF:
    def __init__(self, path):
        self.path=path   
    
    #Rnadom Forests
    def RandomForests(self,input_file, ratio,max_features_pr, n_estimators_pr, function, col, arg, output, col_n):
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.ensemble import RandomForestClassifier
        # feature_inp=pd.DataFrame()
        if not os.path.isfile(self.path+"/"+input_file):
            print("No input file")
            return
        else:
            print("Reading input file")
            feature_inp= pd.read_csv(self.path+"/"+input_file,index_col=0)
            print("input file is imported")

        # feature_inp=feature_inp.dropna()

        X = feature_inp.iloc[:, list(range(15, feature_inp.shape[1]))]
        y = feature_inp.iloc[:, col].values
        columns = X.columns
        X = X.values
        
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=ratio, random_state=42)
 
        print("Performing random Forests")
        sum = 0
        if function == 1:
            rfreg= RandomForestRegressor(n_jobs=-1,max_features= max_features_pr ,n_estimators=int(n_estimators_pr), oob_score = True)

            rfreg.fit(X_train,y_train)
            y_pred=rfreg.predict(X_test)

            y_c = y_pred.astype('int')
            for i in range(len(y_pred)):
                if y_pred[i] >= y_test[i] / 2 and y_pred[i] <= y_test[i] * 2:
                    y_c[i] = 1
                    sum = sum + 1
                else:
                    y_c[i] = 0
            report = list(zip(y_test, y_pred, y_c))
            report = pd.DataFrame(report, columns=['test', 'pred', 'correct or not'])
            report.to_csv(self.path + output + "/" + "RF.csv", sep='\t')
        else:
            rfreg = RandomForestClassifier(n_jobs=-1, max_features=max_features_pr, n_estimators=int(n_estimators_pr),
                                           oob_score=True)
            rfreg.fit(X_train, 1000*y_train)
            y_pred = rfreg.predict(X_test)

            y_c = y_pred.astype('int')
            for i in range(len(y_pred)):
                if y_pred[i] / 1000 >= y_test[i] / 2 and y_pred[i] / 1000 <= y_test[i] * 2:
                    y_c[i] = 1
                    sum = sum + 1
                else:
                    y_c[i] = 0
            report = list(zip(y_test, y_pred / 1000, y_c))
            report = pd.DataFrame(report, columns=['test', 'pred', 'correct or not'])
            report.to_csv(self.path + output + "/" + "RF_class.csv", sep='\t')
        print("Done!")
        return sum/len(y_pred)
    
class GB:
    def __init__(self, path):
        self.path=path 

    def GradientBoosting(self,input_file, ratio,max_features_pr, n_estimators_pr, function, col, arg, output, col_n):
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import explained_variance_score
        from sklearn import metrics
        from sklearn.ensemble import GradientBoostingClassifier

        if not os.path.isfile(self.path+"/"+input_file):
            print("No input file")
            return
        else:
            print("Reading input file")
            feature_inp= pd.read_csv(self.path+"/"+input_file,index_col=0)
            print("input file is imported")
        
        # feature_inp=feature_inp.dropna()
        
        X = feature_inp.iloc[:,list(range(15, feature_inp.shape[1]))]
        y = feature_inp.iloc[:,col].values
        columns = X.columns
        X = X.values
        
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=ratio, random_state=42)
        sum = 0
        print("Performing gradient boositng")
        if function == 1:
            gbreg = GradientBoostingRegressor(n_estimators=n_estimators_pr,learning_rate=0.01,max_depth=15,max_features=max_features_pr,min_samples_leaf=10,min_samples_split=10,loss='ls',random_state =42)
            gbreg.fit(X_train,y_train)
            y_pred=gbreg.predict(X_test)
            print("Accuracy : %.4g" % metrics.explained_variance_score(y_test, y_pred))
            y_c = y_pred.astype('int')
            for i in range(len(y_pred)):
                if y_pred[i] >= y_test[i] / 2 and y_pred[i] <= y_test[i] * 2:
                    y_c[i] = 1
                    sum = sum + 1
                else:
                    y_c[i] = 0
            report = list(zip(y_test, y_pred, y_c))
            report = pd.DataFrame(report, columns=['test', 'pred', 'correct or not'])
            print(sum/len(y_pred))
            report.to_csv(self.path + output + "/"+"GB.csv", sep='\t')
        else:
            gbreg = GradientBoostingClassifier(random_state=10, max_features=max_features_pr,n_estimators=int(n_estimators_pr), verbose=True)
            gbreg.fit(X_train, 1000*y_train)
            y_pred = gbreg.predict(X_test)
            y_c = y_pred.astype('int')
            for i in range(len(y_pred)):
                if y_pred[i] / 1000 >= y_test[i] / 2 and y_pred[i] / 1000 <= y_test[i] * 2:
                    y_c[i] = 1
                    sum = sum + 1
                else:
                    y_c[i] = 0
            report = list(zip(y_test, y_pred / 1000, y_c))
            report = pd.DataFrame(report, columns=['test', 'pred', 'correct or not'])
            report.to_csv(self.path + output + "/" + "GB_class.csv", sep='\t')
        print("Done!")
        return sum/len(y_pred)

    def XGBoosting(self, input_file, ratio, max_features_pr, n_estimators_pr, function, col, arg, output, col_n):

        import xgboost as xgb
        from xgboost import plot_importance
        from sklearn.model_selection import train_test_split
        from matplotlib import pyplot as plt
        import shap as shap

        if not os.path.isfile(self.path + "/" + input_file):
            print("No input file")
            return
        else:
            print("Reading input file")
            feature_inp = pd.read_csv(self.path + "/" + input_file, index_col=0)
            X_file= pd.read_csv(self.path + "/test_data/snp_important_1000.csv" , index_col=1) #merge.info.freq.out.csv
            print("input file is imported")

        # feature_inp = feature_inp.dropna()

        # X = feature_inp.iloc[:, list(range(15, feature_inp.shape[1]))] #feature_inp.shape[1]

        # data = X_file.values  # data是数组，直接从文件读出来的数据格式是数组
        # print(data)
        # indexl = list(X_file.keys())  # 获取原有csv文件的标题，并形成列表
        # data = list(map(list, zip(*data)))  # map()可以单独列出列表，将数组转换成列表
        # data = pd.DataFrame(data, index=indexl)
        X = X_file.iloc[list(range(0, col_n)),2:]#X_file.shape[0]
        y = feature_inp.iloc[:, col].values
        columns = X_file.iloc[list(range(0, col_n)),1:2] #X_file.shape[0]
        columns = columns.T
        # print('col = ', columns)
        # X = X.values
        # print(X)
        X = X.T
        # X = X.values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=42)
        sum = 0
        print("Performing XGboositng")
        if function == 1:
            model = xgb.XGBRegressor(max_depth=5, learning_rate=0.05, n_estimators=160, silent=True, objective='reg:gamma')
            model.fit(X_train, y_train)

            # 对测试集进行预测
            y_pred = model.predict(X_test)

            # 计算准确率
            # accuracy = accuracy_score(y_test, y_pred)
            # print("accuarcy: %.2f%%" % (accuracy * 100.0))

            # 显示重要特征
            # plot_importance(model)
            # plt.show()
            imlist = list(zip(list(columns), model.feature_importances_))
            imreport = pd.DataFrame(imlist, columns=['features', 'importance'])
            imreport.to_csv(self.path + output + "/" + "XGB_importance.csv", sep='\t')

            y_c = y_pred.astype('int')
            for i in range(len(y_pred)):
                if y_pred[i] >= y_test[i] / 2 and y_pred[i] <= y_test[i] * 2:
                    y_c[i] = 1
                    sum = sum + 1
                else:
                    y_c[i] = 0
            report = list(zip(y_test, y_pred, y_c))
            report = pd.DataFrame(report, columns=['test', 'pred', 'correct or not'])
            report.to_csv(self.path + output + "/" + "XGB.csv", sep='\t')
        else:
            model = xgb.XGBClassifier(max_depth=5,learning_rate=0.1,n_estimators=160,silent=True,objective='multi:softmax')
            model.fit(X_train, y_train)
            # 对测试集进行预测
            y_pred = model.predict(X_test)

            # explainer = shap.TreeExplainer(model)
            # shap_values = explainer.shap_values(X)
            # shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])
            # print('shap_values = ', shap_values)
            # print(explainer.expected_value[0])
            # shap.force_plot(explainer.expected_value[0], shap_values[0])
            # shap.summary_plot(shap_values, X)
            # shap_interaction_values = explainer.shap_interaction_values(X)
            # shap.summary_plot(shap_interaction_values[0], X)
            # shap.dependence_plot('2838799', shap_values[1], X)

            # expected_value = explainer.expected_value
            # shap.decision_plot(expected_value[1], shap_values[1], X, ignore_warnings=True)

            imlist = list(zip(list(columns), model.feature_importances_))
            imreport = pd.DataFrame(imlist, columns=['features', 'importance'])
            imreport.to_csv(self.path + output + "/" + "XGB_importance_class.csv", sep='\t')
            y_c = y_pred.astype('int')
            for i in range(len(y_pred)):
                if y_pred[i] >= y_test[i] / 2 and y_pred[i] <= y_test[i] * 2:
                    y_c[i] = 1
                    sum = sum + 1
                else:
                    y_c[i] = 0
            report = list(zip(y_test, y_pred, y_c))
            report = pd.DataFrame(report, columns=['test', 'pred', 'correct or not'])
            report.to_csv(self.path + output + "/" + "XGB_class.csv", sep='\t')

        print("Done!")
        return sum/len(y_pred)

    def XGBoosting_snp(self, input_file, ratio, max_features_pr, n_estimators_pr, function, col, arg, output, col_n):

        import xgboost as xgb
        from sklearn.model_selection import train_test_split

        if not os.path.isfile(self.path + "/" + input_file):
            print("No input file")
            return
        else:
            print("Reading input file")
            feature_inp = pd.read_csv(self.path + "/" + input_file, index_col=0)
            X_file= pd.read_csv(self.path + "/test_data/snp_important_1000.csv" , index_col=1) #merge.info.freq.out.csv
            print("input file is imported")

        X = X_file.iloc[list(range(0, col_n)),2:]#X_file.shape[0]
        y = feature_inp.iloc[:, col].values
        columns = X_file.iloc[list(range(0, col_n)),1:2] #X_file.shape[0]
        columns = columns.T
        X = X.T
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=42)
        sum = 0
        print("Performing XGboositng for snp")
        if function == 1:
            model = xgb.XGBRegressor(max_depth=5, learning_rate=0.05, n_estimators=160, silent=True, objective='reg:gamma')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            imlist = list(zip(list(columns), model.feature_importances_))
            imreport = pd.DataFrame(imlist, columns=['features', 'importance'])
            imreport.to_csv(self.path + output + "/" + "XGB_importance.csv", sep='\t')
            y_c = y_pred.astype('int')
            for i in range(len(y_pred)):
                if y_pred[i] >= y_test[i] / 2 and y_pred[i] <= y_test[i] * 2:
                    y_c[i] = 1
                    sum = sum + 1
                else:
                    y_c[i] = 0
            report = list(zip(y_test, y_pred, y_c))
            report = pd.DataFrame(report, columns=['test', 'pred', 'correct or not'])
            report.to_csv(self.path + output + "/" + "XGB.csv", sep='\t')
        else:
            model = xgb.XGBClassifier(max_depth=5,learning_rate=0.1,n_estimators=160,silent=True,objective='multi:softmax')
            model.fit(X_train, y_train)
            # 对测试集进行预测
            y_pred = model.predict(X_test)
            imlist = list(zip(list(columns), model.feature_importances_))
            imreport = pd.DataFrame(imlist, columns=['features', 'importance'])
            imreport.to_csv(self.path + output + "/" + "XGB_importance_class.csv", sep='\t')
            y_c = y_pred.astype('int')
            for i in range(len(y_pred)):
                if y_pred[i] >= y_test[i] / 2 and y_pred[i] <= y_test[i] * 2:
                    y_c[i] = 1
                    sum = sum + 1
                else:
                    y_c[i] = 0
            report = list(zip(y_test, y_pred, y_c))
            report = pd.DataFrame(report, columns=['test', 'pred', 'correct or not'])
            report.to_csv(self.path + output + "/" + "XGB_class.csv", sep='\t')

        print("Done!")
        # return sum/len(y_pred)

class DL:
    def __init__(self, path):
        self.path=path

    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        from pandas import DataFrame, concat
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def DL_lstm(self, input_file, ratio,function, col, arg, output, col_n):
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from keras.layers import Dense
        from keras.models import Sequential
        from keras.layers import Dropout
        from tensorflow.python.keras.layers import LSTM
        from keras.optimizers import RMSprop

        if not os.path.isfile(self.path + "/" + input_file):
            print("No input file")
            return
        else:
            print("Reading input file")
            feature_inp = pd.read_csv(self.path + "/" + input_file, index_col=0)
            print("input file is imported")

        # feature_inp = feature_inp.dropna()

        X = feature_inp.iloc[:, list(range(15, feature_inp.shape[1]))]  # feature_inp.shape[1]
        y = feature_inp.iloc[:, col:col+1].values
        columns = X.columns
        X = X.values
        data_set = y.astype('float64')
        # print("data_set = ", data_set)

        train_data_set = np.array(data_set)

        reframed_train_data_set = np.array(self.series_to_supervised(train_data_set, 3, 1).values)

        train_days = int(len(reframed_train_data_set) * 0.6)
        valid_days = int(len(reframed_train_data_set) * 0.2)

        train = reframed_train_data_set[:train_days, :]
        valid = reframed_train_data_set[train_days:train_days + valid_days, :]
        test = reframed_train_data_set[train_days + valid_days:, :]

        X_train, y_train = train[:, :-1], train[:, -1]
        X_valid, y_valid = valid[:, :-1], valid[:, -1]
        X_test, y_test = test[:, :-1], test[:, -1]

        X_train = X_train.reshape((X_train.shape[0], 3, 1))
        X_valid = X_valid.reshape((X_valid.shape[0], 3, 1))
        X_test = X_test.reshape((X_test.shape[0], 3, 1))
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=42)

        model = Sequential()
        print("Performing LSTM")
        sum = 0

        # 第一层
        model.add(LSTM(32, return_sequences=True, activation='tanh'))
        # 第二层
        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(0.5))
        # 第三层 因为是回归问题所以使用linear
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        # Trains the model for a given number of epochs (iterations on a dataset).
        model.fit(X_train, y_train, epochs=500, batch_size=50,
                        validation_data=(X_valid, y_valid), verbose=2, shuffle=False)

        model.summary()
        y_pred = model.predict(X_test)
        y_pred = y_pred.reshape(y_pred.shape[0], )
        y_c = y_pred.astype('int')
        # imlist = list(zip(list(columns), model.feature_importances_))
        # imreport = pd.DataFrame(imlist, columns=['features', 'importance'])
        # imreport.to_csv(self.path + output + "/" + "LSTM_importance.csv", sep='\t')
        for i in range(len(y_pred)):
            if y_pred[i] >= y_test[i] / 2 and y_pred[i] <= y_test[i] * 2:
                y_c[i] = 1
                sum = sum + 1
            else:
                y_c[i] = 0
        report = list(zip(y_test, y_pred, y_c))
        report = pd.DataFrame(report, columns=['test', 'pred', 'correct or not'])
        report.to_csv(self.path + output + "/" + "lstm.csv", sep='\t')

        return sum/len(y_pred)

    def DL_lstm_calssifier(self, input_file, ratio,function, col, arg, output, col_n):
        from numpy import array
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import TimeDistributed
        from keras.layers import LSTM
        from sklearn.model_selection import train_test_split
        from keras.layers import Dropout
        import numpy as np

        if not os.path.isfile(self.path + "/" + input_file):
            print("No input file")
            return
        else:
            print("Reading input file")
            feature_inp = pd.read_csv(self.path + "/" + input_file, index_col=0)
            print("input file is imported")

        # feature_inp = feature_inp.dropna()
        print("Performing LSTM classifier")
        X = feature_inp.iloc[:, list(range(15, feature_inp.shape[1]))]
        y = feature_inp.iloc[:, col].values
        columns = X.columns
        X = X.values

        y_unique = sorted(feature_inp.iloc[:, col].unique())
        n_class = len(y_unique)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=42)
        column = X_train.shape[1]
        y_zeros= np.zeros((y_train.shape[0],n_class))
        # y_train = y_train.reshape(y_train.shape[0], 8)
        for j in range(len(y_train)):
            y_zeros[j, list(y_unique).index(y_train[j])] = 1
        # print(X_train.shape)
        X_train = X_train.reshape((X_train.shape[0], 1, column))
        X_test = X_test.reshape((X_test.shape[0], 1, column))
        # print(y_zeros)
        n_batch = 1
        n_epoch = 50
        # create LSTM
        model = Sequential()
        # model.add(LSTM(18, return_sequences=True))
        # # model.add(TimeDistributed(Dense(10, activation='softmax')))
        # 第一层
        model.add(LSTM(16, input_shape=(1, column), return_sequences=True, activation='tanh'))
        # 第二层
        model.add(LSTM(16, return_sequences=False))
        model.add(Dropout(0.5))

        model.add(Dense(n_class, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        # print(model.summary())
        # train LSTM
        model.fit(X_train, y_zeros, epochs=n_epoch, batch_size=n_batch, verbose=2)
        # evaluate
        y_pred_c = model.predict_classes(X_test)
        y_pred = y_pred_c.astype('float64')
        y_c= y_pred_c.astype('int')
        sum = 0
        for yi in range(len(y_pred_c)):
            y_pred[yi] = y_unique[y_pred_c[yi]]
            if y_pred[yi] >= y_test[yi]/2 and y_pred[yi] <= y_test[yi]*2:
                y_c[yi] = 1
                sum = sum + 1
            else:
                y_c[yi] = 0
            # print(y_c[yi])
        # imlist = list(zip(list(columns), model.feature_importances_))
        # imreport = pd.DataFrame(imlist, columns=['features', 'importance'])
        # imreport.to_csv(self.path + output + "/" + "LSTM_importance_class.csv", sep='\t')
        report = list(zip(y_test, y_pred, y_c))
        report = pd.DataFrame(report, columns=['test', 'pred', 'correct or not'])
        report.to_csv(self.path + output + "/" + "lstm_class.csv", sep='\t')
        # for value in result[0, :, 0]:
        #     print('%.1f' % value)
        print("Done!")
        # print(sum/len(y_pred_c))
        return sum/len(y_pred)

    def DeepLearning_fcn(self,input_file, ratio, firstLayer, interlayer, dropout, numblayer, function, col, arg, output, col_n):
        import pandas as pd
        from keras.layers import Dense
        from keras.models import Sequential
        from keras.callbacks import EarlyStopping
        from keras.layers import Dropout
        from sklearn.model_selection import train_test_split
        import numpy as np
        import shap
        import matplotlib.pyplot as plt

        if not os.path.isfile(self.path+"/"+input_file):
            print("No input file")
            return
        else:
            print("Reading input file")
            feature_inp= pd.read_csv(self.path+"/"+input_file,index_col=0)
            X_file = pd.read_csv(self.path + "/test_data/snp_important_1000.csv", index_col=1)
            print("input file is imported")
        
        # feature_inp=feature_inp.dropna()
        
        # X = feature_inp.iloc[:,list(range(15, feature_inp.shape[1]))]  #feature_inp.shape[1]
        # y = feature_inp.iloc[:,col].values
        # columns = X.columns
        # print(type(X))
        # X = X.values

        X = X_file.iloc[list(range(0,col_n )), 2:]#X_file.shape[0]
        y = feature_inp.iloc[:, col].values
        columns = X_file.iloc[list(range(0,col_n )), 1:2]#X_file.shape[0]
        columns = columns.T
        # X = X.values
        X = X.T

        print("Performing FCN")
        sum = 0
        if function == 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=42)
            X_train, y_train = shap.datasets.boston()
            model = Sequential()
            model.add(Dense(int(firstLayer), activation='relu', input_shape=(X_train.shape[1],)))
            model.add(Dropout(dropout, input_shape=(X_train.shape[1],)))
            for i in range(1, int(numblayer)):
                model.add(Dense(int(interlayer), activation='relu'))
                model.add(Dropout(dropout))
            model.add(Dense(1, activation = 'linear'))
            model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
            early_stopping_monitor = EarlyStopping(patience=50)
            model.fit(X_train, y_train, validation_split=0.2, callbacks=[early_stopping_monitor], epochs=50,
                      batch_size=10)
            model.summary()
            # pred_prob = model.predict_proba(X_test)
            y_pred = model.predict(X_test)
            y_pred = y_pred.reshape(y_pred.shape[0], )
            y_c = y_pred.astype('int')
            # explainer = shap.DeepExplainer(model)
            # shap_values = explainer.shap_values(X)
            # print('shap_values = ', shap_values)
            # imlist = list(zip(list(columns), model.))
            # imreport = pd.DataFrame(imlist, columns=['features', 'importance'])
            # imreport.to_csv(self.path + output + "/" + "DL_importance.csv", sep='\t')
            for i in range(len(y_pred)):
                if y_pred[i] >= y_test[i] / 2 and y_pred[i] <= y_test[i] * 2:
                    y_c[i] = 1
                    sum = sum + 1
                else:
                    y_c[i] = 0
            report = list(zip(y_test, y_pred, y_c))
            report = pd.DataFrame(report, columns=['test', 'pred', 'correct or not'])
            report.to_csv(self.path + output + "/" + "DL.csv", sep='\t')
        else:
            # X = feature_inp.iloc[:, list(range(15, 133))].values
            # y = feature_inp.iloc[:, col].values
            y_unique = feature_inp.iloc[:, col].unique()
            y_unique = sorted(y_unique)
            n_class = len(y_unique)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=42)
            y_zeros = np.zeros((y_train.shape[0], n_class))
            # y_train = y_train.reshape(y_train.shape[0], 8)

            for j in range(len(y_train)):
                # print(list(sorted(y_unique)).index(y_train[j]))
                y_zeros[j, list(y_unique).index(y_train[j])] = 1

            # X_train = X_train.reshape((X_train.shape[0], 1, 18))
            # X_test = X_test.reshape((X_test.shape[0], 1, 18))
            model = Sequential()
            model.add(Dense(int(firstLayer), activation='relu'))
            model.add(Dropout(dropout, input_shape=(X_train.shape[1],)))
            for i in range(1, int(numblayer)):
                model.add(Dense(int(interlayer), activation='relu'))
                model.add(Dropout(dropout))
            model.add(Dense(n_class, activation='softmax'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            early_stopping_monitor = EarlyStopping(patience=50)
            model.fit(X_train, y_zeros, validation_split=0.2, callbacks=[early_stopping_monitor],
                      epochs=10, batch_size=128)
            # probability_true = model.predict(X_test)[:, 1]
            # score = model.evaluate(X_test, to_categorical(y_test))
            model.summary()
            y_pred_c = model.predict_classes(X_test)
            y_pred = y_pred_c.astype('float64')
            y_c = y_pred_c.astype('int')
            # print(X_test.iloc[:,1:])
            # shap.initjs()
            explainer = shap.DeepExplainer(model, X_test.values)
            shap_values = explainer.shap_values(X_test.values)
            # print('shap_values = ', shap_values[0])
            # print(explainer.expected_value)
            # shap.force_plot(explainer.expected_value[0], shap_values[0][0], X_test[0], link="logit")
            # shap_interaction_values = explainer.shap_interaction_values(X)
            # shap.summary_plot(shap_interaction_values[0], X)
            # shap.dependence_plot('ABC_efflux', shap_values[1], X_test)
            # shap.save_html(self.path + output + "/" +"test.png",shap.dependence_plot('ABC_efflux', shap_values[1], X_test) )
            # fig.savefig(self.path + output + "/" +"test.png")
            shap.summary_plot(shap_values, X)
            # expected_value = explainer.expected_value
            # shap.decision_plot(expected_value[1], shap_values[1], X, ignore_warnings=True)

            # print('weights = ',model.weights)
            # print('get_weights = ', model.get_weights())
            # imlist = list(zip(list(columns), model.we))
            # imreport = pd.DataFrame(imlist, columns=['features', 'importance'])
            # imreport.to_csv(self.path + output + "/" + "DL_importance_class.csv", sep='\t')
            for yi in range(len(y_pred_c)):
                y_pred[yi] = y_unique[y_pred_c[yi]]
                if y_pred[yi] >= y_test[yi] / 2 and y_pred[yi] <= y_test[yi] * 2:
                    y_c[yi] = 1
                    sum = sum + 1
                else:
                    y_c[yi] = 0

            report = list(zip(y_test, y_pred, y_c))
            report = pd.DataFrame(report, columns=['test', 'pred', 'correct or not'])
            report.to_csv(self.path + output + "/" + "DL_class.csv", sep='\t')

        print("Done!")
        return sum/len(y_pred)



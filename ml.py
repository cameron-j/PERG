import data_processing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt

def main():
    record_ids = data_processing.read_normal_data()

    X_list = []
    Y_list = []

    for record_id in record_ids:
        r = data_processing.Record(record_id)
        avg_re = np.array(list(map(np.average, r.re.T)))
        avg_le = np.array(list(map(np.average, r.le.T)))

        re = data_processing.WaveformData(r.times[0], avg_re)
        le = data_processing.WaveformData(r.times[0], avg_le)

        if not(np.isnan(r.va_re)):
            X_list.append((# re.N35_A,
                           re.N35_implicit_t,
                           # re.N35_latency,
                           re.P50_A,
                           re.P50_implicit_t,
                           # re.P50_latency,
                           re.N95_A,
                           re.N95_implicit_t,
                           # re.N95_latency
                           ))
            Y_list.append(r.va_re)

        if not(np.isnan(r.va_le)):
            X_list.append((# le.N35_A,
                           le.N35_implicit_t,
                           # le.N35_latency,
                           le.P50_A,
                           le.P50_implicit_t,
                           # le.P50_latency,
                           le.N95_A,
                           le.N95_implicit_t,
                           # le.N95_latency
                           ))
            Y_list.append(r.va_le)

    # print(np.array(sorted(list(zip(map(lambda x: "Mercury poisoning" if x==1 else "Normal", diagnosis_counts.keys()), diagnosis_counts.values())), key=lambda x: x[1])))

    x_train, x_test, y_train, y_test = train_test_split(X_list, Y_list, train_size=0.7)

    # Support vector regression
    svr_model = SVR()
    y_pred_svr = svr_model.fit(x_train, y_train).predict(x_test)
    plt.scatter(y_test, y_pred_svr, label="Support Vector Regression")

    # Support vector regression
    sgd_model = SGDRegressor()
    y_pred_sgd = sgd_model.fit(x_train, y_train).predict(x_test)
    plt.scatter(y_test, y_pred_sgd, label="Stochastic Gradient Descent")

    # Linear regression
    linear_regression_model = LinearRegression()
    y_pred_linear_regression = linear_regression_model.fit(x_train, y_train).predict(x_test)
    plt.scatter(y_test, y_pred_linear_regression, label="Linear Regression")

    x_ax = np.linspace(min(y_pred_linear_regression), max(y_pred_linear_regression), 100)
    plt.plot(x_ax, x_ax)
    plt.xlabel("Actual visual acutity")
    plt.ylabel("Predicted visual acutity")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

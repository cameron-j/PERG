import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
import datetime
import random

participants_df = pd.read_csv("dataset/csv/participants_info.csv")

def str_to_datetime(datetime_str):
    date_str, time_str = datetime_str.split(" ")
    date_lst = date_str.split("-")
    time_lst = time_str.split(":")
    time_lst = time_lst[:-1] + time_lst[-1].split(".")
    return datetime.datetime(int(date_lst[0]), int(date_lst[1]), int(date_lst[2]), int(time_lst[0]), int(time_lst[1]), int(time_lst[2]), int(time_lst[3]))

class Record:
    def __init__(self, id):
        data = participants_df[participants_df["id_record"]==id]
        self.id = id
        self.date = data["date"].to_string(index=False)
        self.age = int(data["age_years"].to_string(index=False))
        self.sex = data["sex"].to_string(index=False)
        self.diagnoses = [data["diagnosis1"].to_string(index=False), data["diagnosis2"].to_string(index=False), data["diagnosis3"].to_string(index=False)]
        while "NaN" in self.diagnoses:
            self.diagnoses.remove("NaN")

        self.va_re = float(data["va_re_logMar"].to_string(index=False)) # Visual acutity of right eye (0 normal, +ve worse, -ve better)
        self.va_le = float(data["va_le_logMar"].to_string(index=False))
        self.unilateral = data["unilateral"].to_string(index=False)
        if self.unilateral == "NaN":
            self.unilateral = None
        self.rep_record = data["rep_record"].to_string(index=False)
        if self.rep_record == "NaN":
            self.rep_record = None
        self.comments = data["comments"].to_string(index=False)
        if self.comments == "NaN":
            self.comments = None

    def __repr__(self):
        return f"""RECORD {self.id}:
\tDate of test: {self.date}
\tParticipant age: {self.age}
\tSex: {self.sex}
\tDiagnoses: {self.diagnoses}
\tVisual acutity of right eye: {self.va_re}
\tVisual acutity of left eye: {self.va_le}
\tUnilateral: {self.unilateral}
\tOther records: {self.rep_record}
\tComments: {self.comments}
"""

    # Returns (times, right eye, left eye)
    def read_PERG(self):
        df = pd.read_csv(f"dataset/csv/{str(self.id).zfill(4)}.csv")
        n_tests = int(df.columns[-1][-1])

        times = []
        re = []
        le = []
        for i in range(n_tests):
            test_times = []
            test_re = []
            test_le = []
            time_col = df["TIME_" + str(i+1)].to_list()
            re_col = df["RE_" + str(i+1)].to_list()
            le_col = df["LE_" + str(i+1)].to_list()

            start_time = str_to_datetime(time_col[0])

            for j, time in enumerate(time_col):
                time = str_to_datetime(time_col[j])
                test_times.append((time - start_time) / datetime.timedelta(milliseconds=1))
                test_re.append(float(re_col[j]))
                test_le.append(float(le_col[j]))

            times.append(test_times)
            re.append(test_re)
            le.append(test_le)
        return np.array(times), np.array(re), np.array(le)

for record_id in range(1, len(participants_df)+1):
    print(record_id)
    record = Record(record_id)
    if record.diagnoses == ["Normal"]:
        print(record)
        times, re, le = record.read_PERG()
        avg_times = np.array(list(map(np.average, times.T)))
        avg_re = np.array(list(map(np.average, re.T)))
        avg_le = np.array(list(map(np.average, le.T)))

        plt.plot(avg_times, avg_re, label="Right eye")
        plt.plot(avg_times, avg_le, label="Left eye")
        plt.legend()
        plt.show()

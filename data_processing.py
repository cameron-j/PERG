import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
import datetime

FILTER_ORDER = 12
FILTER_CUTOFF = 0.95

N35_IMPLICIT_T_ERROR = 0.4 # Maximum allowable deviation from normal implicit times
P50_IMPLICIT_T_ERROR = 0.5 # Maximum allowable deviation from normal implicit times
N95_IMPLICIT_T_ERROR = 1 # Maximum allowable deviation from normal implicit times

RESPONSE_ONSET_RATIO = 0.2 # Amplitude multiplied by this value gives the value of the response onset

participants_df = pd.read_csv("dataset/csv/participants_info.csv")

def str_to_datetime(datetime_str):
    date_str, time_str = datetime_str.split(" ")
    date_lst = date_str.split("-")
    time_lst = time_str.split(":")
    time_lst = time_lst[:-1] + time_lst[-1].split(".")
    return datetime.datetime(int(date_lst[0]), int(date_lst[1]), int(date_lst[2]), int(time_lst[0]), int(time_lst[1]), int(time_lst[2]), int(time_lst[3]))

class Record:
    def __init__(self, id, read_data=True):
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

        if read_data:
            self.times, self.re, self.le = self.read_PERG()

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


def filter_data(data):
    b, a = butter(FILTER_ORDER, FILTER_CUTOFF, btype="lowpass")
    return filtfilt(b, a, data)


class WaveformData:
    def __init__(self, times, data):
        self.times = times
        self.data = filter_data(data)

        # Finds the implicit times (stimulus onset to maximum amplitude)
        P50_idx = np.where(data==max(data))[0][0]
        N35_idx = np.where(data==min(data[:P50_idx]))[0][0]
        self.P50_implicit_t = times[P50_idx]
        self.N35_implicit_t = times[N35_idx]
        N95_idx = P50_idx
        min_val = min(data[P50_idx:])
        while data[N95_idx] != min_val:
            N95_idx += 1
        self.N95_implicit_t = times[N95_idx]

        # Finds the response amplitudes
        self.N35_A = abs(self.data[N35_idx])
        self.P50_A = self.data[P50_idx] - self.data[N35_idx]
        self.N95_A = self.data[P50_idx] - self.data[N95_idx]

        # Finds the latency (stimulus onset to response onset)
        i = 0
        while i < len(self.data) - 1 and self.data[i] > self.data[N35_idx] * RESPONSE_ONSET_RATIO:
            i += 1
        N35_onset_idx = i

        i = N35_idx
        while i < len(self.data) - 1 and self.data[i] - self.data[N35_idx] < self.P50_A * RESPONSE_ONSET_RATIO: # P50 latency
            i += 1
        P50_onset_idx = i

        i = P50_idx
        while i < len(self.data) - 1 and self.data[P50_idx] - self.data[i] < self.N95_A * RESPONSE_ONSET_RATIO: # N95 latency
            i += 1
        N95_onset_idx = i

        self.N35_latency = times[N35_onset_idx]
        self.P50_latency = times[P50_onset_idx]
        self.N95_latency = times[N95_onset_idx]


    def __repr__(self):
        return f"""Waveform:
    Implicit times:
        N35: {self.N35_implicit_t}
        P50: {self.P50_implicit_t}
        N95: {self.N95_implicit_t}
    Amplitudes:
        N35: {self.N35_A}
        P50: {self.P50_A}
        N95: {self.N95_A}
    Latencies
        N35: {self.N35_latency}
        P50: {self.P50_latency}
        N95: {self.N95_latency}"""


def check_times_equal(times):
    t0 = times[0]
    for time in times:
        if not((time == t0).all()):
            return False
    return True

def check_normal_data(data: WaveformData):
    result = True
    # Checks positions of peaks
    if abs(data.N35_implicit_t - 0.35) > N35_IMPLICIT_T_ERROR:
        print(f"N35 implicit time not in acceptable range ({data.N35_implicit_t})")
        result = False

    if abs(data.P50_implicit_t - 0.5) > P50_IMPLICIT_T_ERROR:
        print(f"P50 implicit time not in acceptable range ({data.P50_implicit_t})")
        result = False

    if abs(data.N95_implicit_t - 0.95) > N95_IMPLICIT_T_ERROR:
        print(f"N95 implicit time not in acceptable range ({data.N95_implicit_t})")
        result = False

    return result

# Writes the ids of all of the records with normal data to normal_record_ids.txt
def write_normal_records():
    normal_records = []
    for record_id in range(1, len(participants_df)+1):
        record = Record(record_id)
        if check_times_equal(record.times):
            try:
                # Averages the data
                avg_re = np.array(list(map(np.average, record.re.T)))
                avg_le = np.array(list(map(np.average, record.le.T)))
                re = WaveformData(record.times[0], avg_re)
                le = WaveformData(record.times[0], avg_le)

                # Checks that the data fits the normal pattern
                if check_normal_data(re) and check_normal_data(le):
                    print("Normal data: ", record.id)
                    normal_records.append(record.id)
                else:
                    print("Abnormal data: ", record.id)

            except Exception as _:
                print("Failed:", record.id)

    # Writes the ids of records with normal ids to the file
    normal_records_f = open("normal_record_ids.txt", "w")
    normal_records_str = ""
    for record in normal_records:
        normal_records_str += str(record) + " "
    normal_records_f.write(normal_records_str[:-1])
    normal_records_f.close()

def plot_waveform(data: WaveformData, ax, label=""):
    ax.plot(data.times, data.data, label=label)
    # Implicit times
    ax.axvline(x=data.N35_implicit_t, color="red")
    ax.axvline(data.P50_implicit_t, color="red")
    ax.axvline(data.N95_implicit_t, color="red")
    # Latencies
    ax.axvline(x=data.N35_latency, color="green")
    ax.axvline(data.P50_latency, color="green")
    ax.axvline(data.N95_latency, color="green")

def read_normal_data():
    # Reads normal data records from file
    normal_records_f = open("normal_record_ids.txt", "r")
    normal_record_ids = map(int, normal_records_f.read().split(" "))
    normal_records_f.close()
    return normal_record_ids

def main():
    for record_id in read_normal_data():
        fig, ax = plt.subplots()
        r = Record(record_id)
        avg_re = np.array(list(map(np.average, r.re.T)))
        avg_le = np.array(list(map(np.average, r.le.T)))

        re = WaveformData(r.times[0], avg_re)
        le = WaveformData(r.times[0], avg_le)

        print(re)
        print(le)
        plot_waveform(re, ax, label="Right eye")
        plot_waveform(le, ax, label="Left eye")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    write_normal_records()

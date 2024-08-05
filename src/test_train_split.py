"""Splitting data into honest and fraudulent data for training of model"""

columns = df.columns.tolist()
target = "Class"
columns = [c for c in columns if c not in ["Class", "Time", "Amount"]]
fraudulent_data = df[df['Class'] == 1]
honest_data = df[df['Class'] == 0]
X = honest_data[columns]
y = honest_data[target]
R = fraudulent_data[columns]
s = fraudulent_data[target]
print(X.columns)
print(R.columns)
row_count = X.shape[0]

X_train = X[:int(row_count*0.8)]
X_test = X[int(row_count*0.8):]
check1_test = X[int(row_count*0.01):]

y_train = y[:int(row_count*0.8)]
y_test = y[int(row_count*0.8):]
check2_test = y[int(row_count*0.01):]

check1 = np.concatenate([check1_test, R])
check2 = np.concatenate([check2_test, s])
# Shuffle the combined array randomly
np.random.shuffle(check1)
np.random.shuffle(check2)
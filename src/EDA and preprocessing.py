"""#### Data Pre-processing

The data set is read and the first 5 rows are printed
"""

df = pd.read_csv("creditcard.csv")
df.head()

"""We define a function to scale the data set value between 0 and 1"""

def min_max_scaling(data):
    min_vals=np.min(data, axis=0)
    max_vals=np.max(data, axis=0)
    scaled_data=(data - min_vals) / (max_vals - min_vals)
    return scaled_data

df=min_max_scaling(df)
df.head()

"""#### Exploratory Data Analysis (EDA)

Plotting Histograms to get more information about the data set
"""

count_classes = pd.value_counts(df['Class'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Class wise Histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")

hist = df.hist(bins=100, figsize = (20,20))

df.describe()

df.info()
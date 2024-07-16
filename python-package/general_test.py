import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split

# Load your CSV data (adjust file path and column names)
data = pd.read_csv("learn2clean/datasets/gbsg.csv").dropna()
print(data)
#data.drop(['novator', 'independ', 'extraversion', 'greywage', 'head_gender', 'coach', 'traffic', 'industry'], axis=1, inplace=True)
event = 'status'
time = 'age'
y = data[[event, time]]
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

cph = CoxPHFitter(penalizer=0.1)
print(X_train)
cph.fit(X_train, duration_col=time, event_col=event)

# Generate C-index
c_index = concordance_index(y_test[time], -cph.predict_partial_hazard(X_test))

# Print summary, hazard ratios, and C-index
print(cph.print_summary())
print("\nC-index:", c_index) #0.7376835236541599
print()

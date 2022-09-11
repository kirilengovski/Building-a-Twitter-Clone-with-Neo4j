import matplotlib
import matplotlib.pyplot as plt
import numpy as np

profile_kiril = [34315, 34315, 34315, 34315, 34315]
profile_max = [18714, 31017, 43326, 55620, 67934]

n_groups = 5

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, profile_kiril, bar_width,
alpha=opacity,
color='b',
label="Kiri's model")

rects2 = plt.bar(index + bar_width, profile_max, bar_width,
alpha=opacity,
color='r',
label="Max De Marzi's model")

plt.xlabel('Number of days processed')
plt.ylabel('DB Hits')
plt.title('Comparison of number of DB Hits the newsfeed retrieval query\n requires to display an average Twitter user newsfeed')
plt.xticks(index + bar_width, ('1', '2', '3', '4', '5'))
plt.legend()

plt.show()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

profile_kiril = [91.029, 88.817, 87.202, 88.099]
profile_max = [92.725, 76.904, 58.902, 43.731]

num_of_posts_per_day = [5, 25, 50, 100]

fig, ax = plt.subplots()


line_up, =plt.plot(num_of_posts_per_day, profile_max, 'ro-')
line_down, =plt.plot(num_of_posts_per_day, profile_kiril, 'bo-')
plt.ylabel("Number of posts per day per user")
plt.xlabel("Average number of user profiles retrieved")
plt.legend([line_down, line_up], ["Kiril's model", "Max de Marzi's model"])
plt.xticks(np.arange(0, 101, step=5))
plt.yticks(np.arange(0, 101, step=5))
plt.title("Comparison of the average number of profiles retrieved within 0.2\n seconds based on 10000 repetitions")

plt.tight_layout()

fig.set_size_inches(w=5.8, h=4.5)

plt.show()

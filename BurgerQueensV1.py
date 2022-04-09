import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

train_csv = pd.read_csv('/groups/CS156b/data/student_labels/train.csv')
for row in train_csv:
    path = '/groups/CS156b/data/' + train_csv['Path'][row]
    image = mpimg.imread(path)
#     plt.show()
#     plt.imshow(image)
# test_csv = pd.read_csv('/groups/CS156b/data/test')
# solution_csv = pd.read_csv('/groups/CS156b/data/solution')

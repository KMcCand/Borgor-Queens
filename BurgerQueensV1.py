import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

train_csv = pd.read_csv('/groups/CS156b/data/student_labels/train.csv')
for row in range(len(train_csv)):
    path = '/groups/CS156b/data/' + train_csv['Path'][row]
    image = mpimg.imread(path)
    
test_csv = pd.read_csv('/groups/CS156b/data/student_labels/test.csv')
for row in range(len(test_csv)):
    path = '/groups/CS156b/data/' + test_csv['Path'][row]
    image = mpimg.imread(path)
# test_csv = pd.read_csv('/groups/CS156b/data/test')
# solution_csv = pd.read_csv('/groups/CS156b/data/solution')

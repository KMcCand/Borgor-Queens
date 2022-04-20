import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
%matplotlib inline

train_csv = pd.read_csv('/groups/CS156b/data/student_labels/train.csv')
for row in range(1, 100):
    path = '/groups/CS156b/data/' + train_csv['Path'][row]
    image = mpimg.imread(path)
    print(train_csv['Path'][row])
    plt.imshow(image)
    plt.show()
    

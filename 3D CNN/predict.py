import tensorflow as tf
import numpy as np
import os 
from tensorflow.keras.models import Sequential, save_model, load_model

classes = ["Pick","Stitch","Person"]
positions = ["Left"]

path_model = "3D CNN\\Checkpoint\\pick.ckpt" 
path_data = "3D CNN\\Data"

def load_data() : 
    videos = []
    labels = []
    for cls in classes : 
        for pos in positions : 
            for file in os.listdir(f"{path_data}\\{cls}\\{pos}") :
                vid = np.load(os.path.join(f"{path_data}\\{cls}\\{pos}",file))
                videos.append(vid)
                labels.append(classes.index(cls))
    videos = np.array(videos,dtype=np.float32)

    videos = videos.reshape(videos.shape[0], 8, 64, 64, 1)

    return videos,labels

videos,labels = load_data()

#tf.keras.models.load_model(f'{path_model}\\model')

model = load_model(f'{path_model}\\model', compile = True)

predictions = model.predict(videos)
print(predictions)
for i in range(len(predictions)) : 
    print("Actual    :",classes[labels[i]])
    print("Predicted :",classes[np.argmax(predictions[i])])
    print()
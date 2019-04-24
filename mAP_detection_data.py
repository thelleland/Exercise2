from keras.models import load_model
import make_model
import os
import cv2
import numpy as np
from utils import decode_netout

LABELS = ['BlueWhiting', 'Mackerel', 'Benthosema', 'Herring']

IMAGE_H, IMAGE_W = 416, 416
GRID_H,  GRID_W  = 13 , 13
BOX              = 2
CLASS            = len(LABELS)
CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')
OBJ_THRESHOLD    = 0.5#0.5,
NMS_THRESHOLD    = 0.45#0.45
ANCHORS          = [1.02, 2.37, 2.80, 6.89]

TRUE_BOX_BUFFER = 50

def make_detection_data(model):
    path_to_count = os.getcwd() + "/Data/test_annot"
    number_of_images = len([name for name in os.listdir(path_to_count)])
    for i in range(number_of_images):
        if (i % 10 == 0):
            print("Made ", i, " detection data annotations")
        image_path = os.getcwd() + "/Data/test_imgs/test_{}.png".format(str(i+1))
        
        image = cv2.imread(image_path)
        image_h, image_w, _ = image.shape

        dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER, 4))
        input_image = cv2.resize(image, (416,416))
        input_image = input_image[:,:,::-1]
        input_image = np.expand_dims(input_image, 0)

        netout = model.predict([input_image, dummy_array])
        boxes = decode_netout(netout[0],
                              obj_threshold=OBJ_THRESHOLD,
                              nms_threshold=NMS_THRESHOLD,
                              anchors=ANCHORS,
                              nb_class=CLASS)

        log = []
        for box in boxes:
            label = LABELS[box.get_label()]
            score = box.get_score()          
            xmin = int(box.xmin*image_w)
            ymin = int(box.ymin*image_h)
            xmax = int(box.xmax*image_w)
            ymax = int(box.ymax*image_h)
            
            log = log + ["{} {} {} {} {} {}\n".format(label,score,xmin,ymin,xmax,ymax)]
            


        
        path  = "./mAP-master/input/detection-results/img_{}.txt".format(str(i + 1))
        with open(path,'w+') as f:
            for l in log: f.write(l)
    return



def main():

    model = make_model.makemodel()
    model.load_weights("./weights/2_anchors_weights.h5")

    make_detection_data(model)

    

if __name__ == "__main__":
    main()




from pascal_voc_writer import Writer
import sys
import getopt
import os, os.path


def get_paths():
    in_dir = None
    out_dir = None

    argv = sys.argv[1:]

    
    # i: path to images and annotations, o: path to output directory
    try:
        opts, args = getopt.getopt(argv, "i:o:")
    except getopt.GetoptError as err:
        print(err)
        opts = []
        
    for opt, arg in opts:
        if opt in ['-i']: 
            in_dir = arg        
        elif opt in ['-o']:
            out_dir = arg

    return in_dir, out_dir

# adding objects to image
def add_objects(anno_path, writer):
    f = open(anno_path, "r")
    lines = f.readlines()
    
    for x in lines:
        columns = x.split('\t')
        object_name = columns[0]
        xmin = int(columns[2])
        ymin = int(columns[3])
        xmax = int(columns[4])
        ymax = int(columns[5])
        writer.addObject(object_name, xmin, ymin, xmax, ymax)
        
    return


in_dir, out_dir = get_paths()

number_of_images = len([name for name in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, name))]) / 2

number_of_images = int(number_of_images)


# creating data in VOC format
for i in range(1, number_of_images + 1):
    image_path = "{}test_{}.png".format(in_dir, str(i))
    anno_path = "{}test_{}.txt".format(in_dir, str(i))
    writer = Writer(image_path, 1392, 1040)
    add_objects(anno_path, writer)
    writer.save("{}img_{}.xml".format(out_dir, str(i)))
    

from struct import unpack
from numpy import zeros, uint8, float32
import matplotlib.pyplot as plt
from scipy import misc

def get_data():
    images = open('train-images.idx3-ubyte', 'rb')
    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]
    img_500 = 50
    # Get image datas
    img_data = zeros((img_500, rows, cols), dtype=float32)
    for i in range(img_500):
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                img_data[i][row][col] = float(tmp_pixel)/255
        img_data[i][img_data[i] >= 0.5] = 1
        img_data[i][img_data[i] < 0.5] = -1
        print img_data[i]
    cdd = 1  

def view_image(image):
    misc.imsave('imgs/cdd.png', image)
    
def main(): 
    get_data()   
    cdd = 1

if __name__ == "__main__":
    main()
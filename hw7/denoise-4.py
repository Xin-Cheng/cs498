import numpy as np
import matplotlib.pyplot as plt
from numpy import zeros, uint8, float32
from struct import unpack
from scipy import misc
from random import randint
from matplotlib.pylab import *
from math import exp

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
    theta_hhs = [-1, -0.6, -0.3, 0.3, 0.6, 1]
    img_500 = 500
    # Get image datas
    TPRs = zeros((len(theta_hhs), img_500), dtype=float32)
    FPRs = zeros((len(theta_hhs), img_500), dtype=float32)
    img_data = zeros((img_500, rows, cols), dtype=float32)
    noised_imgs = zeros((img_500, rows, cols), dtype=float32)
    denoised_imgs = zeros((len(theta_hhs), img_500, rows, cols), dtype=float32)
    for i in range(img_500):
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                img_data[i][row][col] = float(tmp_pixel)/255
        img_data[i][img_data[i] >= 0.5] = 1
        img_data[i][img_data[i] < 0.5] = -1
        noised_imgs[i] = create_noise(img_data[i])
        # try different theta_hi_hj
        for theta in range(len(theta_hhs)):
            denoised_imgs[theta][i] = denoise(noised_imgs[i], theta_hhs[theta])
            N = len(img_data[i][ np.where( img_data[i] == -1 ) ])
            P = len(img_data[i][ np.where( img_data[i] == 1 ) ])
            TP = 0
            FP = 0
            for r in range(rows):
                for c in range(cols):
                    if denoised_imgs[theta][i][r][c] == 1 and img_data[i][r][c] == 1:
                        TP += 1
                    if denoised_imgs[theta][i][r][c] == 1 and img_data[i][r][c] == -1:
                        FP += 1
            TPRs[theta][i] = float(TP)/P
            FPRs[theta][i] = float(FP)/N
    plot_ROC(theta_hhs, TPRs, FPRs)

# plot accuracy rate
def plot_ROC(theta_hhs, TPRs, FPRs):
    TPRs.sort(axis=1)
    FPRs.sort(axis=1)
    plt.figure()
    for i in range(6):
        plt.plot(FPRs[i], TPRs[i], lw=2, label='Theta = %0.2f' % theta_hhs[i])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic of Different Thetas')
    plt.legend(loc="lower right")
    plt.savefig("ROC.png", bbox_inches='tight', pad_inches=0.2)

# create a noisy version by randomly flipping 2% of the bits
def create_noise(image):
    noised_imgs = np.copy(image)
    for i in range(28):
        for j in range(28):
            rand = randint(0, 99)
            if rand < 2:
                noised_imgs[i][j] = -noised_imgs[i][j]
    return noised_imgs

def denoise(image, theta_hh):
    dim = 28
    theta_hx = 2
    # initialize probability
    probabilities = np.full((dim, dim), 0.5)
    energy = 0
    energy_prev = 1
    while abs(energy - energy_prev) > 0.001:
        energy_prev = energy
        energy = 0
        for r in range(dim):
            for c in range(dim):
                first_up = 0 if r == 0 else theta_hh*(2*probabilities[r - 1][c] - 1)
                first_down = 0 if r == dim - 1 else theta_hh*(2*probabilities[r + 1][c] - 1)
                first_left = 0 if c == 0 else theta_hh*(2*probabilities[r][c - 1] - 1)
                first_right = 0 if c == dim - 1 else theta_hh*(2*probabilities[r][c + 1] - 1)
                first_h = theta_hx*(2*probabilities[r][c] - 1)
                second_up = 0 if r == 0 else theta_hx*image[r - 1][c]
                second_down = 0 if r == dim - 1 else theta_hx*image[r + 1][c]
                second_left = 0 if c == 0 else theta_hx*image[r][c - 1]
                second_right = 0 if c == dim - 1 else theta_hx*image[r][c + 1]
                second_x = theta_hx*image[r][c]
                log_positive = first_up + first_down + first_left + first_right + first_h + second_up + second_down + second_left + second_right + second_x
                probabilities[r][c] = exp(log_positive)/(exp(log_positive) + exp(0 - log_positive))
                energy += probabilities[r][c]*log_positive
    return construct_image(image, probabilities)
        
def construct_image(image, probabilities):
    image = probabilities
    image[image > 0.5] = 1
    image[image < 0.5] = -1
    return image
    
def main(): 
    get_data()

if __name__ == "__main__":
    main()
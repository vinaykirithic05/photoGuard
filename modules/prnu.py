import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt
import os

class PRNUAnalyzer:

    def analyze(self, image_path):

        print("\n========== SENSOR NOISE ANALYSIS ==========\n")

        img = cv2.imread(image_path, 0)

        coeffs = pywt.dwt2(img, 'haar')

        LL, (LH, HL, HH) = coeffs

        residual = HH

        variance = np.var(residual)

        print("Noise Variance :", round(variance,2))

        if variance > 30:

            print("Sensor Pattern : Strong")

            print("Prediction : Likely Real")

        else:

            print("Sensor Pattern : Weak")

            print("Prediction : Possible AI")

        os.makedirs("outputs", exist_ok=True)

        plt.imshow(residual, cmap='gray')
        plt.axis("off")
        plt.savefig("outputs/prnu_noise.png")
        plt.close()

        print("\nNoise Image Saved -> outputs/prnu_noise.png")
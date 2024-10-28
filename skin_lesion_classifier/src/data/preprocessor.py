
import cv2
import numpy as np


class ImagePreprocessor:
    @staticmethod
    def hair_remove(image: np.ndarray) -> np.ndarray:
        """Remove hair artifacts from skin lesion images."""

        grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        kernel = cv2.getStructuringElement(1, (17, 17))

        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

        _, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

        final_image = cv2.inpaint(image, threshold, 1, cv2.INPAINT_TELEA)

        return final_image

    @staticmethod
    def preprocess_image(image: np.ndarray, remove_hair: bool = True) -> np.ndarray:

        if remove_hair:
            image = ImagePreprocessor.hair_remove(image)
        return image

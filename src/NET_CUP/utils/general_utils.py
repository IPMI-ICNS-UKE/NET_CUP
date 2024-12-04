# Base dependencies
import PIL
import PIL.Image
from typing import Union, List
# Other dependencies
import SimpleITK.SimpleITK
import numpy as np
import SimpleITK
import sklearn.metrics
import seaborn as sns


def convert_to_pil(img: Union[str, PIL.Image.Image, SimpleITK.SimpleITK.Image, np.ndarray]) -> PIL.Image.Image:
    """
    Converts an input image in various formats (file path, PIL image, SimpleITK image, or numpy array)
    to a PIL.Image.Image object.

    :param img: Image in one of the supported formats.
    :return: PIL image.
    """
    if isinstance(img, str):
        pil_img = PIL.Image.open(img)
    elif isinstance(img, PIL.Image.Image):
        pil_img = img
    elif isinstance(img, SimpleITK.SimpleITK.Image):
        img_arr = SimpleITK.GetArrayFromImage(img)
        pil_img = PIL.Image.fromarray(img_arr)
    elif isinstance(img, np.ndarray):
        pil_img = PIL.Image.fromarray(img)
    return pil_img


def convert_to_sitk(img: Union[str, PIL.Image.Image, SimpleITK.SimpleITK.Image, np.ndarray]) -> SimpleITK.Image:
    """
    Converts an input image in various formats (file path, PIL image, SimpleITK image, or numpy array)
    to a SimpleITK.Image object.

    :param img: Image in one of the supported formats.
    :return: SimpleITK image.
    """
    if isinstance(img, str):
        sitk_img = SimpleITK.ReadImage(img)
    elif isinstance(img, PIL.Image.Image):
        img_arr = np.array(img)
        sitk_img = SimpleITK.GetImageFromArray(img_arr,
                                               isVector=True)  # isVector=True important! otherwise treated as 3D
    elif isinstance(img, SimpleITK.SimpleITK.Image):
        sitk_img = img
    elif isinstance(img, np.ndarray):
        sitk_img = SimpleITK.GetImageFromArray(img)
    return sitk_img


def convert_to_arr(img: Union[str, PIL.Image.Image, SimpleITK.SimpleITK.Image, np.ndarray]) -> np.ndarray:
    """
    Converts an input image in various formats (file path, PIL image, SimpleITK image, or numpy array)
    to a numpy array.

    :param img: Image in one of the supported formats.
    :return: Numpy array representation of the image.
    """
    if isinstance(img, str):
        pil_img = PIL.Image.open(img)
        img_arr = np.array(pil_img)
    elif isinstance(img, PIL.Image.Image):
        img_arr = np.array(img)
    elif isinstance(img, SimpleITK.SimpleITK.Image):
        img_arr = SimpleITK.GetArrayFromImage(img)
    elif isinstance(img, np.ndarray):
        img_arr = img
    return img_arr


def convert_e_number_format(e_number: str, seperations: bool) -> str:
    """
    Converts an E-number to a desired format with or without separators.

    :param e_number: E-number in any of the two formats
    :param seperations: True -> Convert to E-Number with / seperations
                        False -> Converr to E-Number without / seperations
    :return: E-number in the desired format
    """

    if seperations:
        if "/" in e_number:
            converted_e_number = e_number
        else:
            converted_e_number = f'E/{e_number[1:5]}/{e_number[5:11]}'
    else:
        if "/" in e_number:
            converted_e_number = f'E{e_number[2:6] + e_number[7:14]}'
        else:
            converted_e_number = e_number
    return converted_e_number


def split_integer(n: int, k: int) -> List[int]:
    """
    Splits up an Integer n in k equal parts, the rest r is distributed evenly among the first r parts


    :param n: Integer to split
    :param k: Number of parts
    :return: List with the size of k parts
    """
    distribution = []
    for i in range(k):
        distribution.append(int(n / k))
    for i in range(n % k):
        distribution[i] = distribution[i] + 1
    return distribution


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]):
    """
    Generates a confusion matrix heatmap with absolute and relative values annotated.
    
    :param y_true: Ground truth target values.
    :param y_pred: Predicted target values.
    :param labels: List of class labels for the confusion matrix axes.
    :return: Seaborn heatmap plot of the confusion matrix.
    """
    cfm_abs = sklearn.metrics.confusion_matrix(y_true, y_pred)
    cfm_rel = sklearn.metrics.confusion_matrix(
        y_true, y_pred, normalize='true')
    cfm_rel = np.round(cfm_rel, 4)
    cfm = np.array([f'{absolute_value} \n \n ({relative_value})' for absolute_value, relative_value in
                    zip(cfm_abs.ravel(), cfm_rel.ravel())]).reshape(cfm_abs.shape)
    return sns.heatmap(cfm_rel, xticklabels=labels, yticklabels=labels, fmt='', annot=cfm, annot_kws={}).set(
        xlabel='Predicted origin', ylabel='True origin')

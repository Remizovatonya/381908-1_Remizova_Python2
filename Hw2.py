# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 13:41:33 2021

@author: Rearo
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


"""
Функция конвертации из BGR в RGB
"""
def Convert(image_name):
    img = image_name
    b, g, r = cv.split(img) # get b, g, r
    img_rgb = cv.merge([r, g, b]) # switch it to rgb
    return img_rgb


"""
Функция вывода изображения на экран
"""
def Output(img_rgb, img_blur, img_mask):
    plt.subplot(131), plt.imshow(img_rgb)
    plt.subplot(132), plt.imshow(img_blur)
    plt.subplot(133), plt.imshow(img_mask)
    plt.show()
    
    
"""
Функция для вывода всех функций
"""
def OutputAll(image_name, image_number):
    Denoising(image_name, image_number)
    Bilateral(image_name, image_number)
    Gaussian(image_name, image_number)
    Median(image_name, image_number)



"""
Удаление шума методом Non Local Means
"""
def Denoising(image_name, image_number):
    img = image_name
    nmb = image_number
    # Denoising 
    denoising_blur = cv.fastNlMeansDenoisingColored(Convert(img), None, 10, 10, 7, 21)
    mask = CalcOfDamageAndNonDamage(denoising_blur, nmb)
    Output(Convert(img), denoising_blur, mask)


"""
Билатеральный фильтр
"""
def Bilateral(image_name, image_number):
    img = image_name
    nmb = image_number
    # diameter=15, sigmaColor = sigmaSpace = 75.
    bilateral_blur = cv.bilateralFilter(Convert(img), 15, 75, 75)
    mask = CalcOfDamageAndNonDamage(bilateral_blur, nmb)
    Output(Convert(img), bilateral_blur, mask)


"""
Фильтр Гаусса
"""
def Gaussian(image_name, image_number):
    img = image_name
    nmb = image_number
    # фильтр Гаусса встроен в систему
    gaussian_blur = cv.GaussianBlur(Convert(img),(3,3),1)
    mask = CalcOfDamageAndNonDamage(gaussian_blur, nmb)
    Output(Convert(img), gaussian_blur, mask)


"""
Медианный фильтр
"""
def Median(image_name, image_number):
    img = image_name
    nmb = image_number
    # медианный фильтр встроен в систему
    median_blur = cv.medianBlur(Convert(img), 5)
    mask = CalcOfDamageAndNonDamage(median_blur, nmb)
    Output(Convert(img), median_blur, mask)
    

"""
Функция для подбора маркеров
"""
def LeafMarkers(image_name, image_number):
    
    img = image_name
    nmb = image_number
    
    markers = np.zeros((img.shape[0], img.shape[1]), dtype ="int32")
    
    #Для определенных листьев подберем маркеры
    if nmb == 2: 
        markers[90:183, 90:180] = 255
        markers[236:255, 0:70] = 1
        markers[0:20, 0:20] = 1
        markers[0:20, 236:255] = 1
        markers[236:255, 236:255] = 1
    elif nmb == 9:
        markers[70:166, 45:140] = 255
        markers[236:255, 0:20] = 1
        markers[0:20, 0:20] = 1
        markers[0:20, 236:255] = 1
        markers[236:255, 236:255] = 1
    else:
        #Оставим дефолтные из презентации
        markers[90:140, 90:140] = 255
        markers[236:255, 0:20] = 1
        markers[0:20, 0:20] = 1
        markers[0:20, 236:255] = 1
        markers[236:255, 236:255] = 1
    return markers


"""
Функция для получения маски листьев и их повреждений
"""
def CalcOfDamageAndNonDamage(image_name, image_number):
    image = image_name
    number = image_number
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
    image_erode = cv.erode(image, kernel)
    #erode - морфологическое преобразование размытия (операция сужения)

    hsv_img = cv.cvtColor(image_erode, cv.COLOR_BGR2HSV) #меняет цветовое пространство на RGB
    #H-цветовой тон, S-насыщенность, V-светлота
    
    #watershed предназначен для отделения объектов на изображении от фона
    leafs_area_BGR = cv.watershed(image_erode, LeafMarkers(image, number))
    #Via inRange define healthy part of leaf
    healthy_part = cv.inRange (hsv_img, (36, 25, 25), (86, 255, 255)) #выделим цвет здорового листа
    shadow_part = cv.inRange (hsv_img, (0, 0, 0), (180, 255, 30)) #выделим черную тень
    #два параметра это левая правая граница диапазона цвета
    ill_part = leafs_area_BGR - healthy_part #выделим цвет больного листа
    
    
    mask = np.zeros_like(image, np.uint8) #возвращает массив нулей с формой и типом данных массива image 
    #np.unit определяет тип данных возвращаемого массива
    mask[leafs_area_BGR > 1] = (255, 0, 255)
    mask[ill_part > 1 ] = (0, 0, 255)
    mask[shadow_part > 1] = (0, 0, 0)
    return mask


"""
Конечный вывод преобразований изображения
"""
img = cv.imread("1.jpg")
OutputAll(img, 1)

img = cv.imread("2.jpg")
OutputAll(img, 2)

img = cv.imread("9.jpg")
OutputAll(img, 9)

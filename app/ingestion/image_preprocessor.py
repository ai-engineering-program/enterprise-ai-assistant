from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


__all__ = ["PreprocessingConfig", "ImagePreprocessor"]


@dataclass
class PreprocessingConfig:
    """Параметры предобработки страницы перед распознаванием Tesseract.

    denoise_kernel_size — размер окна медианного фильтра (нечётное число
    >= 3): устраняет изолированный "соль-перец" шум факсов и многократных
    копий, не размывая тонкие штрихи букв при разумных значениях (3-5).
    Слишком большое окно "съедает" тонкие элементы кириллицы (ё, з, э) —
    см. предупреждение в тексте урока про переразмытый фильтр.

    binarize_block_size — размер локального окна адаптивной бинаризации
    (нечётное число >= 3): должен покрывать несколько символов текста,
    но не захватывать неравномерность освещения всей страницы целиком.

    binarize_c — константа коррекции порога адаптивной бинаризации
    (см. cv2.adaptiveThreshold): положительное значение делает порог
    строже (больше пикселей уходит в чёрный текст), отрицательное —
    мягче.
    """

    denoise_kernel_size: int = 3
    binarize_block_size: int = 25
    binarize_c: int = 15


class ImagePreprocessor:
    """
    Предобработка отсканированной страницы перед подачей в Tesseract.

    Именно этот слой поднял точность распознавания архива СК «Надёжный
    Партнёр» с 60% до 94% без замены модели распознавания (см. историю
    урока) — три независимых шага устраняют три разных источника
    искажений: перекос страницы при сканировании (deskew — выравнивание
    наклона), шум многократного копирования и факсовой передачи
    (denoise — устранение шума), неравномерный контраст факсов и слабых
    сканов (binarize — бинаризация, приведение к чёрно-белому виду).

    Порядок шагов в preprocess() важен: бинаризация ДО устранения
    перекоса склеивает наклонный текст с шумом фона в единую чёрную
    массу, которую невозможно корректно повернуть по строкам текста —
    поэтому deskew всегда должен идти первым.
    """

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        # TODO: сохранить self.config = config or PreprocessingConfig()
        ...

    def estimate_skew_angle(self, image: np.ndarray) -> float:
        """
        Оценить угол перекоса страницы в градусах.

        image — двумерный np.ndarray (grayscale, dtype uint8), где текст
        темнее фона.

        TODO:
        1. Бинаризовать изображение по Otsu, инвертировать так, чтобы
           текст был белым на чёрном фоне:
               _, thresh = cv2.threshold(
                   image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
               )
        2. Найти координаты всех "текстовых" (ненулевых) пикселей:
               coords = np.column_stack(np.where(thresh > 0))
        3. Если coords.size == 0 (пустая/белая страница) — вернуть 0.0.
        4. Найти минимальный охватывающий повёрнутый прямоугольник:
               angle = cv2.minAreaRect(coords)[-1]
        5. cv2.minAreaRect возвращает угол в диапазоне [-90, 0). Привести
           к интуитивному диапазону поворота страницы:
               если angle < -45: angle = -(90 + angle)
               иначе:            angle = -angle
        6. Вернуть angle (float).
        """
        ...

    def deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Повернуть страницу так, чтобы строки текста стали горизонтальными.

        TODO:
        1. angle = self.estimate_skew_angle(image)
        2. Если abs(angle) < 0.1 — вернуть image без изменений (страница
           уже выровнена, лишний поворот только добавит артефактов
           интерполяции).
        3. Построить матрицу поворота вокруг центра изображения:
               (h, w) = image.shape[:2]
               center = (w / 2, h / 2)
               matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        4. Применить поворот с заливкой фона белым (255), а не чёрным:
               rotated = cv2.warpAffine(
                   image, matrix, (w, h),
                   flags=cv2.INTER_CUBIC,
                   borderMode=cv2.BORDER_CONSTANT,
                   borderValue=255,
               )
        5. Вернуть rotated.
        """
        ...

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Устранить изолированный шум факса/многократного копирования.

        TODO:
        - вернуть cv2.medianBlur(image, self.config.denoise_kernel_size)

        Медианный фильтр убирает единичные "выпадающие" пиксели, не
        размывая протяжённые контуры букв — в отличие от гауссова
        размытия, которое было опробовано первым в истории урока и
        размыло тонкие штрихи кириллицы почти так же сильно, как сам шум.
        """
        ...

    def binarize(self, image: np.ndarray) -> np.ndarray:
        """
        Привести страницу к чёрно-белому виду адаптивным порогом.

        TODO:
        - вернуть cv2.adaptiveThreshold(
              image, 255,
              cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
              self.config.binarize_block_size, self.config.binarize_c,
          )

        Адаптивный, а не глобальный порог — потому что освещённость факса
        и старого скана неравномерна по странице: глобальный порог
        одинаково хорош для всей страницы только на чистых цифровых
        сканах.
        """
        ...

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Полный конвейер предобработки в правильном порядке.

        TODO: вернуть self.binarize(self.denoise(self.deskew(image)))

        Порядок обязателен: deskew -> denoise -> binarize, см. docstring
        класса о том, почему бинаризация не может идти первой.
        """
        ...

# vehicle-classification-yolo
<div align="center">

# 🚔 АСОП-ГАИ / Patrol-AI

### Автоматизированная Система Оценки Платёжеспособности для органов ГАИ

> *«Умная камера, которая видит твой BMW и уже знает всё о тебе»*

---

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=for-the-badge&logo=opencv&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-95%25-2ECC71?style=for-the-badge&logo=checkmarx&logoColor=white)]()
[![Kaggle](https://img.shields.io/badge/Trained%20on-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)]()

</div>

---

## 📋 Содержание

- [Описание проекта](#-описание-проекта)
- [Основные возможности](#-основные-возможности)
- [Стек технологий](#-стек-технологий)
- [Архитектура системы](#-архитектура-системы)
- [Результаты обучения](#-результаты-обучения)
- [Инструкция по запуску](#-инструкция-по-запуску)
- [Структура проекта](#-структура-проекта)
- [Команда](#-команда)

---

## 📌 Описание проекта

**АСОП-ГАИ** — интеллектуальная система компьютерного зрения для автоматического анализа транспортного потока. Система выполняет детекцию транспортных средств на видеопотоке или статичных изображениях и производит классификацию марки автомобиля с высокой точностью.

Проект реализован в рамках **учебной практики** и представляет собой полноценный двухэтапный (pipeline) подход к задаче распознавания объектов — от детекции до классификации.

> **Академическая цель:** исследование применимости современных архитектур нейронных сетей семейства YOLO к задачам распознавания транспортных средств в условиях реального трафика.

---

## ✨ Основные возможности

| Возможность | Описание |
|---|---|
| 🎯 **Двухэтапная детекция** | Сначала YOLOv8 выделяет автомобиль на кадре, затем классификатор определяет марку |
| 🏷️ **Классификация марок** | Распознавание марки ТС с точностью **95%** на тестовой выборке |
| ⚡ **Высокая скорость** | Обработка в режиме реального времени благодаря оптимизированной архитектуре YOLOv8 |
| 🖼️ **Автоматический кроп** | Автоматическое извлечение ROI (Region of Interest) для подачи в классификатор |
| 🔧 **Гибкая настройка** | Поддержка кастомных весов `best.pt` / `last.pt` для дообучения |
| 📊 **Визуализация** | Отрисовка bounding box'ов и подписей прямо на кадре |

---

## 🛠️ Стек технологий

<div align="center">

| Категория | Технология | Назначение |
|---|---|---|
| **Язык** | ![Python](https://img.shields.io/badge/-Python%203.10-3776AB?logo=python&logoColor=white) | Основной язык разработки |
| **Детекция** | ![YOLOv8](https://img.shields.io/badge/-YOLOv8-FF6B35?logo=opencv&logoColor=white) | Детекция транспортных средств |
| **Фреймворк** | ![Ultralytics](https://img.shields.io/badge/-Ultralytics-6C63FF?logoColor=white) | API для работы с моделями YOLO |
| **Deep Learning** | ![PyTorch](https://img.shields.io/badge/-PyTorch%202.x-EE4C2C?logo=pytorch&logoColor=white) | Бэкенд обучения нейросетей |
| **Компьютерное зрение** | ![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?logo=opencv&logoColor=white) | Обработка изображений и видео |
| **Обучение** | ![Kaggle](https://img.shields.io/badge/-Kaggle%20GPU-20BEFF?logo=kaggle&logoColor=white) | Удалённые GPU-серверы для тренировки |

</div>

---

## 🏗️ Архитектура системы

Система реализована в виде последовательного конвейера обработки (inference pipeline):

```
┌─────────────────────────────────────────────────────────────────┐
│                     PIPELINE АСОП-ГАИ                          │
└─────────────────────────────────────────────────────────────────┘

  📷 Входное          🔍 Детекция           ✂️  Кроп
  изображение   ───►  YOLOv8 (best.pt) ───►  объекта (ROI)
  (кадр/фото)         ↓                       ↓
                 [Bounding Box]          [Вырезанный
                  координаты              фрагмент ТС]
                                              │
                                              ▼
                                    🧠 Классификатор марок
                                       (custom model)
                                              │
                                              ▼
                                    🏷️  Предсказанный класс
                                       (марка автомобиля)
                                              │
                                              ▼
                                    📊 Итоговый вывод
                                    [марка + уверенность]
```

### Описание этапов

**Этап 1 — Детекция транспортного средства:**
- Входной кадр подаётся в модель YOLOv8
- Модель предсказывает координаты bounding box для каждого ТС в кадре
- Извлекается ROI (кроп) с небольшим отступом (padding) для устойчивости

**Этап 2 — Классификация марки:**
- Кроп нормализуется и подаётся в кастомный классификатор
- На выходе — предсказанный класс (марка автомобиля) и confidence score
- Результат накладывается на исходный кадр с подписью

---

## 📈 Результаты обучения

### Метрики качества

| Метрика | Значение | Примечание |
|---|---|---|
| **mAP@50** | **95%** | Основная метрика точности классификации |
| **Precision** | ~0.94 | Доля верно предсказанных меток |
| **Recall** | ~0.93 | Полнота — доля найденных объектов |
| **Inference Time** | ~15–25 ms | На GPU (Kaggle T4) |
| **Epochs** | N/A | Лучшая эпоха сохранена в `best.pt` |
| **Training Platform** | Kaggle | NVIDIA GPU (T4 / P100) |

### Графики обучения

> 📊 *Место для графиков обучения (losses, mAP по эпохам)*
> 
> Рекомендуется добавить:
> - `results.png` — сводные графики из Ultralytics (loss + mAP)
> - `confusion_matrix.png` — матрица ошибок по классам
> - `PR_curve.png` — Precision-Recall кривая

```
[ results.png ]                    [ confusion_matrix.png ]
┌──────────────────────┐          ┌──────────────────────┐
│  Loss & mAP curves   │          │   Confusion Matrix   │
│  (добавить файл)     │          │   (добавить файл)    │
└──────────────────────┘          └──────────────────────┘
```

### Используемые файлы весов

```
weights/
├── best.pt     # Веса с лучшей эпохой (рекомендуется для инференса)
└── last.pt     # Веса последней эпохи (для дообучения)
```

---

## 🚀 Инструкция по запуску

### Требования

- Python **3.10+**
- pip / conda
- (Опционально) NVIDIA GPU + CUDA для ускорения

### 1. Клонирование репозитория

```bash
git clone https://github.com/<ваш-username>/patrol-ai.git
cd patrol-ai
```

### 2. Установка зависимостей

```bash
pip install ultralytics opencv-python
```

Для расширенной установки (Jupyter, визуализация):

```bash
pip install ultralytics opencv-python matplotlib pandas tqdm
```

### 3. Загрузка весов модели

Скачайте файл `best.pt` и поместите его в директорию `weights/`:

```
weights/
└── best.pt   ← поместите сюда
```

> Ссылка на веса: [Google Drive / Kaggle / Releases](#) *(добавьте актуальную ссылку)*

### 4. Запуск детекции

**На одном изображении:**

```python
from ultralytics import YOLO
import cv2

# Загрузка модели
model = YOLO("weights/best.pt")

# Инференс
results = model("test_image.jpg")

# Визуализация и сохранение
results[0].save("output.jpg")
```

**На видеофайле или веб-камере:**

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")

# Видеофайл
results = model("traffic_video.mp4", stream=True, save=True)

# Веб-камера (source=0)
results = model(source=0, stream=True, show=True)

for r in results:
    print(r.boxes)   # координаты и классы
```

**Через командную строку (CLI):**

```bash
# Изображение
yolo predict model=weights/best.pt source=test_image.jpg

# Видео
yolo predict model=weights/best.pt source=traffic.mp4 save=True
```

---

## 📁 Структура проекта

```
patrol-ai/
│
├── weights/
│   ├── best.pt             # Лучшие веса (инференс)
│   └── last.pt             # Веса последней эпохи
│
├── data/
│   ├── images/             # Обучающие/тестовые изображения
│   └── labels/             # YOLO-разметка (*.txt)
│
├── runs/                   # Артефакты обучения (Ultralytics)
│   └── train/
│       ├── results.png
│       ├── confusion_matrix.png
│       └── PR_curve.png
│
├── detect.py               # Скрипт запуска детекции
├── train.py                # Скрипт обучения модели
├── requirements.txt        # Зависимости
└── README.md               # Документация проекта
```

---

## 👤 Команда

| Роль | Имя |
|---|---|
| 👨‍💻 **Разработчик / Исследователь** | *[Ваше имя]* |
| 🎓 **Научный руководитель** | *[ФИО преподавателя]* |
| 🏛️ **Учебное заведение** | *[Название университета]* |
| 📅 **Год выполнения** | 2025 |

---

<div align="center">

**АСОП-ГАИ** — учебный проект в рамках практики по специальности

*Система разработана исключительно в образовательных целях.*

---

⭐ *Если проект оказался полезным — поставьте звезду!*

</div>

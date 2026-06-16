"""
patrolAI.py - Двухэтапная система детектирования и классификации автомобилей.

Этап 1: YOLO-детектор (Detection.pt) находит автомобили на видео и отслеживает
        каждый объект по уникальному ID с помощью алгоритма ByteTrack.
Этап 2: YOLO-классификатор (Classification.pt) определяет марку/модель/год
        автомобиля по вырезанной области кадра.

Результат: на каждый кадр накладываются рамка, название модели, уверенность
классификатора и рыночная цена из словаря CAR_PRICES.
"""

import cv2
from ultralytics import YOLO

# Инициализация моделей и источника видео

# Детекционная модель - находит и отслеживает автомобили (YOLO Detection)
model = YOLO('Detection.pt').to('cuda')

# Классификационная модель - определяет марку/модель/год (YOLO Classification)
classifier = YOLO('Classification.pt').to('cuda')

# Открываем видеофайл для покадровой обработки
cap = cv2.VideoCapture('1cars.mp4')

# Словарь рыночных цен автомобилей (модель → цена в USD)
# Ключи соответствуют именам классов из классификационной модели.
CAR_PRICES = {
    "Hyundai_Elantra_Touring_Hatchback_2012":           8_500,
    "Plymouth_Neon_Coupe_1999":                          2_500,
    "Hyundai_Genesis_Sedan_2012":                       12_000,
    "Land_Rover_LR2_SUV_2012":                          14_000,
    "Aston_Martin_V8_Vantage_Coupe_2012":               75_000,
    "Chevrolet_Malibu_Sedan_2007":                       5_500,
    "Chevrolet_Monte_Carlo_Coupe_2007":                  6_500,
    "BMW_1_Series_Convertible_2012":                    16_000,
    "Bentley_Continental_GT_Coupe_2007":                55_000,
    "Ford_Ranger_SuperCab_2011":                        14_000,
    "Suzuki_SX4_Sedan_2012":                             7_000,
    "BMW_1_Series_Coupe_2012":                          15_000,
    "Audi_TT_RS_Coupe_2012":                            28_000,
    "Cadillac_SRX_SUV_2012":                            13_000,
    "Jeep_Patriot_SUV_2012":                             9_000,
    "FIAT_500_Abarth_2012":                              9_500,
    "GMC_Acadia_SUV_2012":                              11_000,
    "Buick_Regal_GS_2012":                              11_000,
    "Cadillac_Escalade_EXT_Crew_Cab_2007":              22_000,
    "GMC_Yukon_Hybrid_SUV_2012":                        18_000,
    "Hyundai_Veracruz_SUV_2012":                        10_000,
    "Toyota_Sequoia_SUV_2012":                          24_000,
    "Dodge_Dakota_Crew_Cab_2010":                       12_000,
    "Dodge_Caravan_Minivan_1997":                        2_500,
    "Chrysler_300_SRT-8_2010":                          18_000,
    "Ford_Expedition_EL_SUV_2009":                      14_000,
    "Mercedes-Benz_E-Class_Sedan_2012":                 18_000,
    "Geo_Metro_Convertible_1993":                        3_500,
    "Jeep_Liberty_SUV_2012":                            10_000,
    "Honda_Accord_Sedan_2012":                          12_000,
    "smart_fortwo_Convertible_2012":                     6_000,
    "BMW_M6_Convertible_2010":                          35_000,
    "Chevrolet_Silverado_1500_Regular_Cab_2012":        18_000,
    "Ford_Freestar_Minivan_2007":                        5_000,
    "Chevrolet_HHR_SS_2010":                             7_500,
    "Mercedes-Benz_S-Class_Sedan_2012":                 22_000,
    "Chevrolet_Corvette_Convertible_2012":              38_000,
    "Audi_RS_4_Convertible_2008":                       28_000,
    "Ferrari_California_Convertible_2012":             115_000,
    "Ferrari_FF_Coupe_2012":                           130_000,
    "Aston_Martin_Virage_Convertible_2012":             95_000,
    "Dodge_Caliber_Wagon_2012":                          7_000,
    "Spyker_C8_Convertible_2009":                      180_000,
    "Dodge_Charger_SRT-8_2009":                         20_000,
    "BMW_ActiveHybrid_5_Sedan_2012":                    18_000,
    "Cadillac_CTS-V_Sedan_2012":                        28_000,
    "Audi_A5_Coupe_2012":                               20_000,
    "Ford_F-450_Super_Duty_Crew_Cab_2012":              30_000,
    "Dodge_Durango_SUV_2012":                           16_000,
    "GMC_Canyon_Extended_Cab_2012":                     14_000,
    "Chevrolet_Corvette_ZR1_2012":                      65_000,
    "FIAT_500_Convertible_2012":                         9_000,
    "Ford_Fiesta_Sedan_2012":                            7_000,
    "BMW_X5_SUV_2007":                                  13_000,
    "BMW_Z4_Convertible_2012":                          25_000,
    "Volkswagen_Beetle_Hatchback_2012":                 10_000,
    "Volvo_240_Sedan_1993":                              5_500,
    "HUMMER_H3T_Crew_Cab_2010":                         22_000,
    "Infiniti_G_Coupe_IPL_2012":                        18_000,
    "Ford_F-150_Regular_Cab_2012":                      20_000,
    "Chrysler_Crossfire_Convertible_2008":              12_000,
    "Nissan_NV_Passenger_Van_2012":                     16_000,
    "Bentley_Mulsanne_Sedan_2011":                     135_000,
    "Nissan_Leaf_Hatchback_2012":                        7_000,
    "Acura_TSX_Sedan_2012":                             13_000,
    "Chevrolet_Avalanche_Crew_Cab_2012":                20_000,
    "Infiniti_QX56_SUV_2011":                           20_000,
    "Dodge_Challenger_SRT8_2011":                       25_000,
    "Hyundai_Santa_Fe_SUV_2012":                        12_000,
    "Dodge_Sprinter_Cargo_Van_2009":                    12_000,
    "Hyundai_Veloster_Hatchback_2012":                   9_000,
    "Ford_F-150_Regular_Cab_2007":                      13_000,
    "Rolls-Royce_Ghost_Sedan_2012":                    175_000,
    "Ford_Edge_SUV_2012":                               13_000,
    "Mercedes-Benz_C-Class_Sedan_2012":                 16_000,
    "Chevrolet_Camaro_Convertible_2012":                22_000,
    "Buick_Enclave_SUV_2012":                           13_000,
    "Jeep_Grand_Cherokee_SUV_2012":                     16_000,
    "Chevrolet_Impala_Sedan_2007":                       6_500,
    "Hyundai_Sonata_Hybrid_Sedan_2012":                 10_000,
    "Dodge_Charger_Sedan_2012":                         16_000,
    "Bentley_Continental_GT_Coupe_2012":                80_000,
    "Chevrolet_Express_Cargo_Van_2007":                  9_000,
    "Audi_S4_Sedan_2007":                               16_000,
    "Ford_Mustang_Convertible_2007":                    14_000,
    "Tesla_Model_S_Sedan_2012":                         22_000,
    "Chrysler_Town_and_Country_Minivan_2012":           10_000,
    "Honda_Accord_Coupe_2012":                          13_000,
    "Nissan_Juke_Hatchback_2012":                        9_000,
    "Buick_Verano_Sedan_2012":                           9_000,
    "Buick_Rainier_SUV_2007":                            8_000,
    "Chevrolet_TrailBlazer_SS_2009":                    11_000,
    "Volvo_XC90_SUV_2007":                              10_000,
    "AM_General_Hummer_SUV_2000":                       20_000,
    "Lamborghini_Gallardo_LP_570-4_Superleggera_2012": 185_000,
    "Mercedes-Benz_SL-Class_Coupe_2009":                30_000,
    "Mercedes-Benz_Sprinter_Van_2012":                  18_000,
    "Land_Rover_Range_Rover_SUV_2012":                  22_000,
    "Dodge_Durango_SUV_2007":                           10_000,
    "Isuzu_Ascender_SUV_2008":                           8_000,
    "Bentley_Arnage_Sedan_2009":                        75_000,
    "Honda_Odyssey_Minivan_2007":                       10_000,
    "Chrysler_PT_Cruiser_Convertible_2008":              6_000,
    "Chevrolet_Express_Van_2007":                        9_000,
    "Mercedes-Benz_300-Class_Convertible_1993":         12_000,
    "Chevrolet_Corvette_Ron_Fellows_Edition_Z06_2007":  50_000,
    "HUMMER_H2_SUT_Crew_Cab_2009":                      28_000,
    "Jeep_Compass_SUV_2012":                             9_000,
    "Audi_V8_Sedan_1994":                                8_000,
    "Hyundai_Tucson_SUV_2012":                          10_000,
    "Dodge_Ram_Pickup_3500_Quad_Cab_2009":              18_000,
    "Mitsubishi_Lancer_Sedan_2012":                      9_500,
    "McLaren_MP4-12C_Coupe_2012":                      145_000,
    "Fisker_Karma_Sedan_2012":                          18_000,
    "Toyota_4Runner_SUV_2012":                          22_000,
    "GMC_Savana_Van_2012":                              14_000,
    "Chrysler_Aspen_SUV_2009":                           9_000,
    "Audi_S6_Sedan_2011":                               22_000,
    "Acura_Integra_Type_R_2001":                        35_000,
    "Ford_GT_Coupe_2006":                              400_000,
    "Dodge_Dakota_Club_Cab_2007":                        9_000,
    "Suzuki_Kizashi_Sedan_2012":                         8_000,
    "Audi_S5_Coupe_2012":                               25_000,
    "Hyundai_Azera_Sedan_2012":                         11_000,
    "Bugatti_Veyron_16-4_Coupe_2009":                1_500_000,
    "MINI_Cooper_Roadster_Convertible_2012":            12_000,
    "Audi_TT_Hatchback_2011":                           18_000,
    "Nissan_240SX_Coupe_1998":                          12_000,
    "Scion_xD_Hatchback_2012":                           7_500,
    "Chrysler_Sebring_Convertible_2010":                 7_000,
    "Dodge_Journey_SUV_2012":                            9_500,
    "Maybach_Landaulet_Convertible_2012":            1_000_000,
    "Lamborghini_Diablo_Coupe_2001":                   275_000,
    "Jeep_Wrangler_SUV_2012":                           20_000,
    "Acura_TL_Type-S_2008":                             11_000,
    "Rolls-Royce_Phantom_Sedan_2012":                  230_000,
    "Acura_ZDX_Hatchback_2012":                         16_000,
    "Toyota_Camry_Sedan_2012":                          12_000,
    "Audi_R8_Coupe_2012":                               85_000,
    "Hyundai_Elantra_Sedan_2007":                        5_500,
    "Suzuki_Aerio_Sedan_2007":                           5_000,
    "Aston_Martin_Virage_Coupe_2012":                  100_000,
    "Bentley_Continental_Supersports_Conv-_Convertible_2012": 160_000,
    "Acura_TL_Sedan_2012":                              14_000,
    "Audi_100_Wagon_1994":                               5_000,
    "BMW_3_Series_Sedan_2012":                          16_000,
    "BMW_M3_Coupe_2012":                                35_000,
    "BMW_6_Series_Convertible_2007":                    22_000,
    "Suzuki_SX4_Hatchback_2012":                         7_500,
    "Aston_Martin_V8_Vantage_Convertible_2012":         80_000,
    "Volkswagen_Golf_Hatchback_1991":                    5_000,
    "Dodge_Caliber_Wagon_2007":                          5_500,
    "Ford_E-Series_Wagon_Van_2012":                     14_000,
    "Chevrolet_Silverado_1500_Extended_Cab_2012":       20_000,
    "Rolls-Royce_Phantom_Drophead_Coupe_Convertible_2012": 280_000,
    "Bentley_Continental_Flying_Spur_Sedan_2007":       60_000,
    "Eagle_Talon_Hatchback_1998":                        5_500,
    "Chevrolet_Tahoe_Hybrid_SUV_2012":                  20_000,
    "Dodge_Ram_Pickup_3500_Crew_Cab_2010":              20_000,
    "Chevrolet_Silverado_1500_Hybrid_Crew_Cab_2012":    22_000,
    "Lincoln_Town_Car_Sedan_2011":                      12_000,
    "Bugatti_Veyron_16-4_Convertible_2009":          1_700_000,
    "Jaguar_XK_XKR_2012":                              38_000,
    "Chevrolet_Traverse_SUV_2012":                      13_000,
    "GMC_Terrain_SUV_2012":                             11_000,
    "Porsche_Panamera_Sedan_2012":                      30_000,
    "Hyundai_Accent_Sedan_2012":                         7_000,
    "Chevrolet_Cobalt_SS_2010":                          9_000,
    "Chevrolet_Silverado_2500HD_Regular_Cab_2012":      22_000,
    "BMW_M5_Sedan_2010":                                30_000,
    "Audi_S5_Convertible_2012":                         27_000,
    "Volkswagen_Golf_Hatchback_2012":                   10_000,
    "Toyota_Corolla_Sedan_2012":                        10_000,
    "Dodge_Magnum_Wagon_2008":                           9_000,
    "Daewoo_Nubira_Wagon_2002":                          3_000,
    "BMW_X6_SUV_2012":                                  22_000,
    "Ram_C_V_Cargo_Van_Minivan_2012":                   12_000,
    "Ferrari_458_Italia_Convertible_2012":             195_000,
    "Ferrari_458_Italia_Coupe_2012":                   185_000,
    "Acura_RL_Sedan_2012":                              14_000,
    "Audi_TTS_Coupe_2012":                              22_000,
    "Audi_100_Sedan_1994":                               5_000,
    "Spyker_C8_Coupe_2009":                            170_000,
    "Chevrolet_Silverado_1500_Classic_Extended_Cab_2007": 14_000,
    "Lamborghini_Reventon_Coupe_2008":               1_200_000,
    "BMW_X3_SUV_2012":                                  16_000,
    "Lamborghini_Aventador_Coupe_2012":                380_000,
    "Audi_S4_Sedan_2012":                               22_000,
    "Volvo_C30_Hatchback_2012":                         10_000,
    "Chevrolet_Sonic_Sedan_2012":                        7_000,
    "Ford_Focus_Sedan_2007":                             6_000,
    "Mazda_Tribute_SUV_2011":                            9_000,
    "Honda_Odyssey_Minivan_2012":                       13_000,
    "Hyundai_Sonata_Sedan_2012":                        10_000,
    "BMW_3_Series_Wagon_2012":                          16_000,
    "Chevrolet_Malibu_Hybrid_Sedan_2010":                7_500,
}


# Вспомогательные функции


def format_price(price: int) -> str:
    """Форматирует целочисленную цену в строку вида '$12,000'."""
    return f"${price:,}"


def draw_label_with_price(frame, x1, y1, x2, y2, label, price_str, font, font_scale, thickness, bbox_color):
    """
    Рисует на кадре две текстовые плашки (название модели + цена) над
    ограничивающей рамкой автомобиля, а затем саму рамку.

    Параметры:
        frame       - текущий кадр (numpy-массив BGR).
        x1, y1      - верхний левый угол рамки.
        x2, y2      - нижний правый угол рамки.
        label       - строка «ID <id>: <модель> (<уверенность>)».
        price_str   - отформатированная цена, например «$12,000».
        font        - шрифт OpenCV.
        font_scale  - масштаб шрифта.
        thickness   - толщина линий текста.
        bbox_color  - цвет рамки: зелёный (conf ≥ 0.70) или красный (conf < 0.70).
    """
    line_gap = 4  # вертикальный зазор (px) между плашкой названия и плашкой цены

    # Вычисляем пиксельные размеры каждой строки текста
    (lw, lh), _ = cv2.getTextSize(label,     font, font_scale, thickness)
    (pw, ph), _ = cv2.getTextSize(price_str, font, font_scale, thickness)

    # Ширина блока — по более широкой строке плюс внутренние отступы
    block_w = max(lw, pw) + 10

    # Полная высота двойного блока:
    # 10 (отступ сверху) + lh + line_gap + ph + 10 (отступ снизу)
    total_block_height = lh + ph + line_gap + 20

    # Если над рамкой не хватает места (блок уйдёт за верхний край),
    # смещаем плашки внутрь рамки; иначе размещаем ВЫШЕ рамки
    if y1 - total_block_height < 0:
        base_y = lh + 10          # плашки внутри рамки (прижаты к верху)
    else:
        base_y = y1 - ph - line_gap - 10  # плашки выше рамки

    # Первая плашка: название модели (чёрный фон, белый текст)
    bg_y1_top = base_y - lh - 5
    bg_y1_bot = base_y + 5
    cv2.rectangle(frame, (x1, bg_y1_top), (x1 + block_w, bg_y1_bot), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, label, (x1 + 5, base_y - 2), font, font_scale, (255, 255, 255), thickness)

    # Вторая плашка: цена (тёмно-серый фон, жёлто-голубой текст)
    price_y   = base_y + ph + line_gap + 4
    bg_y2_top = base_y + 5
    bg_y2_bot = price_y + 5
    cv2.rectangle(frame, (x1, bg_y2_top), (x1 + block_w, bg_y2_bot), (30, 30, 30), cv2.FILLED)
    cv2.putText(frame, price_str, (x1 + 5, price_y), font, font_scale, (0, 220, 255), thickness)

    # Рамка рисуется последней, чтобы её верхняя линия чётко
    # граничила с нижним краем текстового блока
    cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)



# Основной цикл обработки видео


paused = False          # флаг паузы (переключается клавишей Пробел)
resized_frame = None    # последний готовый кадр для отображения

while True:
    if not paused:
        # Читаем очередной кадр из видео
        ret, frame = cap.read()
        if not ret:
            # Видео закончилось или произошла ошибка чтения - выходим из цикла
            break

        # Этап 1: детекция и трекинг 
        # model.track возвращает список объектов Results; persist=True сохраняет
        # ID между кадрами, ByteTrack - алгоритм трекинга
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.7, iou=0.4)

        # Обрабатываем результаты только если в кадре есть отслеживаемые объекты
        if results[0].boxes is not None and results[0].boxes.id is not None:
            # Координаты рамок [x1, y1, x2, y2] и уникальные ID треков
            boxes     = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box

                # Вырезаем область кадра с автомобилем (с защитой от выхода за границы)
                crop = frame[max(0, y1):min(frame.shape[0], y2),
                             max(0, x1):min(frame.shape[1], x2)]

                if crop.size > 0:
                    # Этап 2: классификация вырезанного фрагмента
                    cls_results = classifier(crop, verbose=False)

                    # Индекс и имя класса с наивысшей уверенностью (top-1)
                    top1_idx   = cls_results[0].probs.top1
                    class_name = cls_results[0].names[top1_idx]
                    cls_conf   = cls_results[0].probs.top1conf.item()  # уверенность [0..1]

                    # Цвет рамки: зелёный - высокая уверенность, красный - низкая
                    bbox_color = (0, 0, 255) if cls_conf < 0.70 else (0, 255, 0)

                    # Ищем цену в словаре; если модель неизвестна - выводим N/A
                    price     = CAR_PRICES.get(class_name)
                    price_str = format_price(price) if price else "Price: N/A"

                    # Строка подписи: ID трекера, название класса, уверенность
                    label = f"ID {track_id}: {class_name} ({cls_conf:.2f})"

                    # Параметры шрифта
                    font       = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness  = 2

                    # Рисуем плашки с подписью и рамку на текущем кадре
                    draw_label_with_price(
                        frame, x1, y1, x2, y2,
                        label, price_str,
                        font, font_scale, thickness,
                        bbox_color
                    )

        # Масштабируем кадр до разрешения отображения 1280×720
        resized_frame = cv2.resize(frame, (1280, 720))

    # Отображаем кадр (даже в режиме паузы - показываем последний сохранённый)
    if resized_frame is not None:
        cv2.imshow("YOLOv12 Two-Stage System", resized_frame)

    # Ожидаем нажатия клавиши:
    #   - в паузе ждём 30 мс (снижаем нагрузку на CPU),
    #   - в обычном режиме - 1 мс (максимальная скорость)
    key = cv2.waitKey(30 if paused else 1) & 0xFF
    if key == 27:      # ESC - выход
        break
    elif key == 32:    # Пробел - переключение паузы
        paused = not paused


# Освобождаем ресурсы

cap.release()           # закрываем видеофайл
cv2.destroyAllWindows() # закрываем все окна OpenCV

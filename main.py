#%%
import cv2
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

class AccurateGPUAggressiveDrivingDetector:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {device}")

        self.track_history = defaultdict(lambda: {
            'positions': torch.zeros((120, 2), device=self.device),
            'timestamps': torch.zeros(120, device=self.device),
            'count': 0,
            'aggressive_count': 0,
            'last_aggressive_frame': 0,
            'world_speed': torch.tensor(0.0, device=self.device),
            'world_acceleration': torch.tensor(0.0, device=self.device),
            'movement_angle': torch.tensor(0.0, device=self.device),
            'speed_history': torch.zeros(30, device=self.device),
            'speed_history_count': 0,
            'is_calibrated': False,
            'pixel_positions': torch.zeros((120, 2), device=self.device)  # Сохраняем пиксельные координаты
        })

        # Консервативные пороги
        self.ACCELERATION_THRESHOLD = torch.tensor(8.0, device=self.device)
        self.DECELERATION_THRESHOLD = torch.tensor(-8.0, device=self.device)
        self.LANE_CHANGE_ANGLE_THRESHOLD = torch.tensor(10.0, device=self.device)
        self.MIN_TRACK_LENGTH = 10
        self.MIN_CALIBRATION_FRAMES = 3

        # Параметры калибровки перспективы
        self.perspective_matrix = None
        self.inverse_perspective_matrix = None
        self.is_perspective_calibrated = False
        self.calibration_points = []
        self.calibration_complete = False
        self.real_world_size = None  # Реальные размеры в метрах

        self.calibration_data = {
            'frame_size': None,
            'scale_factor': 1.0  # Коэффициент масштабирования
        }

    def mouse_callback(self, event, x, y, flags, param):
        """Callback для мыши для сбора точек калибровки"""
        if event == cv2.EVENT_LBUTTONDOWN and not self.calibration_complete:
            self.calibration_points.append((x, y))
            print(f"Point {len(self.calibration_points)}: ({x}, {y})")

            if len(self.calibration_points) == 4:
                self.calibration_complete = True
                print("Calibration complete! Computing perspective matrix...")

    def interactive_calibrate_perspective(self, frame):
        """Интерактивная калибровка перспективы с реальными размерами"""
        height, width = frame.shape[:2]
        self.calibration_data['frame_size'] = (width, height)

        # Создаем окно и устанавливаем callback для мыши
        cv2.namedWindow("Perspective Calibration")
        cv2.setMouseCallback("Perspective Calibration", self.mouse_callback)

        calibration_frame = frame.copy()

        instructions = [
            "Click 4 points to define a rectangle on the road:",
            "1. Bottom-left point of road section",
            "2. Bottom-right point of road section",
            "3. Top-right point of road section",
            "4. Top-left point of road section",
            "Press 'r' to reset, 'c' to cancel"
        ]

        while not self.calibration_complete:
            # Отображаем инструкции
            for i, text in enumerate(instructions):
                cv2.putText(calibration_frame, text, (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Отображаем выбранные точки
            for i, point in enumerate(self.calibration_points):
                cv2.circle(calibration_frame, point, 8, (0, 255, 0), -1)
                cv2.putText(calibration_frame, str(i + 1), (point[0] + 10, point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Рисуем линии между точками
            if len(self.calibration_points) >= 2:
                for i in range(len(self.calibration_points) - 1):
                    cv2.line(calibration_frame, self.calibration_points[i],
                             self.calibration_points[i + 1], (0, 255, 0), 2)
                if len(self.calibration_points) == 4:
                    cv2.line(calibration_frame, self.calibration_points[3],
                             self.calibration_points[0], (0, 255, 0), 2)

            cv2.imshow("Perspective Calibration", calibration_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):  # Reset
                self.calibration_points = []
                calibration_frame = frame.copy()
                print("Calibration reset")
            elif key == ord('c'):  # Cancel
                print("Calibration cancelled")
                cv2.destroyWindow("Perspective Calibration")
                return False

        # Запрашиваем реальные размеры области
        print("\n=== Real World Dimensions ===")
        print("Please enter the real-world dimensions of the calibrated area.")
        print("For example, if you calibrated a 50m section of road, enter 50.")

        try:
            real_width = float(input("Enter the WIDTH of the calibrated area in meters: "))
            real_height = float(input("Enter the LENGTH of the calibrated area in meters: "))
            self.real_world_size = (real_width, real_height)
            print(f"Real world size set to: {real_width}m x {real_height}m")
        except:
            # Значения по умолчанию, если пользователь не ввел
            self.real_world_size = (50.0, 150.0)
            print(f"Using default size: {self.real_world_size[0]}m x {self.real_world_size[1]}m")

        # Вычисляем матрицу перспективного преобразования
        if len(self.calibration_points) == 4:
            src_points = np.float32(self.calibration_points)

            # Определяем целевую область (прямоугольник)
            width, height = self.calibration_data['frame_size']
            dst_points = np.float32([
                [0, height],  # Нижний левый
                [width, height],  # Нижний правый
                [width, 0],  # Верхний правый
                [0, 0]  # Верхний левый
            ])

            self.perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            self.inverse_perspective_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
            self.is_perspective_calibrated = True

            # Вычисляем коэффициент масштабирования на основе реальных размеров
            # Преобразуем точки и вычисляем масштаб
            transformed_points = cv2.perspectiveTransform(
                src_points.reshape(1, -1, 2), self.perspective_matrix
            ).reshape(-1, 2)

            # Вычисляем размеры в преобразованных координатах
            transformed_width = np.abs(transformed_points[1, 0] - transformed_points[0, 0])
            transformed_height = np.abs(transformed_points[2, 1] - transformed_points[1, 1])

            # Коэффициенты масштабирования (метров на пиксель)
            self.meters_per_pixel_x = self.real_world_size[0] / transformed_width
            self.meters_per_pixel_y = self.real_world_size[1] / transformed_height

            print(f"Meters per pixel - X: {self.meters_per_pixel_x:.6f}, Y: {self.meters_per_pixel_y:.6f}")

            # Визуализируем результат
            result_frame = frame.copy()

            # Рисуем исходные точки
            for point in self.calibration_points:
                cv2.circle(result_frame, point, 8, (0, 255, 0), -1)

            # Рисуем преобразованную область
            back_points = cv2.perspectiveTransform(
                dst_points.reshape(1, -1, 2), self.inverse_perspective_matrix
            ).reshape(-1, 2)

            # Рисуем преобразованную область
            for i in range(4):
                cv2.line(result_frame,
                         tuple(back_points[i].astype(int)),
                         tuple(back_points[(i + 1) % 4].astype(int)),
                         (255, 0, 0), 2)

            # Отображаем реальные размеры
            cv2.putText(result_frame, f"Real size: {self.real_world_size[0]}m x {self.real_world_size[1]}m",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result_frame, "Calibration Complete! Press any key to continue",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Perspective Calibration", result_frame)
            cv2.waitKey(2000)  # Показать 2 секунды

        cv2.destroyWindow("Perspective Calibration")
        return True

    def pixel_to_world(self, pixel_coords):
        """Преобразование пиксельных координат в мировые (в метрах) с правильным масштабированием"""
        if not self.is_perspective_calibrated:
            return torch.tensor([pixel_coords[0], pixel_coords[1]], device=self.device)

        # Преобразуем в однородные координаты
        pixel_points = np.array([[pixel_coords[0], pixel_coords[1]]], dtype=np.float32)

        # Применяем преобразование перспективы
        world_points_pixels = cv2.perspectiveTransform(pixel_points.reshape(1, -1, 2),
                                                       self.perspective_matrix)

        # Масштабируем в метры с использованием реальных размеров
        world_x = world_points_pixels[0, 0, 0] * self.meters_per_pixel_x
        world_y = world_points_pixels[0, 0, 1] * self.meters_per_pixel_y

        return torch.tensor([world_x, world_y], device=self.device)

    def update_track_history(self, track_id, pixel_position, current_time):
        """Обновление истории в мировых координатах"""
        track_data = self.track_history[track_id]

        # Сохраняем пиксельные координаты для отладки
        if track_data['count'] < 120:
            track_data['pixel_positions'][track_data['count']] = pixel_position
            track_data['timestamps'][track_data['count']] = current_time

            # Преобразуем в мировые координаты
            if self.is_perspective_calibrated:
                world_position = self.pixel_to_world(pixel_position.cpu().numpy())
                track_data['positions'][track_data['count']] = world_position

            track_data['count'] += 1
        else:
            track_data['pixel_positions'] = torch.roll(track_data['pixel_positions'], -1, 0)
            track_data['timestamps'] = torch.roll(track_data['timestamps'], -1, 0)
            track_data['pixel_positions'][-1] = pixel_position
            track_data['timestamps'][-1] = current_time

            if self.is_perspective_calibrated:
                world_position = self.pixel_to_world(pixel_position.cpu().numpy())
                track_data['positions'] = torch.roll(track_data['positions'], -1, 0)
                track_data['positions'][-1] = world_position

        # Калибруем после достаточного количества кадров
        if (track_data['count'] >= self.MIN_CALIBRATION_FRAMES and
                not track_data['is_calibrated'] and
                self.is_perspective_calibrated):
            self._calibrate_track_speed(track_id)

    def _calibrate_track_speed(self, track_id):
        """Калибровка скорости для конкретного трека"""
        track_data = self.track_history[track_id]

        if track_data['count'] < 10:
            return

        valid_positions = track_data['positions'][:track_data['count']]
        valid_timestamps = track_data['timestamps'][:track_data['count']]

        # Вычисляем среднюю скорость за период калибровки
        speeds = []
        for i in range(1, min(20, track_data['count'])):
            idx1 = track_data['count'] - i
            idx2 = track_data['count'] - i - 1

            if idx2 < 0:
                break

            pos1 = valid_positions[idx1]
            pos2 = valid_positions[idx2]
            time1 = valid_timestamps[idx1]
            time2 = valid_timestamps[idx2]

            if time1 > time2:
                displacement = torch.norm(pos1 - pos2)
                time_delta = time1 - time2

                if time_delta > 0.001:
                    speed = displacement / time_delta
                    speeds.append(speed)

        if speeds:
            # Используем медиану для устойчивости к выбросам
            speeds_tensor = torch.stack(speeds)
            median_speed = torch.median(speeds_tensor)
            track_data['world_speed'] = median_speed
            track_data['is_calibrated'] = True

            # Сохраняем в историю скоростей
            track_data['speed_history'][0] = median_speed
            track_data['speed_history_count'] = 1

            # Отладочная информация
            speed_kmh = median_speed.item() * 3.6
            print(f"Track {track_id} calibrated: {speed_kmh:.1f} km/h")

    def calculate_world_metrics(self, track_data):
        """Расчет метрик в мировых координатах"""
        if track_data['count'] < 10 or not track_data['is_calibrated']:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        valid_positions = track_data['positions'][:track_data['count']]
        valid_timestamps = track_data['timestamps'][:track_data['count']]

        # Используем окно из 10 последних точек для стабильности
        window_size = min(10, track_data['count'])
        recent_positions = valid_positions[-window_size:]
        recent_timestamps = valid_timestamps[-window_size:]

        # Вычисляем скорость методом наименьших квадратов
        if len(recent_positions) >= 5:
            # Линейная регрессия для каждой координаты
            t_relative = recent_timestamps - recent_timestamps[0]

            # Для X координаты
            x_coords = recent_positions[:, 0]
            A_x = torch.stack([t_relative, torch.ones_like(t_relative)], dim=1)
            try:
                coefficients_x = torch.linalg.lstsq(A_x, x_coords.unsqueeze(1)).solution
                speed_x = coefficients_x[0]
            except:
                speed_x = torch.tensor(0.0, device=self.device)

            # Для Y координаты
            y_coords = recent_positions[:, 1]
            A_y = torch.stack([t_relative, torch.ones_like(t_relative)], dim=1)
            try:
                coefficients_y = torch.linalg.lstsq(A_y, y_coords.unsqueeze(1)).solution
                speed_y = coefficients_y[0]
            except:
                speed_y = torch.tensor(0.0, device=self.device)

            # Общая скорость
            current_speed = torch.sqrt(speed_x ** 2 + speed_y ** 2)

            # Угол движения (для детекции перестроения)
            if torch.abs(current_speed) > 0.1:
                track_data['movement_angle'] = torch.atan2(speed_y, speed_x) * 180 / torch.pi

            # Расчет ускорения на основе истории скоростей
            if track_data['speed_history_count'] > 0:
                # Обновляем историю скоростей
                if track_data['speed_history_count'] < 30:
                    track_data['speed_history'][track_data['speed_history_count']] = current_speed
                    track_data['speed_history_count'] += 1
                else:
                    track_data['speed_history'] = torch.roll(track_data['speed_history'], -1, 0)
                    track_data['speed_history'][-1] = current_speed

                # Вычисляем ускорение методом наименьших квадратов
                if track_data['speed_history_count'] >= 10:
                    speed_history = track_data['speed_history'][:track_data['speed_history_count']]
                    t_accel = torch.arange(track_data['speed_history_count'],
                                           dtype=torch.float32, device=self.device)

                    A_accel = torch.stack([t_accel, torch.ones_like(t_accel)], dim=1)
                    try:
                        coefficients_accel = torch.linalg.lstsq(A_accel, speed_history.unsqueeze(1)).solution
                        acceleration = coefficients_accel[0]
                    except:
                        acceleration = torch.tensor(0.0, device=self.device)

                    # Сглаживаем ускорение
                    smooth_factor = 0.9
                    track_data['world_acceleration'] = (
                            smooth_factor * track_data['world_acceleration'] +
                            (1 - smooth_factor) * acceleration
                    )

                    return current_speed, track_data['world_acceleration']

            return current_speed, torch.tensor(0.0, device=self.device)

        return track_data['world_speed'], torch.tensor(0.0, device=self.device)

    def detect_lane_change_robust(self, track_data):
        """Надёжная детекция смены полосы на основе изменения угла движения"""
        if track_data['count'] < 30 or not track_data['is_calibrated']:
            return False

        valid_positions = track_data['positions'][:track_data['count']]

        # Анализируем изменение угла движения
        if track_data['count'] >= 20:
            # Берем три сегмента: начало, середину и конец
            segment_size = track_data['count'] // 3

            segment1 = valid_positions[:segment_size]
            segment2 = valid_positions[segment_size:2 * segment_size]
            segment3 = valid_positions[2 * segment_size:]

            if len(segment1) >= 5 and len(segment2) >= 5 and len(segment3) >= 5:
                # Вычисляем углы для каждого сегмента
                angle1 = self._calculate_movement_angle(segment1)
                angle2 = self._calculate_movement_angle(segment2)
                angle3 = self._calculate_movement_angle(segment3)

                # Проверяем значительное изменение угла
                angle_change_12 = torch.abs(angle1 - angle2)
                angle_change_23 = torch.abs(angle2 - angle3)

                # Смена полосы должна быть устойчивым изменением
                significant_change = (angle_change_12 > self.LANE_CHANGE_ANGLE_THRESHOLD and
                                      angle_change_23 > self.LANE_CHANGE_ANGLE_THRESHOLD and
                                      torch.sign(angle1 - angle2) == torch.sign(angle2 - angle3))

                return significant_change

        return False

    def detect_lane_change_angle_robust(self, track_data):
        """Надёжная детекция смены полосы на основе изменения угла движения"""
        if track_data['count'] < 30 or not track_data['is_calibrated']:
            return 0, 0, 0

        valid_positions = track_data['positions'][:track_data['count']]

        # Анализируем изменение угла движения
        if track_data['count'] >= 20:
            # Берем три сегмента: начало, середину и конец
            segment_size = track_data['count'] // 3

            segment1 = valid_positions[:segment_size]
            segment2 = valid_positions[segment_size:2 * segment_size]
            segment3 = valid_positions[2 * segment_size:]

            if len(segment1) >= 5 and len(segment2) >= 5 and len(segment3) >= 5:
                # Вычисляем углы для каждого сегмента
                angle1 = self._calculate_movement_angle(segment1)
                angle2 = self._calculate_movement_angle(segment2)
                angle3 = self._calculate_movement_angle(segment3)

                # Проверяем значительное изменение угла
                angle_change_12 = torch.abs(angle1 - angle2)
                angle_change_23 = torch.abs(angle2 - angle3)

                # Смена полосы должна быть устойчивым изменением
                significant_change = (angle_change_12 > self.LANE_CHANGE_ANGLE_THRESHOLD and
                                      angle_change_23 > self.LANE_CHANGE_ANGLE_THRESHOLD and
                                      torch.sign(angle1 - angle2) == torch.sign(angle2 - angle3))

                return angle1.item(), angle2.item(), angle3.item()

        return 0, 0, 0

    def _calculate_movement_angle(self, positions):
        """Вычисление угла движения для сегмента позиций"""
        if len(positions) < 2:
            return torch.tensor(0.0, device=self.device)

        # Линейная регрессия для определения направления
        t = torch.arange(len(positions), dtype=torch.float32, device=self.device)
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]

        # Для X
        A_x = torch.stack([t, torch.ones_like(t)], dim=1)
        try:
            coeff_x = torch.linalg.lstsq(A_x, x_coords.unsqueeze(1)).solution
            dx = coeff_x[0]
        except:
            dx = torch.tensor(0.0, device=self.device)

        # Для Y
        A_y = torch.stack([t, torch.ones_like(t)], dim=1)
        try:
            coeff_y = torch.linalg.lstsq(A_y, y_coords.unsqueeze(1)).solution
            dy = coeff_y[0]
        except:
            dy = torch.tensor(0.0, device=self.device)

        # Угол в градусах
        if torch.abs(dx) > 0.001:
            angle = torch.atan2(dy, dx) * 180 / torch.pi
        else:
            angle = torch.tensor(90.0 if dy > 0 else -90.0, device=self.device)

        return angle

    def detect_aggressive_behavior_robust(self, track_data, track_id, current_frame):
        """Надёжная детекция агрессивного поведения"""
        if track_data['count'] < self.MIN_TRACK_LENGTH or not track_data['is_calibrated']:
            return False, []

        speed, acceleration = self.calculate_world_metrics(track_data)
        behaviors = []

        # Переводим в км/ч для проверки
        speed_kmh = speed.item() * 3.6 if speed.numel() == 1 else 0.0
        acceleration_value = acceleration.item() if acceleration.numel() == 1 else 0.0

        # ФИЛЬТР НЕРЕАЛИСТИЧНЫХ СКОРОСТЕЙ
        if speed_kmh > 250.0 or speed_kmh < 1.0:
            return False, []

        # КОНСЕРВАТИВНЫЕ ПРОВЕРКИ

        # 1. Ускорение/торможение - требуем значительных изменений
        acceleration_significant = abs(acceleration_value) > 1.5

        if acceleration_significant:
            # Торможение - строгие условия
            if acceleration_value < self.DECELERATION_THRESHOLD.item():
                # Дополнительные проверки:
                # - Высокая исходная скорость
                # - Устойчивое торможение
                if speed_kmh > 50.0:  # > 50 км/ч
                    if track_data['aggressive_count'] == 0 or \
                            (current_frame - track_data['last_aggressive_frame']) > 60:
                        behaviors.append("REZKOE_TORMOZHENIE")

            # Ускорение - строгие условия
            elif acceleration_value > self.ACCELERATION_THRESHOLD.item():
                if speed_kmh > 30.0 and track_data['aggressive_count'] == 0:
                    behaviors.append("RESKOE_USKORENIE")

        # 2. Смена полосы - надежная детекция
        if self.detect_lane_change_robust(track_data):
            # Только если не было недавних агрессивных событий
            if current_frame - track_data['last_aggressive_frame'] > 90:
                behaviors.append("REZKOE_PERESTROENIE")

        is_aggressive = len(behaviors) > 0

        # Консервативное обновление счетчиков
        if is_aggressive and (current_frame - track_data['last_aggressive_frame'] > 45):
            track_data['aggressive_count'] += 1
            track_data['last_aggressive_frame'] = current_frame


        return is_aggressive, behaviors


def main():
    detector = AccurateGPUAggressiveDrivingDetector()
    model = YOLO("models/yolo11x.pt")
    model.model.to(detector.device)

    cap = cv2.VideoCapture("data/first_60_seconds.mp4")
    frame_count = 0

    # Статистика
    stats = {
        'total_detections': 0,
        'calibrated_tracks': 0,
        'aggressive_detections': 0,
        'realistic_speeds': 0
    }

    # Сначала калибруем перспективу на первом кадре
    ret, first_frame = cap.read()
    if ret:
        print("Starting interactive perspective calibration...")
        success = detector.interactive_calibrate_perspective(first_frame)
        if not success:
            print("Calibration was cancelled. Exiting.")
            return

        # Возвращаемся к началу
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        current_time = frame_count / 30.0

        results = model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            classes=[2, 3, 5, 7], conf=0.25, verbose=False, device=detector.device
        )

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confidences = results[0].boxes.conf.float().cpu().tolist()

            for box, track_id, conf in zip(boxes, track_ids, confidences):
                stats['total_detections'] += 1

                x, y, w, h = box
                center = torch.tensor([float(x), float(y)], device=detector.device)

                detector.update_track_history(track_id, center, torch.tensor(current_time, device=detector.device))

                track_data = detector.track_history[track_id]
                is_aggressive, behaviors = detector.detect_aggressive_behavior_robust(
                    track_data, track_id, frame_count
                )

                # Визуализация
                if track_data['is_calibrated']:
                    stats['calibrated_tracks'] += 1
                    speed, acceleration = detector.calculate_world_metrics(track_data)
                    angle1, angle2, angle3 = detector.detect_lane_change_angle_robust(track_data)

                    speed_kmh = speed.item() * 3.6 if speed.numel() == 1 else 0.0

                    # Проверяем реалистичность скорости
                    if 20.0 <= speed_kmh <= 300.0:
                        stats['realistic_speeds'] += 1
                        color = (0, 255, 0)  # Зеленый для реалистичных скоростей
                        speed_text = f"{speed_kmh:.1f}km/h"
                        angle_text = f"{angle1:.1f}, {angle1:.1f}, {angle1:.1f}deg"
                    else:
                        color = (128, 128, 128)  # Серый для нереалистичных
                        speed_text = f"{speed_kmh:.1f}km/h"  # Все равно показываем скорость

                    if is_aggressive:
                        color = (0, 0, 255)  # Красный для агрессивных
                        stats['aggressive_detections'] += 1
                        reason_text = behaviors[0] if behaviors else "AGGRESSIVE"
                        display_text = f"ID:{track_id} {reason_text}"
                    else:
                        display_text = f"ID:{track_id} {speed_text}"

                    # Рисуем bounding box
                    cv2.rectangle(frame,
                                  (int(x - w / 2), int(y - h / 2)),
                                  (int(x + w / 2), int(y + h / 2)),
                                  color, 2)

                    # Текст
                    cv2.putText(frame, display_text,
                                (int(x - w / 2), int(y - h / 2 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                else:
                    # Некалиброванные треки - серым
                    cv2.rectangle(frame,
                                  (int(x - w / 2), int(y - h / 2)),
                                  (int(x + w / 2), int(y + h / 2)),
                                  (0, 255, 0), 1)
                    # Показываем прогресс калибровки
                    progress = min(track_data['count'] / detector.MIN_CALIBRATION_FRAMES, 1.0)
                    cv2.putText(frame, f"ID:{track_id} CALIB: {progress:.0%}",
                                (int(x - w / 2), int(y - h / 2 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Статистика
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Calibrated: {stats['calibrated_tracks']}/{stats['total_detections']}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Aggressive: {stats['aggressive_detections']}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Realistic speeds: {stats['realistic_speeds']}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Показываем информацию о калибровке
        if detector.is_perspective_calibrated:
            cv2.putText(frame, f"Scale: {detector.meters_per_pixel_x:.4f} m/px",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Traffic Analysis - ACCURATE", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):  # Перекалибровка
            print("Recalibrating perspective...")
            success = detector.interactive_calibrate_perspective(frame)
            if not success:
                print("Calibration was cancelled.")

    cap.release()
    cv2.destroyAllWindows()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

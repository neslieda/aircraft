import time
import cv2
from ultralytics import YOLO
import numpy as np
from statistics import mean
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class TrackedObject:
    positions: List[List[int]]
    last_positions: List[List[List[int]]]
    control_values: List[float]
    stability_checks: List[float]
    detection_time: Optional[float] = None
    fps_frame_time: float = 0
    has_moved: bool = False
    is_detected: bool = False


class AirplaneBridgeTracker:
    def __init__(self, model_path: str, video_path: str, output_path: str):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps,
                                   (self.width, self.height))

        self.window = TrackedObject([], [], [], [])
        self.bridge = TrackedObject([], [], [], [])

        self.frame_count = 0
        self.window_count = 0
        self.bridge_count = 0

    def calculate_movement(self, current: List[List[int]],
                           previous: List[List[int]]) -> float:
        return np.sum(np.abs(np.mean(current, axis=0) -
                             np.mean(previous, axis=0)))

    def process_detection(self, tracked_obj: TrackedObject,
                          coords: List[int],
                          count: int,
                          stability_threshold: float,
                          min_stable_time: float = 5.0) -> bool:
        tracked_obj.positions.append(coords)

        if count % 10 == 0 and count != 0:
            tracked_obj.last_positions.append(tracked_obj.positions)

            if count > 10:
                movement = self.calculate_movement(tracked_obj.positions,
                                                   tracked_obj.last_positions[-2])

                if movement < stability_threshold:
                    tracked_obj.control_values.append(movement)

                    if len(tracked_obj.control_values) == 10:
                        avg_movement = mean(tracked_obj.control_values)

                        if avg_movement < stability_threshold:
                            if len(tracked_obj.stability_checks) == 5:
                                tracked_obj.stability_checks.pop(0)
                            tracked_obj.stability_checks.append(avg_movement)

                            if (len(tracked_obj.stability_checks) == 5 and
                                    min(tracked_obj.stability_checks) < 2):
                                return True

                        tracked_obj.control_values = []

            tracked_obj.positions = []

        return False

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float, float]:
        results = self.model(frame)
        self.frame_count += 1
        runway_time = self.frame_count / self.fps

        window_parked = False
        bridge_connected = False

        for result in results[0].boxes.data:
            x1, y1, x2, y2 = map(int, result[:4])
            conf = result[4].item()
            cls = int(result[5].item())

            if self.model.names[cls] == 'window':
                self.window_count += 1
                if self.process_detection(self.window, [x1, y1, x2, y2],
                                          self.window_count, 3.0):
                    window_parked = True
                    if not self.window.detection_time:
                        self.window.detection_time = time.time()
                        self.window.fps_frame_time = runway_time

            elif self.model.names[cls] == 'bridge' and conf > 0.7:
                self.bridge_count += 1
                if self.process_detection(self.bridge, [x1, y1, x2, y2],
                                          self.bridge_count, 6.0, 15.0):
                    bridge_connected = True
                    if not self.bridge.detection_time:
                        self.bridge.detection_time = time.time()
                        self.bridge.fps_frame_time = runway_time

        # Add timing information to frame
        cv2.putText(frame, f'Runway Time: {runway_time:.2f}s',
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        park_time = self._calculate_duration(self.window, 5.0)
        cv2.putText(frame, f'Parking Time: {park_time:.2f}s',
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        bridge_time = self._calculate_duration(self.bridge, 15.0)
        cv2.putText(frame, f'Bridge Connection Time: {bridge_time:.2f}s',
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame, park_time, bridge_time

    def _calculate_duration(self, obj: TrackedObject,
                            min_time: float) -> float:
        if obj.detection_time is None:
            return 0.0

        current_duration = time.time() - obj.detection_time
        if current_duration >= min_time:
            return self.frame_count / self.fps - obj.fps_frame_time
        return 0.0

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame, park_time, bridge_time = self.process_frame(frame)
            self.out.write(frame)
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


# Usage
if __name__ == "__main__":
    model_path = r'C:\Users\edayu\PycharmProjects\Yapayzeka\airplane\best (1).pt'
    video_path = r'C:\Users\edayu\PycharmProjects\Yapayzeka\airplane\vlc-record-2024-06-26-18h29m32s-VDGS 40 Park ve Hizmet.mp4-.mp4'
    output_path = r'C:\Users\edayu\PycharmProjects\Yapay zeka\airplane\output_video_kopru2.avi'

    tracker = AirplaneBridgeTracker(model_path, video_path, output_path)
    tracker.run()
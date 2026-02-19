import cv2
import sys
from .utils import *
from .sort import *
import numpy as np


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else './test.mp4'
    out_csv = sys.argv[2] if len(sys.argv) > 2 else './test.csv'
    from ultralytics import YOLO
    results = {}
    mot_tracker = Sort()
    coco_model = YOLO('./models/yolo26n.pt')
    license_plate_detector = YOLO('./models/license_plate_detector_model.pt')
    cap = cv2.VideoCapture(video_path)
    vehicles = [2, 3, 5, 7]
    frame_nmr = -1
    ret = True
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if ret:
            results[frame_nmr] = {}
            detections = coco_model(frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])
            track_ids = mot_tracker.update(np.asarray(detections_))
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                xcar1, ycar1, xcar2, ycar2, car_id = get_vehicle(license_plate, track_ids)
                if car_id != -1:
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                    if license_plate_text is not None:
                        results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                      'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': license_plate_text,
                                                                        'bbox_score': score,
                                                                        'text_score': license_plate_text_score}}
    # Always write the output file, even if empty
    write_csv(results, out_csv)

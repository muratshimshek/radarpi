import unittest

from vehicle_validation_worker import VehicleDetection, _choose_ocr_result, _class_aware_nms, _select_vehicle, _status_filename
from plate_reader import _iter_rapidocr_text_candidates, _normalize_plate_candidate


class VehicleValidationTests(unittest.TestCase):
    def test_class_aware_nms_removes_overlapping_same_class(self):
        detections = [
            VehicleDetection(2, "car", 0.90, (10, 10, 110, 110)),
            VehicleDetection(2, "car", 0.80, (12, 12, 108, 108)),
            VehicleDetection(7, "truck", 0.70, (12, 12, 108, 108)),
        ]
        result = _class_aware_nms(detections)
        self.assertEqual([(item.class_id, item.confidence) for item in result], [(2, 0.90), (7, 0.70)])

    def test_selector_prefers_confident_large_lower_vehicle(self):
        detections = [
            VehicleDetection(2, "car", 0.80, (10, 10, 80, 80)),
            VehicleDetection(5, "bus", 0.78, (20, 100, 500, 350)),
        ]
        selected = _select_vehicle(detections, (360, 640, 3))
        self.assertIsNotNone(selected)
        self.assertEqual(selected.class_name, "bus")

    def test_selector_handles_empty_input(self):
        self.assertIsNone(_select_vehicle([], (360, 640, 3)))

    def test_ocr_requires_consensus_for_normal_confidence(self):
        plate, confidence, decision = _choose_ocr_result([
            {"plate": "49IZI0937", "confidence": 0.9368},
            {"plate": "16OIS625", "confidence": 0.9368},
            {"plate": "13OO6452", "confidence": 0.9368},
        ])
        self.assertIsNone(plate)
        self.assertIsNone(confidence)
        self.assertEqual(decision, "conflicting_or_weak_candidates")

    def test_ocr_accepts_two_crop_consensus(self):
        plate, confidence, decision = _choose_ocr_result([
            {"plate": "34ABC123", "confidence": 0.88},
            {"plate": "34ABC123", "confidence": 0.91},
            {"plate": None, "confidence": 0.0},
        ])
        self.assertEqual(plate, "34ABC123")
        self.assertEqual(confidence, 0.91)
        self.assertEqual(decision, "multi_crop_consensus")

    def test_rapidocr_parser_reads_text_not_box_coordinates(self):
        payload = ([
            [[[10.0, 20.0], [110.0, 20.0], [110.0, 50.0], [10.0, 50.0]], "34 ABC 123", 0.94],
        ], [0.01, 0.02, 0.03])
        self.assertEqual(_iter_rapidocr_text_candidates(payload), [("34 ABC 123", 0.94)])

    def test_rapidocr_parser_reads_recognition_only_format(self):
        payload = ([["34 ABC 123", 0.91]], [0.02])
        self.assertEqual(_iter_rapidocr_text_candidates(payload), [("34 ABC 123", 0.91)])

    def test_six_digit_plate_preserves_leading_zero(self):
        self.assertEqual(_normalize_plate_candidate("012345"), "012345")

    def test_civilian_plate_stays_on_civilian_path(self):
        self.assertEqual(_normalize_plate_candidate("34 ABC 123"), "34ABC123")

    def test_single_six_digit_candidate_is_not_accepted(self):
        plate, confidence, decision = _choose_ocr_result([
            {"plate": "012345", "confidence": 0.99},
            {"plate": None, "confidence": 0.0},
        ])
        self.assertIsNone(plate)
        self.assertIsNone(confidence)
        self.assertEqual(decision, "six_digit_candidate_requires_consensus")

    def test_two_six_digit_candidates_are_accepted(self):
        plate, confidence, decision = _choose_ocr_result([
            {"plate": "012345", "confidence": 0.91},
            {"plate": "012345", "confidence": 0.94},
        ])
        self.assertEqual(plate, "012345")
        self.assertEqual(confidence, 0.94)
        self.assertEqual(decision, "six_digit_multi_crop_consensus")

    def test_rejected_filename_gets_single_vehicle_absent_suffix(self):
        from pathlib import Path
        self.assertEqual(_status_filename(Path("ihlal_50kmh.jpg"), "_ARAC_YOK"), "ihlal_50kmh_ARAC_YOK.jpg")
        self.assertEqual(_status_filename(Path("ihlal_50kmh_ARAC_YOK.jpg"), "_ARAC_YOK"), "ihlal_50kmh_ARAC_YOK.jpg")


if __name__ == "__main__":
    unittest.main()

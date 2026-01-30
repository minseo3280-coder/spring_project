# 파일명: plate_ocr.py

import cv2
import numpy as np
import re
import logging
import os
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# 로깅 설정
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =====================================================================
# 1. 전처리 클래스 (EasyOCR 최적화)
# =====================================================================
class PlateImagePreprocessor:
    @staticmethod
    def preprocess_for_ocr(plate_image: np.ndarray) -> np.ndarray:
        # Step 1: 그레이스케일 변환
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image

        # Step 2: 이미지 확대 (2배) - 작은 번호판 인식률 향상
        enlarged = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        
        # Step 3: 명암 개선 (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(enlarged)
        
        # Step 4: 노이즈 제거
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10, templateWindowSize=7, searchWindowSize=21)
        
        return denoised

# =====================================================================
# 2. 기울기 보정 클래스
# =====================================================================
class PlateDeskewer:
    @staticmethod
    def deskew_plate(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 50)
        
        if lines is None or len(lines) == 0:
            return image
        
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            angles.append(angle)
        
        if not angles: return image
        
        angles = sorted(angles)
        median_angle = np.median(angles[len(angles)//4:-len(angles)//4]) if len(angles) > 4 else np.median(angles)
        
        if abs(median_angle) > 30: return image
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return rotated

# =====================================================================
# 3. 다중 엔진 OCR (Paddle + EasyOCR)
# =====================================================================
class MultiEngineOCR:
    def __init__(self):
        self.engines = {}
        self._initialize_engines()
    
    def _initialize_engines(self):
        # 1. PaddleOCR 시도
        try:
            from paddleocr import PaddleOCR
            import logging as py_logging
            py_logging.getLogger("ppocr").setLevel(py_logging.ERROR)
            
            self.engines['paddle'] = PaddleOCR(
                use_angle_cls=True, lang='korean', use_gpu=False, show_log=False
            )
        except Exception as e:
            logger.warning(f"⚠️ PaddleOCR 로드 실패 (EasyOCR 사용): {e}")

        # 2. EasyOCR 시도 (필수)
        try:
            import easyocr
            self.engines['easy'] = easyocr.Reader(['ko', 'en'], gpu=False)
        except Exception as e:
            logger.error(f"❌ EasyOCR 로드 실패: {e}")

    def recognize_with_all_engines(self, plate_image: np.ndarray):
        results = {}
        confidences = {}

        # PaddleOCR
        if 'paddle' in self.engines:
            try:
                p_res = self.engines['paddle'].ocr(plate_image, cls=True)
                if p_res and p_res[0]:
                    txts = [line[1][0] for line in p_res[0]]
                    confs = [line[1][1] for line in p_res[0]]
                    results['paddle'] = "".join(txts)
                    confidences['paddle'] = np.mean(confs)
            except: pass

        # EasyOCR
        if 'easy' in self.engines:
            try:
                e_res = self.engines['easy'].readtext(plate_image)
                if e_res:
                    txts = [res[1] for res in e_res]
                    confs = [res[2] for res in e_res]
                    results['easy'] = "".join(txts)
                    confidences['easy'] = np.mean(confs)
            except: pass

        if results:
            best_engine = max(confidences.items(), key=lambda x: x[1])[0]
            return {
                'text': results[best_engine],
                'engine': best_engine,
                'confidence': confidences[best_engine]
            }
        return {'text': '', 'engine': 'none', 'confidence': 0.0}

# =====================================================================
# 4. 후처리 (정규화)
# =====================================================================
class OCRPostProcessor:
    @staticmethod
    def postprocess_korean_plate(text: str) -> str:
        if not text: return ''
        text = text.replace(' ', '').replace('-', '').replace('.', '')
        
        # 오인식 문자 교정
        corrections = {'O': '0', 'I': '1', 'S': '5', 'l': '1', 'Z': '2', 'B': '8', 'G': '9', 'A': '4', 'T': '1', 'o': '0'}
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
            
        # 한글, 숫자만 남김
        text = re.sub(r'[^가-힣0-9]', '', text)
        return text
    
    @staticmethod
    def validate_plate_format(text: str):
        if not text: return False, "텍스트 없음"
        if len(text) < 7: return False, "길이 짧음"
        if not re.findall(r'[가-힣]', text): return False, "한글 없음"
        if len(re.findall(r'\d', text)) < 6: return False, "숫자 부족"
        return True, "유효함"

# =====================================================================
# 5. 통합 OCR 파이프라인
# =====================================================================
class HighAccuracyOCR:
    def __init__(self):
        self.preprocessor = PlateImagePreprocessor()
        self.deskewer = PlateDeskewer()
        self.multi_ocr = MultiEngineOCR()
        self.postprocessor = OCRPostProcessor()
    
    def recognize_plate(self, plate_image: np.ndarray):
        # 1. 각도 보정
        plate_image = self.deskewer.deskew_plate(plate_image)
        # 2. 전처리
        preprocessed = self.preprocessor.preprocess_for_ocr(plate_image)
        # 3. 인식
        ocr_result = self.multi_ocr.recognize_with_all_engines(preprocessed)
        # 4. 후처리
        raw_text = ocr_result['text']
        normalized = self.postprocessor.postprocess_korean_plate(raw_text)
        is_valid, msg = self.postprocessor.validate_plate_format(normalized)
        
        return {
            'normalized_text': normalized,
            'is_valid': is_valid,
            'ocr_confidence': ocr_result.get('confidence', 0.0)
        }

# =====================================================================
# 6. [핵심] 서버 연동용 모듈 (YOLO + Voting 포함)
# =====================================================================
class PlateRecognizerModule:
    """서버에서 위반 구간 영상을 받아 번호판을 추출하는 클래스"""
    def __init__(self, model_path: str):
        print(f"🔧 번호판 인식 모듈 초기화 중... (YOLO: {model_path})")
        self.model = YOLO(model_path) 
        self.ocr = HighAccuracyOCR()
        
    def process_segment(self, video_path: str, start_frame: int, count: int):
        """
        특정 영상의 특정 구간(start_frame부터 count만큼)만 읽어서
        가장 많이 검출된(Voting) 번호판 텍스트를 반환
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        detected_plates = []
        
        # print(f"🔍 번호판 정밀 분석 시작 (구간: {start_frame} ~ {start_frame+count})")
        
        for _ in range(count):
            ret, frame = cap.read()
            if not ret: break
            
            # 1. YOLO로 번호판 위치 탐지
            results = self.model(frame, conf=0.4, verbose=False)
            if not results: continue
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # 좌표 보정
                h, w = frame.shape[:2]
                pad = 5
                crop = frame[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
                
                if crop.size == 0: continue

                # 2. OCR 수행
                ocr_res = self.ocr.recognize_plate(crop)
                
                if ocr_res['is_valid']:
                    detected_plates.append(ocr_res['normalized_text'])

        cap.release()
        
        # 3. 투표 (최빈값 선정)
        if detected_plates:
            # 가장 많이 나온 번호와 횟수
            most_common = Counter(detected_plates).most_common(1)[0]
            plate_text, count = most_common
            
            # 최소 2번 이상은 동일하게 인식되어야 인정
            if count >= 2:
                return plate_text
            else:
                return f"{plate_text}(불확실)"
        
        return "식별불가"
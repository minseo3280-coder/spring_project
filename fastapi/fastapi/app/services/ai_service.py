import os
import cv2
import numpy as np
import tensorflow as tf
import csv
import requests
import urllib.parse  # 👈 파일명 디코딩을 위해 추가
from datetime import datetime
from app.core.config import (
    MODEL_PATH, YOLO_PATH, SEQUENCE_LENGTH, STEP_SIZE, 
    CATEGORIES, CSV_FILE, TEMP_VIDEO_DIR,
    USE_JAVA_SYNC, JAVA_SERVER_URL
)
from app.core.global_state import detection_logs
from app.services.s3_service import s3_manager
from .llm_service import get_llm_manager  # 👈 추가

# 🟢 번호판 인식 모듈 임포트
try:
    from .plate_ocr import PlateRecognizerModule
except ImportError:
    print("❌ 에러: plate_ocr.py 파일을 찾을 수 없습니다.")
    PlateRecognizerModule = None

# 💡 중복 처리 방지를 위한 전역 변수
processing_files = set()

class AIService:
    def __init__(self):
        # 1. 위반 감지 모델 로드
        print("⏳ AI 모델 로딩 중...")
        self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
        # 2. 번호판 인식기 로드
        try:
            self.lpr_system = PlateRecognizerModule(YOLO_PATH)
            print("✅ AI 및 번호판 인식 시스템 로드 완료")
        except Exception as e:
            print(f"⚠️ 번호판 인식 모델 로드 실패: {e}")
            self.lpr_system = None
            
        # CSV 헤더 초기화
        if not os.path.exists(CSV_FILE):
            with open(CSV_FILE, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['시간', '위반종류', '차량번호', '신뢰도', '위치', '파일명'])

    def process_video_task(self, video_key):
        """백그라운드에서 실행될 메인 판독 로직 (분석 -> 초안 생성 순서)"""
        
        # 1. 영상 준비 및 전처리
        decoded_key = urllib.parse.unquote_plus(video_key)
        filename = os.path.basename(decoded_key)
        
        # 중복 방지 로직
        if filename in processing_files: return
        processing_files.add(filename)

        try:
            local_path = os.path.join(TEMP_VIDEO_DIR, filename)
            s3_manager.download_file(decoded_key, local_path) #

            # 2. [영상 분석 단계] 위반 행동 인식
            cap = cv2.VideoCapture(local_path)
            all_frames = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                all_frames.append(cv2.resize(frame, (128, 128)) / 255.0)
            cap.release()

            # AI 모델 예측 실행
            windows = [all_frames[i : i + SEQUENCE_LENGTH] for i in range(0, len(all_frames) - SEQUENCE_LENGTH + 1, STEP_SIZE)]
            predictions = self.model.predict(np.array(windows), batch_size=2)

            best_prob, best_class_idx, best_window_idx = 0, -1, -1
            for i, pred in enumerate(predictions):
                idx = np.argmax(pred)
                if pred[idx] > best_prob:
                    best_prob, best_class_idx, best_window_idx = pred[idx], idx, i

            # 3. [번호판 인식 단계] 위반 감지 시에만 수행
            if best_class_idx != -1:
                result_label = CATEGORIES[best_class_idx]
                start_frame = best_window_idx * STEP_SIZE
                
                # 번호판 인식 모듈 호출
                plate_text = "인식 모듈 미작동"
                if self.lpr_system:
                    raw_plate = self.lpr_system.process_segment(local_path, start_frame, SEQUENCE_LENGTH)
                    plate_text = raw_plate if raw_plate and raw_plate != "식별불가" else "번호판 인식 불가"
                
                now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                location = "수원시 팔달구 매산로 1"
                video_url = s3_manager.get_presigned_url(decoded_key) #

                # ---------------------------------------------------------
                # 4. [LLM 초안 작성 단계] 모든 분석 데이터가 준비된 후 실행
                # ---------------------------------------------------------
                print(f"📝 분석 결과 기반 신고 초안 생성 시작: {result_label}")
                
                # LLM에 전달할 질문 구성
                llm_input = (
                    f"일시: {now_time}, "
                    f"위치: {location}, "
                    f"위반 항목: {result_label}, "
                    f"차량번호: {plate_text}"
                )
                
                # llm_service의 신고 초안 생성 함수 호출
                llm_manager = get_llm_manager()
                report_draft = llm_manager.get_report_draft(llm_input)
                # ---------------------------------------------------------

                # 5. [결과 전송] 초안이 포함된 페이로드 구성
                payload = {
                    "result": result_label,
                    "plate": plate_text,
                    "location": location,
                    "time": now_time,
                    "prob": float(best_prob * 100),
                    "info": filename,
                    "video_url": video_url,
                    "report_draft": report_draft  # LLM이 생성한 초안 추가
                }
                
                detection_logs.append(payload)

                # 자바 서버(Spring Boot)로 최종 데이터 전송
                if USE_JAVA_SYNC:
                    try:
                        requests.post(JAVA_SERVER_URL, json=payload, timeout=3)
                        print(f"✅ 분석 및 초안 전송 완료: {result_label}")
                    except Exception:
                        print("⚠️ 자바 서버 전송 실패")

            # 파일 정리
            if os.path.exists(local_path): os.remove(local_path)
            processing_files.remove(filename)

        except Exception as e:
            print(f"❌ 분석 중 에러: {e}")
            if filename in processing_files: processing_files.remove(filename)

# 싱글톤 인스턴스
ai_manager = AIService()
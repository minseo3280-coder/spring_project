from fastapi import APIRouter, Request, BackgroundTasks, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import os
import requests 

from app.core.config import TEMP_VIDEO_DIR, BUCKET_NAME
from app.core.global_state import detection_logs
from app.services.s3_service import s3_manager
from app.services.ai_service import ai_manager
from app.services.llm_service import get_llm_manager

# 💡 [설정] ngrok 대신 학원 컴퓨터 로컬(localhost) 주소를 사용합니다.
USE_JAVA_SYNC = True 

# 자바 서버(Spring Boot)의 수신 엔드포인트 주소 설정
JAVA_TARGET_URL = "http://localhost:5001/api/chatbot-response"
JAVA_VIOLATION_URL = "http://localhost:5001/api/violations"

llm_manager = get_llm_manager()
router = APIRouter()

# 템플릿 경로 설정
templates = Jinja2Templates(directory="D:\\차량신고시스템\\fastapi\\fastapi\\app\\templates")

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """관제 시스템 메인 페이지 렌더링"""
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/upload-video")
async def upload_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """로컬 영상을 S3에 업로드하고 분석을 시작하는 엔드포인트"""
    try:
        temp_file = os.path.join(TEMP_VIDEO_DIR, file.filename)
        with open(temp_file, "wb") as buffer:
            buffer.write(await file.read())
        
        # S3 업로드
        s3_key = f"raspberrypi_video/{file.filename}"
        s3_manager.upload_file(temp_file, s3_key)
        
        # 백그라운드에서 AI 분석 시작
        if background_tasks:
            background_tasks.add_task(ai_manager.process_video_task, s3_key)
        
        # 임시 파일 삭제
        if os.path.exists(temp_file): 
            os.remove(temp_file)
        
        # React용 JSON 응답 반환
        return JSONResponse(content={
            "success": True,
            "message": "영상 업로드 및 분석이 시작되었습니다.",
            "filename": file.filename,
            "s3_key": s3_key
        }, status_code=200)
    except Exception as e:
        print(f"❌ 업로드 에러: {e}")
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        }, status_code=500)

@router.get("/api/logs")
async def get_logs():
    """AI 분석 로그 데이터를 브라우저 및 자바 서버에 반환"""
    updated_logs = []
    for log in detection_logs:
        video_key = f"raspberrypi_video/{log['info']}"
        # S3에서 영상 재생을 위한 미리보기 URL 생성
        updated_logs.append({**log, "video_url": s3_manager.get_presigned_url(video_key)})
    return updated_logs

@router.post("/s3-webhook")
async def s3_webhook(request: Request, background_tasks: BackgroundTasks):
    """S3 업로드 신호를 감지하여 AI 분석 작업 시작"""
    data = await request.json()
    for record in data.get('Records', []):
        video_key = record['s3']['object']['key']
        if video_key.lower().endswith('.mp4'):
            print(f"🔔 S3 신호 수신: {video_key}")
            # 비동기 방식으로 영상 분석 실행
            background_tasks.add_task(ai_manager.process_video_task, video_key)
    return {"status": "ok"}

# --- [LLM 채팅 연동 핵심 구간] ---

@router.post("/api/ask")
async def ask_traffic_llm(request: Request):
    """자바 서버와 연동하여 챗봇 질문 답변 처리 및 동기화 (분기 로직 추가)"""
    try:
        data = await request.json()
        question = data.get("question")
        
        if not question:
            return {"answer": "질문이 없습니다."}
        
        # ---------------------------------------------------------
        # 1. AI 답변 생성 (질문 내용에 따른 프롬프트 분기 처리)
        # ---------------------------------------------------------
        # 질문에 '신고' 또는 '초안'이라는 단어가 포함되어 있는지 확인합니다.
        if "신고" in question or "초안" in question:
            print(f"📝 신고 초안 모드 가동: {question[:15]}...")
            answer = llm_manager.get_report_draft(question)  # 초안 전용 프롬프트 사용
        else:
            print(f"⚖️ 법률 상담 모드 가동: {question[:15]}...")
            answer = llm_manager.get_law_answer(question)   # 법률 전문가 프롬프트 사용
        # ---------------------------------------------------------
        
        # 2. 자바 서버로 답변 내용 전송 (데이터 동기화)
        if USE_JAVA_SYNC:
            try:
                chatbot_payload = {"answer": answer, "question": question}
                # 로컬 자바 서버(8080)로 데이터 전송
                requests.post(JAVA_TARGET_URL, json=chatbot_payload, timeout=3)
                print(f"🚀 자바 서버(Local)로 챗봇 데이터 전송 성공!")
            except Exception as e:
                print(f"⚠️ 자바 서버 전송 실패: {e}")

        return {"answer": answer}
        
    except Exception as e:
        print(f"LLM 에러: {e}")
        return {"answer": f"서버 오류 발생: {str(e)}"}

@router.get("/api/ask")
def ask_simple(question: str):
    """테스트용 단순 GET 방식 질문 엔드포인트"""
    answer = llm_manager.get_law_answer(question)
    return {"answer": answer}

from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware # 👈 [필수] 세션 기능 추가
from fastapi.middleware.cors import CORSMiddleware # [선택] 프론트엔드 연동 위해 추가 추천
# 라우터 파일들 임포트
from app.routers import traffic, login  # 👈 [추가] auth 라우터 임포트

app = FastAPI(title="AI 교통관제 시스템")

# 1. 세션 미들웨어 설정 (카카오 로그인 시 사용자 정보 저장용)
# secret_key는 임의의 문자열을 넣으시면 됩니다.
app.add_middleware(SessionMiddleware, secret_key="[ENCRYPTION_KEY]")

# CORS 설정 - React 앱과 통신을 위해 필수
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5001", "http://localhost:5001"],  # React 개발 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(traffic.router) # 기존 트래픽 라우터
app.include_router(login.router)    # 👈 [추가] 카카오 인증 라우터

# 헬스 체크용 기본 엔드포인트
@app.get("/")
def read_root():
    return {"status": "running", "message": "AI 관제 시스템 가동 중"}

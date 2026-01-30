import uvicorn
import os
import sys
import boto3
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경변수 읽기
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
NGROK_TOKEN = os.getenv("NGROK_AUTHTOKEN")

# ⚠️ 본인의 AWS Lambda 함수 이름을 정확히 적어주세요!
LAMBDA_FUNCTION_NAME = "s3-to-fastapi" 

try:
    from pyngrok import ngrok
except ImportError:
    print("❌ pyngrok가 설치되지 않았습니다. 'pip install pyngrok'를 실행하세요.")
    sys.exit(1)

def update_lambda_env(public_url):
    """AWS Lambda의 환경변수를 자동으로 수정하는 함수"""
    print("⏳ AWS Lambda 설정 업데이트 중...")
    try:
        client = boto3.client('lambda',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )
        
        # Lambda 환경변수 업데이트
        client.update_function_configuration(
            FunctionName=LAMBDA_FUNCTION_NAME,
            Environment={
                'Variables': {
                    'TARGET_URL': public_url
                }
            }
        )
        print(f"✅ AWS Lambda [{LAMBDA_FUNCTION_NAME}] 주소 업데이트 완료!")
        print(f"   새 타겟: {public_url}")
        
    except Exception as e:
        print(f"❌ AWS 설정 실패: {e}")

if __name__ == "__main__":
    PORT = 5000
    
    # 1. Ngrok 인증 및 터널 열기
    if NGROK_TOKEN:
        ngrok.set_auth_token(NGROK_TOKEN)
    
    try:
        ngrok.kill()
        public_url = ngrok.connect(PORT).public_url
        
        print("=" * 60)
        print(f"🚀 Ngrok 터널 개방: {public_url}")
        
        # 2. AWS Lambda 주소 자동 변경
        update_lambda_env(public_url)
        
        print("=" * 60)

        # 3. 서버 실행
        uvicorn.run("app.main:app", host="127.0.0.1", port=PORT, reload=False)
    except Exception as e:
        print(f"❌ 실행 에러: {e}")
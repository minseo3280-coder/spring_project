// 모든 API 및 인증 요청은 실제 로직이 있는 파이썬 서버(5000)로 보냅니다.
const KAKAO_SERVER_URL = 'http://localhost:5000';
const KAKAO_API_BASE = `${KAKAO_SERVER_URL}/api`;

// ===== 카카오 인증 관리자 =====
const KakaoAuthManager = {
    // 로그인 상태 확인
    checkAuthStatus: async () => {
        try {
            // 자바(5001)가 아닌 파이썬(5000)에 물어봐야 합니다.
            const response = await fetch(`${KAKAO_SERVER_URL}/auth/check`);
            const data = await response.json();
            return data.authenticated;
        } catch (error) {
            console.error('인증 상태 확인 실패:', error);
            return false;
        }
    },

    // 사용자 정보 조회
    getUserInfo: async () => {
        try {
            const response = await fetch(`${KAKAO_API_BASE}/user/info`);
            if (response.status === 401) {
                return null;
            }
            const data = await response.json();
            return data.success ? data.user : null;
        } catch (error) {
            console.error('사용자 정보 조회 실패:', error);
            return null;
        }
    },

    // 로그인 시작 (Whitelabel 에러 해결 핵심!)
    login: () => {
        // 상대 경로 '/auth/kakao/login'을 쓰면 자바 서버(5001)를 찾아가서 에러가 납니다.
        // 파이썬 서버 주소를 직접 적어줘야 합니다.
        window.location.href = `${KAKAO_SERVER_URL}/auth/kakao/login`;
    },

    // 로그아웃
    logout: async () => {
        try {
            const response = await fetch(`${KAKAO_SERVER_URL}/auth/logout`, {
                method: 'POST'
            });
            const data = await response.json();
            if (data.success) {
                window.location.reload();
            }
            return data.success;
        } catch (error) {
            console.error('로그아웃 실패:', error);
            return false;
        }
    }
};

// ===== UI 관리 =====
const UIManager = {
    showMessage: (text, type = 'info') => {
        const messageEl = document.getElementById('message');
        if (!messageEl) return;
        messageEl.textContent = text;
        messageEl.className = `message ${type}`;
        messageEl.classList.remove('hidden');
        setTimeout(() => {
            messageEl.classList.add('hidden');
        }, 5000);
    },

    showLoginSection: () => {
        document.getElementById('loginSection')?.classList.remove('hidden');
        document.getElementById('userSection')?.classList.add('hidden');
    },

    showUserSection: (userData) => {
        document.getElementById('loginSection')?.classList.add('hidden');
        document.getElementById('userSection')?.classList.remove('hidden');

        if (document.getElementById('userName')) document.getElementById('userName').textContent = userData.nickname || '사용자';
        if (document.getElementById('userEmail')) document.getElementById('userEmail').textContent = userData.email || '이메일 정보 없음';
        if (document.getElementById('userId')) document.getElementById('userId').textContent = userData.id || '-';
        if (document.getElementById('loginTime')) document.getElementById('loginTime').textContent = new Date().toLocaleString('ko-KR');

        if (userData.profile_image_url && document.getElementById('profileImage')) {
            document.getElementById('profileImage').innerHTML = `<img src="${userData.profile_image_url}" alt="프로필" style="width:50px; border-radius:50%;">`;
        }
    }
};

// ===== API 테스트 함수 =====
async function getUserInfoAPI() {
    try {
        UIManager.showMessage('사용자 정보 조회 중...', 'info');
        const response = await fetch(`${KAKAO_API_BASE}/user/info`);
        if (response.status === 401) {
            UIManager.showMessage('로그인이 필요합니다', 'error');
            return;
        }
        const data = await response.json();
        showAPIResponse('사용자 정보', data.user);
        UIManager.showMessage('✓ 사용자 정보 조회 성공', 'success');
    } catch (error) {
        UIManager.showMessage(`❌ 사용자 정보 조회 실패: ${error.message}`, 'error');
    }
}

async function getAccessTokenInfo() {
    try {
        UIManager.showMessage('액세스 토큰 정보 조회 중...', 'info');
        const response = await fetch(`${KAKAO_API_BASE}/user/token/info`);
        if (response.status === 401) {
            UIManager.showMessage('로그인이 필요합니다', 'error');
            return;
        }
        const data = await response.json();
        if (data.success) {
            showAPIResponse('액세스 토큰 정보', data.token_info);
            UIManager.showMessage('✓ 토큰 정보 조회 성공', 'success');
        } else {
            UIManager.showMessage(`❌ 토큰 조회 실패: ${data.error}`, 'error');
        }
    } catch (error) {
        UIManager.showMessage(`❌ 토큰 조회 실패: ${error.message}`, 'error');
    }
}

function showAPIResponse(title, data) {
    const responseEl = document.getElementById('apiResponse');
    if (!responseEl) return;
    responseEl.textContent = `[${title}]\n${JSON.stringify(data, null, 2)}`;
    responseEl.classList.remove('hidden');
}

// ===== 이벤트 리스너 등록 =====
document.addEventListener('DOMContentLoaded', async () => {
    // 로그인 상태 확인 (파이썬 서버에 물어봄)
    const isAuthenticated = await KakaoAuthManager.checkAuthStatus();

    if (isAuthenticated) {
        const userData = await KakaoAuthManager.getUserInfo();
        if (userData) {
            UIManager.showUserSection(userData);
        }
    } else {
        UIManager.showLoginSection();
    }

    // 버튼 이벤트 등록
    document.getElementById('kakaoLoginBtn')?.addEventListener('click', () => {
        KakaoAuthManager.login();
    });

    document.getElementById('logoutBtn')?.addEventListener('click', async () => {
        const success = await KakaoAuthManager.logout();
        if (!success) {
            UIManager.showMessage('로그아웃에 실패했습니다', 'error');
        }
    });

    document.getElementById('getUserInfoBtn')?.addEventListener('click', getUserInfoAPI);
    document.getElementById('getAccessTokenBtn')?.addEventListener('click', getAccessTokenInfo);
});
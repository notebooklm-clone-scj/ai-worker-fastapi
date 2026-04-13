import logging
import time
import uuid

from fastapi import FastAPI, Request
from api.pdf_router import router as pdf_router
from api.chat_router import router as chat_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

logger = logging.getLogger("ai_worker")

# FastAPI 객체 생성 (이 한 줄이 스프링의 @SpringBootApplication 역할)
app = FastAPI(
    title="NotebookLM AI Worker",
    description="PDF 텍스트 추출 및 AI 처리용 파이썬 서버",
    version="1.0.0"
)

# 모든 요청에 requestId를 부여하고 공통 완료 로그를 남김
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))
    request.state.request_id = request_id
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        response.headers["X-Request-Id"] = request_id

        logger.info(
            "event=http_request_complete requestId=%s method=%s path=%s status=%s latencyMs=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            latency_ms
        )
        return response
    except Exception:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        logger.exception(
            "event=http_request_failed requestId=%s method=%s path=%s latencyMs=%s",
            request_id,
            request.method,
            request.url.path,
            latency_ms
        )
        raise

# 1. 서버 잘 켜졌는지 확인
@app.get("/")
def health_check():
    return {"status": "ok", "message": "AI Worker 서버가 정상 작동"}

# 실제 기능은 라우터 단위로 분리해서 등록한다.
app.include_router(pdf_router)
app.include_router(chat_router)

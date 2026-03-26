FROM python:3.11-slim
WORKDIR /app

# 라이브러리 먼저 설치 (캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 코드를 복사하긴 하지만, 개발 중에는 docker-compose의 볼륨이 이걸 덮어씌울 겁니다!
COPY . .

EXPOSE 8000
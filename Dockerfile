# Local CapScout app. The process inside the container listens on all
# interfaces so Docker can forward a port. docker-compose.yml publishes
# that port only on 127.0.0.1.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--browser.gatherUsageStats", "false"]

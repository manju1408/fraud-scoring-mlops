FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY fraud ./fraud
RUN python -m fraud.train --out /app/artifacts
ENV MODEL_DIR=/app/artifacts
EXPOSE 8000
CMD ["uvicorn", "fraud.serve:app", "--host", "0.0.0.0", "--port", "8000"]

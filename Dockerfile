﻿FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# (προαιρετικό, αλλά χρήσιμο αν χρησιμοποιήσεις βιβλιοθήκες που θέλουν OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080
# ΣΗΜΑΝΤΙΚΟ: Τρέξε main.py ώστε να ακούσει σε 0.0.0.0:8080
CMD ["python", "-u", "main.py"]

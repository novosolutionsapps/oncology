# Builder-independent image for the Railway dashboard deploy.
# Railway migrated its default builder from Nixpacks -> Railpack; rebuilding the
# previously-working Nixpacks config now fails. A Dockerfile is builder-independent
# and reuses the existing pinned requirements (all prebuilt wheels -> no compilation).
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + data + variant_review.html (served by a Flask route in dashboard.py).
COPY . .

# Railway injects PORT at runtime. dashboard.py exposes `server = app.server`.
# gunicorn.conf.py reads PORT via os.environ (NOT shell expansion) to avoid the
# "'$PORT' is not a valid port number" crash when the start command runs via exec.
ENV PORT=8080
CMD ["gunicorn", "dashboard:server", "-c", "gunicorn.conf.py"]

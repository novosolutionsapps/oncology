import os

# Railway injects $PORT at runtime. Read it in Python (NOT via shell variable
# expansion) so the bind address never depends on the start command being run
# through a shell. This fixes the "'$PORT' is not a valid port number" crash that
# happens when the start command is executed via exec (e.g. a Procfile/dashboard
# command passing a literal $PORT to gunicorn).
bind = "0.0.0.0:" + os.environ.get("PORT", "8080")
workers = 2
timeout = 120

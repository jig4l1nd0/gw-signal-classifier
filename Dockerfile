# --- STAGE 1: 'Builder' ---
# Use a lightweight Python image as base
FROM python:3.10-slim-bookworm AS builder

WORKDIR /app

# 1. Copy only the requirements file
COPY requirements.txt .

# 2. Install dependencies
# Key! Use PyTorch CPU-only index to keep the image lightweight
RUN pip install --no-cache-dir \
    -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cpu

# --- STAGE 2: 'Final' ---
# Start again from the same clean base image
FROM python:3.10-slim-bookworm

WORKDIR /app

# 3. Copy installed packages from the 'builder' stage
# This makes the final image small and doesn't contain build files
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 4. Copy only the necessary application files to RUN it
# We don't copy 'scripts/', 'data/', etc.
COPY app.py .
COPY unet_gw_model.pth .
COPY example_signal.npy .
COPY src/ /app/src/

# 5. Expose Streamlit's standard port
EXPOSE 8501

# 6. Configure health check (so Render knows the app is alive)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# 7. Command to run the application
# Use 0.0.0.0 to make it accessible from outside the container
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir mlflow==2.19.0 pandas scikit-learn

# Copy the MLProject files
COPY MLProject/ /app/

# Expose port for serving
EXPOSE 5001

# Default command: serve the model
CMD ["python", "modelling.py"]

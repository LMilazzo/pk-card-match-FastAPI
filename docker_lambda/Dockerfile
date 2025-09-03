# ===== Base image: AWS Lambda Python 3.11 runtime =====
FROM public.ecr.aws/lambda/python:3.11

# ===== Copy and install dependencies =====
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ===== Copy FastAPI app code into container =====
COPY . .

# ===== Set Lambda entrypoint =====
# 'main.handler' uses your Mangum-wrapped FastAPI app
CMD ["main.handler"]

FROM python:3.10
LABEL authors="flash"

# Expose port you want your app on
EXPOSE 8080

# Upgrade pip and install requirements
COPY requirements.txt requirements.txt
RUN pip install -U pip
RUN pip install -r requirements.txt

# Copy app code and set working directory
COPY . .
WORKDIR /app

# Run
ENTRYPOINT ["streamlit", "run", "Home.py", "–server.port=8080", "–server.address=0.0.0.0"]
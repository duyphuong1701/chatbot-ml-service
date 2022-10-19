FROM python:3.8
WORKDIR /chatbot
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8765
CMD ["python", "app.py"]
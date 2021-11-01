# News-Article-Classifier
End-to-end pipeline for News article topic classification

- Live News data from mediastack api
- Kakfa + Zookeeper for data streaming
- MongoDB database
- API endpoints using FastAPI
- UI using ReactJS


## The setup

- Clone the repo
- Run `docker-compose up` (Sets up kafka and Zookeeper for data streaming + MongoDB and Mongo express)
- In a new terminal run `python bin/receiveNews.py send_stream` (Kafka consumer)
- In a new terminal run `python bin/sendNews.py send_stream` (Kafka Producer)
- In a new terminal navigate to `bin/fastapi directory`. Run `python fastapi_endpoints.py` (FastAPI endpoints)
- Navigate to the UI folder, and run `npm start` to start the react application.
- Use the application for all your News Article Classification needs.

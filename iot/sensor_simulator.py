import requests
import random
import time
import json

BACKEND_URL = "http://localhost:8000/get_sensor_data"  # Change to POST endpoint if needed
POST_URL = "http://localhost:8000/sensor_data"  # If backend accepts POST sensor data

def generate_sensor_data():
    data = {
        "soil_moisture": round(random.uniform(20, 80), 2),
        "temperature": round(random.uniform(20, 35), 2),
        "humidity": round(random.uniform(40, 90), 2)
    }
    return data

def main():
    while True:
        data = generate_sensor_data()
        print(f"Sending sensor data: {data}")
        # If backend supports POST sensor data, uncomment below:
        # response = requests.post(POST_URL, json=data)
        # print(f"Response: {response.status_code} {response.text}")
        time.sleep(5)

if __name__ == "__main__":
    main()
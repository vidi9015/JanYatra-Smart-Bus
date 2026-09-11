# JanYatra Prototype - Step 1

## What this prototype demonstrates
Camera/AI data can eventually be sent to a backend, and the backend can provide live occupancy information to the JanYatra dashboard.

Current MVP:
- Bus-wise passenger count
- Bus capacity
- Automatic Low/Moderate/High crowd classification
- Route, location and arrival information
- SOS demo button
- Feedback/rating demo
- REST API endpoints
- Dashboard refreshes every 5 seconds

## Run it

1. Install Python 3.
2. Open a terminal in this folder.
3. Install Flask:

   pip install -r requirements.txt

4. Start the server:

   python app.py

5. Open the address shown in the terminal, usually:

   http://127.0.0.1:5000

## API
GET /api/buses
POST /api/update

The next stage will replace the manual passenger count with YOLO-based detection from a sample bus video/image.

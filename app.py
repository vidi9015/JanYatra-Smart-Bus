from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import os
from datetime import datetime

app = Flask(__name__)

# =========================================================
# UPLOAD CONFIGURATION
# =========================================================

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# =========================================================
# YOLO MODEL
# =========================================================

# Model is loaded only when AI detection is requested.
# This prevents the model from loading during normal startup.

model = None


def get_model():

    global model

    if model is None:

        print("Loading YOLO model...")

        model = YOLO("yolo11n.pt")

        print("YOLO model loaded successfully.")

    return model


# =========================================================
# BUS DATA
# =========================================================

buses = [

    {
        "bus_id": "RJ14-1234",
        "route": "Jaipur → Ajmer",
        "passenger_count": 6,
        "capacity": 60,
        "arrival": "8 min",
        "location": "Near Sindhi Camp",
        "latitude": 26.9196,
        "longitude": 75.7878
    },

    {
        "bus_id": "RJ14-5678",
        "route": "Jaipur → Sanganer",
        "passenger_count": 24,
        "capacity": 60,
        "arrival": "5 min",
        "location": "Near Tonk Road",
        "latitude": 26.8720,
        "longitude": 75.8070
    },

    {
        "bus_id": "RJ14-9012",
        "route": "Jaipur → Amer",
        "passenger_count": 52,
        "capacity": 60,
        "arrival": "12 min",
        "location": "Near Jal Mahal",
        "latitude": 26.9530,
        "longitude": 75.8470
    }

]


# =========================================================
# FEEDBACK STORAGE
# =========================================================

feedback_list = []


# =========================================================
# HISTORICAL PASSENGER DATA
# =========================================================

historical_demand = {

    "Jaipur → Ajmer": {

        "8 AM": 42,
        "9 AM": 55,
        "10 AM": 38,
        "11 AM": 30,
        "12 PM": 28,
        "1 PM": 35,
        "2 PM": 32,
        "3 PM": 36,
        "4 PM": 45,
        "5 PM": 60,
        "6 PM": 72,
        "7 PM": 68,
        "8 PM": 52

    },

    "Jaipur → Sanganer": {

        "8 AM": 48,
        "9 AM": 58,
        "10 AM": 45,
        "11 AM": 40,
        "12 PM": 36,
        "1 PM": 42,
        "2 PM": 44,
        "3 PM": 48,
        "4 PM": 55,
        "5 PM": 65,
        "6 PM": 75,
        "7 PM": 70,
        "8 PM": 58

    },

    "Jaipur → Amer": {

        "8 AM": 30,
        "9 AM": 35,
        "10 AM": 40,
        "11 AM": 45,
        "12 PM": 50,
        "1 PM": 48,
        "2 PM": 52,
        "3 PM": 55,
        "4 PM": 60,
        "5 PM": 68,
        "6 PM": 78,
        "7 PM": 74,
        "8 PM": 60

    }

}


# =========================================================
# DEMAND STATUS
# =========================================================

def demand_status(passengers):

    if passengers <= 40:

        return "LOW"

    elif passengers <= 70:

        return "MODERATE"

    elif passengers <= 100:

        return "HIGH"

    else:

        return "VERY HIGH"


# =========================================================
# CROWD STATUS
# =========================================================

def crowd_status(passengers, capacity):

    if capacity <= 0:

        return "UNKNOWN"

    percentage = (passengers / capacity) * 100

    if percentage <= 40:

        return "LOW"

    elif percentage <= 70:

        return "MODERATE"

    elif percentage <= 100:

        return "HIGH"

    else:

        return "OVER-CAPACITY"


# =========================================================
# BUS INFORMATION
# =========================================================

def bus_information(bus):

    passengers = bus["passenger_count"]

    capacity = bus["capacity"]

    occupancy = (passengers / capacity) * 100

    data = bus.copy()

    data["occupancy"] = round(
        occupancy,
        1
    )

    data["crowd_status"] = crowd_status(
        passengers,
        capacity
    )

    return data


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# GET BUS INFORMATION
# =========================================================

@app.route("/api/buses")
def get_buses():

    result = []

    for bus in buses:

        result.append(
            bus_information(bus)
        )

    return jsonify(result)


# =========================================================
# SMART BUS RECOMMENDATION
# =========================================================

@app.route("/api/recommend")
def recommend():

    available = []

    for bus in buses:

        occupancy = (
            bus["passenger_count"]
            /
            bus["capacity"]
        ) * 100

        available.append(
            (
                occupancy,
                bus
            )
        )

    available.sort(
        key=lambda x: x[0]
    )

    best_bus = available[0][1]

    return jsonify({

        "success": True,

        "recommendation":
            bus_information(best_bus)

    })


# =========================================================
# AI PASSENGER DETECTION
# =========================================================

@app.route("/detect", methods=["POST"])
def detect():

    try:

        # -------------------------------------------------
        # CHECK IMAGE
        # -------------------------------------------------

        if "crowd_image" not in request.files:

            return jsonify({

                "success": False,

                "message":
                    "Please select an image."

            })

        file = request.files["crowd_image"]

        # -------------------------------------------------
        # GET BUS
        # -------------------------------------------------

        bus_id = request.form.get(
            "bus_id"
        )

        if file.filename == "":

            return jsonify({

                "success": False,

                "message":
                    "Please select an image."

            })

        if not bus_id:

            return jsonify({

                "success": False,

                "message":
                    "Please select a bus."

            })

        # -------------------------------------------------
        # SAVE IMAGE
        # -------------------------------------------------

        filename = secure_filename(
            file.filename
        )

        filepath = os.path.join(

            app.config["UPLOAD_FOLDER"],

            filename

        )

        file.save(filepath)

        print("Image saved:", filepath)

        # -------------------------------------------------
        # LOAD YOLO MODEL
        # -------------------------------------------------

        yolo_model = get_model()

        # -------------------------------------------------
        # RUN YOLO ON CPU
        # Smaller image size reduces memory usage.
        # -------------------------------------------------

        results = yolo_model.predict(

            source=filepath,

            conf=0.35,

            imgsz=320,

            device="cpu",

            fuse=False,

            verbose=False

        )

        # -------------------------------------------------
        # COUNT PERSONS
        #
        # YOLO class 0 = person
        # -------------------------------------------------

        passenger_count = 0

        for result in results:

            if result.boxes is not None:

                for cls in result.boxes.cls:

                    if int(cls) == 0:

                        passenger_count += 1

        # Release prediction results
        del results

        # -------------------------------------------------
        # FIND SELECTED BUS
        # -------------------------------------------------

        selected_bus = None

        for bus in buses:

            if bus["bus_id"] == bus_id:

                selected_bus = bus

                break

        if selected_bus is None:

            return jsonify({

                "success": False,

                "message":
                    "Selected bus not found."

            })

        # -------------------------------------------------
        # UPDATE PASSENGER COUNT
        # -------------------------------------------------

        selected_bus["passenger_count"] = (
            passenger_count
        )

        information = bus_information(
            selected_bus
        )

        # -------------------------------------------------
        # PRINT AI RESULT
        # -------------------------------------------------

        print()
        print("--------------------------------")
        print("JANyatra AI Passenger Detection")
        print("--------------------------------")
        print(
            "Bus:",
            bus_id
        )
        print(
            "Passengers detected:",
            passenger_count
        )
        print(
            "Capacity:",
            information["capacity"]
        )
        print(
            "Occupancy:",
            information["occupancy"],
            "%"
        )
        print(
            "Crowd Status:",
            information["crowd_status"]
        )
        print("--------------------------------")
        print()

        # -------------------------------------------------
        # RETURN RESULT
        # -------------------------------------------------

        return jsonify({

            "success": True,

            "bus_id":
                information["bus_id"],

            "passenger_count":
                information["passenger_count"],

            "capacity":
                information["capacity"],

            "occupancy":
                information["occupancy"],

            "crowd_status":
                information["crowd_status"],

            "location":
                information["location"]

        })

    except Exception as error:

        print(
            "AI ERROR:",
            error
        )

        return jsonify({

            "success": False,

            "message":
                "AI detection error: "
                + str(error)

        }), 500


# =========================================================
# BUS LOCATION
# =========================================================

@app.route("/api/location/<bus_id>")
def location(bus_id):

    for bus in buses:

        if bus["bus_id"] == bus_id:

            return jsonify({

                "success": True,

                "bus_id":
                    bus["bus_id"],

                "route":
                    bus["route"],

                "location":
                    bus["location"],

                "latitude":
                    bus["latitude"],

                "longitude":
                    bus["longitude"]

            })

    return jsonify({

        "success": False,

        "message":
            "Bus not found."

    }), 404


# =========================================================
# UPDATE PASSENGER COUNT
# =========================================================

@app.route("/api/update", methods=["POST"])
def update():

    try:

        data = request.get_json()

        bus_id = data["bus_id"]

        passenger_count = int(
            data["passenger_count"]
        )

        for bus in buses:

            if bus["bus_id"] == bus_id:

                bus["passenger_count"] = max(
                    0,
                    passenger_count
                )

                return jsonify({

                    "success": True,

                    "bus":
                        bus_information(bus)

                })

        return jsonify({

            "success": False,

            "message":
                "Bus not found."

        })

    except Exception as error:

        return jsonify({

            "success": False,

            "message":
                str(error)

        }), 500


# =========================================================
# SOS
# =========================================================

@app.route("/api/sos", methods=["POST"])
def sos():

    try:

        data = request.get_json()

        bus_id = data.get(
            "bus_id"
        )

        if not bus_id:

            return jsonify({

                "success": False,

                "message":
                    "Please select a bus."

            })

        selected_bus = None

        for bus in buses:

            if bus["bus_id"] == bus_id:

                selected_bus = bus

                break

        if selected_bus is None:

            return jsonify({

                "success": False,

                "message":
                    "Bus not found."

            })

        information = bus_information(
            selected_bus
        )

        current_time = datetime.now().strftime(
            "%d-%m-%Y %I:%M:%S %p"
        )

        print()
        print("--------------------------------")
        print("JANyatra SOS ALERT")
        print("--------------------------------")
        print(
            "Time:",
            current_time
        )
        print(
            "Bus:",
            information["bus_id"]
        )
        print(
            "Route:",
            information["route"]
        )
        print(
            "Location:",
            information["location"]
        )
        print("--------------------------------")

        return jsonify({

            "success": True,

            "message":
                "SOS alert activated successfully!",

            "time":
                current_time,

            "bus_id":
                information["bus_id"],

            "route":
                information["route"],

            "location":
                information["location"],

            "passenger_count":
                information["passenger_count"],

            "capacity":
                information["capacity"],

            "occupancy":
                information["occupancy"],

            "crowd_status":
                information["crowd_status"]

        })

    except Exception as error:

        print(
            "SOS ERROR:",
            error
        )

        return jsonify({

            "success": False,

            "message":
                "SOS error: "
                + str(error)

        }), 500


# =========================================================
# SUBMIT FEEDBACK
# =========================================================

@app.route("/api/feedback", methods=["POST"])
def submit_feedback():

    try:

        data = request.get_json()

        bus_id = data.get(
            "bus_id"
        )

        rating = data.get(
            "rating"
        )

        comment = data.get(
            "comment",
            ""
        ).strip()

        if not bus_id:

            return jsonify({

                "success": False,

                "message":
                    "Please select a bus."

            })

        if not rating:

            return jsonify({

                "success": False,

                "message":
                    "Please select a rating."

            })

        rating = int(rating)

        if rating < 1 or rating > 5:

            return jsonify({

                "success": False,

                "message":
                    "Rating must be between 1 and 5."

            })

        feedback = {

            "bus_id":
                bus_id,

            "rating":
                rating,

            "comment":
                comment
                if comment
                else
                "No written feedback.",

            "time":
                datetime.now().strftime(
                    "%d-%m-%Y %I:%M:%S %p"
                )

        }

        feedback_list.append(
            feedback
        )

        print()
        print("--------------------------------")
        print("JANyatra Feedback Received")
        print("--------------------------------")
        print(
            "Bus:",
            bus_id
        )
        print(
            "Rating:",
            rating,
            "/ 5"
        )
        print(
            "Feedback:",
            comment
            if comment
            else
            "No written feedback."
        )
        print("--------------------------------")

        return jsonify({

            "success": True,

            "message":
                "Thank you! Your feedback has been submitted.",

            "feedback":
                feedback

        })

    except Exception as error:

        print(
            "FEEDBACK ERROR:",
            error
        )

        return jsonify({

            "success": False,

            "message":
                "Feedback error: "
                + str(error)

        }), 500


# =========================================================
# GET FEEDBACK
# =========================================================

@app.route("/api/feedback", methods=["GET"])
def get_feedback():

    return jsonify({

        "success": True,

        "feedback":
            feedback_list

    })


# =========================================================
# HISTORICAL DEMAND
# =========================================================

@app.route("/api/demand-history")
def demand_history():

    route = request.args.get(
        "route",
        "Jaipur → Ajmer"
    )

    data = historical_demand.get(
        route
    )

    if data is None:

        return jsonify({

            "success": False,

            "message":
                "Route not found."

        }), 404

    return jsonify({

        "success": True,

        "route":
            route,

        "history":
            data

    })


# =========================================================
# DEMAND PREDICTION
# =========================================================

@app.route("/api/demand-prediction")
def demand_prediction():

    route = request.args.get(
        "route",
        "Jaipur → Ajmer"
    )

    data = historical_demand.get(
        route
    )

    if data is None:

        return jsonify({

            "success": False,

            "message":
                "Route not found."

        }), 404

    # -----------------------------------------------------
    # Prototype prediction:
    # Average of latest three historical values
    # -----------------------------------------------------

    values = list(
        data.values()
    )

    recent_values = values[-3:]

    predicted = sum(
        recent_values
    ) / len(
        recent_values
    )

    predicted = round(
        predicted
    )

    status = demand_status(
        predicted
    )

    if predicted <= 40:

        recommendation = (
            "Normal bus service "
            "should be sufficient."
        )

    elif predicted <= 70:

        recommendation = (
            "Moderate demand expected. "
            "Monitor passenger load."
        )

    elif predicted <= 100:

        recommendation = (
            "High demand expected. "
            "Consider additional bus service."
        )

    else:

        recommendation = (
            "Very high demand expected. "
            "Additional buses may be required."
        )

    return jsonify({

        "success": True,

        "route":
            route,

        "predicted_passengers":
            predicted,

        "demand_status":
            status,

        "recommendation":
            recommendation,

        "recent_values":
            recent_values

    })


# =========================================================
# SERVER
# =========================================================

if __name__ == "__main__":

    print()
    print("----------------------------------------")
    print("       JANyatra Smart Bus System")
    print("----------------------------------------")
    print("AI Passenger Detection : READY")
    print("Bus Occupancy          : READY")
    print("Smart Recommendation   : READY")
    print("Live Bus Location      : READY")
    print("SOS Emergency          : READY")
    print("Feedback & Rating      : READY")
    print("Demand Prediction      : READY")
    print("----------------------------------------")
    print("Open:")
    print("http://127.0.0.1:5000")
    print("----------------------------------------")
    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )

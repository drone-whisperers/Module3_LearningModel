import numpy as np

PITCH = ["ascend", "descend"]
AIRCRAFT_NAMES = ["Big Jet 345", "Airbus 380", "Cessna 420", "Being 787", "Lockheed C130", "Supermarine Spitfire"]
FLIGHT_LEVEL_MIN = 25
FLIGHT_LEVEL_MAX = 200
SPEED_RESTRICTION_MIN = 90
SPEED_RESTRICTION_MAX = 240
DESTINATIONS = ["BONNY", "CLYDE", "SUNNY", "CALI"]
ALTITUDE_MIN = 10000
ALTITUDE_MAX = 42000

def create_fly_training_example_1(aircraft_name, heading, pitch, flight_level, speed_restriction):
    if speed_restriction == 0:
        speed_cmd = "no speed restrictions"
    else:
        speed_cmd = f"speed {speed_restriction} knots"

    x = f"{aircraft_name} fly heading {heading} {pitch} to flight level {flight_level} {speed_cmd} "
    y = f"fly {heading} {pitch} {flight_level} {speed_restriction}"
    return f"{x} \t {y}"

def create_fly_training_example_2(aircraft_name, pitch, flight_level, destination):
    if pitch == "ascend":
        pitch_cmd = "climb"
    else:
        pitch_cmd = "descend"

    x = f"{aircraft_name} fly direct {destination} {pitch_cmd} to flight level {flight_level}"
    y = f"fly {destination} {pitch} {flight_level} 0"
    return f"{x} \t {y}"

def create_fly_training_example_3(aircraft_name, pitch, altitude):
    if pitch == "ascend":
        pitch_cmd = "climb"
    else:
        pitch_cmd = "descend"

    x = f"{aircraft_name} hold position after departure {pitch_cmd} to altitude {altitude} feet"
    y = f"fly 0 {pitch} {altitude}"
    return f"{x} \t {y}"

def generate_fly_training_examples(count):
    training_examples = []

    for i in range(count):
        aircraft_name = np.random.choice(AIRCRAFT_NAMES)
        heading = np.random.randint(0, 360)
        pitch = np.random.choice(PITCH)
        flight_level = np.random.randint(FLIGHT_LEVEL_MIN, FLIGHT_LEVEL_MAX)
        if (np.random.randint(0, 10)) == 0:
            speed_restriction = 0
        else:
            speed_restriction = np.random.randint(SPEED_RESTRICTION_MIN, SPEED_RESTRICTION_MAX)
        destination = np.random.choice(DESTINATIONS)
        altitude = np.random.randint(ALTITUDE_MIN, ALTITUDE_MAX)

        i = np.random.randint(0, 2)
        if i == 0:
            training_examples.append(create_fly_training_example_1(aircraft_name, heading, pitch, flight_level, speed_restriction))
        elif i == 1:
            training_examples.append(create_fly_training_example_2(aircraft_name, pitch, flight_level, destination))
        elif i == 2:
            training_examples.append(create_fly_training_example_3(aircraft_name, pitch, altitude))

    return training_examples




training_examples = generate_fly_training_examples(10000)

f = open("generatedTrainingData.txt", "a")
for i in range(len(training_examples)):
    f.write(f"{training_examples[i]}\n")

f.close()
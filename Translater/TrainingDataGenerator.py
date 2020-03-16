import numpy as np

GENERATED_TRAINING_DATA_FILE_NAME = "./TrainingData/generated.training.data.txt"
DEFAULT_FLY_TRAINING_EXAMPLES = 15000
DEFAULT_CONTACT_TRAINING_EXAMPLES = 15000

AIRCRAFT_NAMES = ["Big Jet", "Airbus", "Cessna", "Boeing", "Lockheed", "Supermarine Spitfire"]
AIRPORT_NAMES = ["LaGuardia", "Kennedy", "Pearson", "Bush", "Phoenix"]
AIRPORT_ENTITIES = ["Ground", "Tower", "Radar"]
CONTACT_REASONS = [None, "taxi", "further", "takeoff", "startup", "next", "landing", "navigation"]
DESTINATIONS = ["BONNY", "CLYDE", "SUNNY", "CALI"]
ACTIONS = ["take off", "start up", "taxi", "land", "ILS approach"]
PITCH = ["ascend", "descend"]
APPROVAL_STATUSES = ["approved", "not approved"]
FLIGHT_LEVEL_MIN = 25
FLIGHT_LEVEL_MAX = 200
SPEED_RESTRICTION_MIN = 90
SPEED_RESTRICTION_MAX = 240
ALTITUDE_MIN = 10000
ALTITUDE_MAX = 42000
CHANNEL_MIN = 116
CHANNEL_MAX = 137

class TrainingDataGenerator:
    def __create_fly_training_example_1(self, aircraft_name, heading, pitch, flight_level, speed_restriction):
        if speed_restriction == 0:
            speed_cmd = "no speed restrictions"
        else:
            speed_cmd = f"speed {speed_restriction} knots"

        x = f"{aircraft_name} fly heading {heading} {pitch} to flight level {flight_level} {speed_cmd} "
        y = f"fly {heading} {pitch} {flight_level} {speed_restriction}"
        return f"{x} \t {y}"

    def __create_fly_training_example_2(self, aircraft_name, pitch, flight_level, destination):
        if pitch == "ascend":
            pitch_cmd = "climb"
        else:
            pitch_cmd = "descend"

        x = f"{aircraft_name} fly direct {destination} {pitch_cmd} to flight level {flight_level}"
        y = f"fly {destination} {pitch} {flight_level} 0"
        return f"{x} \t {y}"

    def __create_fly_training_example_3(self, aircraft_name, pitch, altitude):
        if pitch == "ascend":
            pitch_cmd = "climb"
        else:
            pitch_cmd = "descend"

        x = f"{aircraft_name} hold position after departure {pitch_cmd} to altitude {altitude} feet"
        y = f"fly 0 {pitch} {altitude}"
        return f"{x} \t {y}"

    def __create_contact_example_1(self, aircraft_name, airport_name, airport_entity, channel, reason):
        if reason is None:
            reason_cmd = ""
            reason = ""
        else:
            reason_cmd = f"for {reason} instructions"

        x = f"{aircraft_name} contact {airport_name} {airport_entity} {channel} {reason_cmd}"
        y = f"contact {channel} {reason}"
        return f"{x} \t {y}"

    def __create_contact_and_approval_example_1(self, aircraft_name, airport_name, airport_entity, channel, reason, action, approval_status):
        if reason is None:
            reason_cmd = ""
            reason = ""
        else:
            reason_cmd = f"for {reason} instructions"

        x = f"{aircraft_name} {action} {approval_status} contact {airport_name} {airport_entity} {channel} {reason_cmd}"
        y = f"{action.replace(' ', '_')} {approval_status.replace(' ', '_')} and contact {channel} {reason}"
        return f"{x} \t {y}"

    def __generate_fly_training_examples(self, count):
        training_examples = []

        for i in range(count):
            aircraft_name = np.random.choice(AIRCRAFT_NAMES)
            aircraft_number = np.random.randint(0, 999)
            heading = np.random.randint(0, 360)
            pitch = np.random.choice(PITCH)
            flight_level = np.random.randint(FLIGHT_LEVEL_MIN, FLIGHT_LEVEL_MAX)
            if (np.random.randint(0, 10)) == 0:
                speed_restriction = 0
            else:
                speed_restriction = np.random.randint(SPEED_RESTRICTION_MIN, SPEED_RESTRICTION_MAX)
            destination = np.random.choice(DESTINATIONS)
            altitude = np.random.randint(ALTITUDE_MIN, ALTITUDE_MAX)

            j = np.random.randint(0, 2)
            if j == 0:
                training_examples.append(self.__create_fly_training_example_1(f"{aircraft_name} {aircraft_number}", heading, pitch, flight_level, speed_restriction))
            elif j == 1:
                training_examples.append(self.__create_fly_training_example_2(f"{aircraft_name} {aircraft_number}", pitch, flight_level, destination))
            #elif j == 2:
            #    training_examples.append(create_fly_training_example_3(f"{aircraft_name} {aircraft_number}", pitch, altitude))

        return training_examples

    def __generate_contact_examples(self, count):
        training_examples = []

        for i in range(count):
            aircraft_name = np.random.choice(AIRCRAFT_NAMES)
            aircraft_number = np.random.randint(0, 999)
            airport_name = np.random.choice(AIRPORT_NAMES)
            airport_entity = np.random.choice(AIRPORT_ENTITIES)
            reason = np.random.choice(CONTACT_REASONS)
            channel = np.random.randint(CHANNEL_MIN, CHANNEL_MAX) + (0.025 * np.random.randint(0, 39))
            action = np.random.choice(ACTIONS)
            approval_status = np.random.choice(APPROVAL_STATUSES)

            j = np.random.randint(0, 2)
            if j == 0:
                training_examples.append(self.__create_contact_example_1(f"{aircraft_name} {aircraft_number}", airport_name, airport_entity, channel, reason))
            else:
                training_examples.append(self.__create_contact_and_approval_example_1(aircraft_name, airport_name, airport_entity, channel, reason, action, approval_status))

        return training_examples

    def generate(self, to_file=True, filename=None, fly_training_examples=None, contact_training_examples=None):
        if fly_training_examples is None:
            training_examples = self.__generate_fly_training_examples(DEFAULT_FLY_TRAINING_EXAMPLES)
        else:
            training_examples = self.__generate_fly_training_examples(fly_training_examples)

        if contact_training_examples is None:
            training_examples = training_examples + self.__generate_contact_examples(DEFAULT_CONTACT_TRAINING_EXAMPLES)
        else:
            training_examples = training_examples + self.__generate_contact_examples(contact_training_examples)

        if to_file:
            if filename is None:
                f = open(GENERATED_TRAINING_DATA_FILE_NAME, "a")
            else:
                f = open(filename, "a")

            for i in range(len(training_examples)):
                f.write(f"{training_examples[i]}\n")

            f.close()
        else:
            return training_examples

        return
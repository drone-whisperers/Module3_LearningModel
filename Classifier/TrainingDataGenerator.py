import numpy as np
import random

GENERATED_TRAINING_DATA_FILE = "./TrainingData/generated.training.data.txt"
GENERATED_LABEL_MATRIX_FILE = "./TrainingData/generated.label.matrix.csv"
GENERATED_KNOWN_AIRCRAFT_NAME_FILE = "./TrainingData/generated.aircraft.names.txt"
GENERATED_KNOWN_AIRPORT_ENTITY_FILE = "./TrainingData/generated.airport.entities.txt"
GENERATED_KNOWN_LOCATIONS_FILE = "./TrainingData/generated.known.locations.txt"
DEFAULT_FLY_TRAINING_EXAMPLES = 1000
DEFAULT_CONTACT_TRAINING_EXAMPLES = 1000

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

        x = f"{aircraft_name} fly heading {heading} {pitch} to flight level {flight_level} {speed_cmd}"
        return f"{x}"

    # Generate a training example for a fly command to hold current position, after departure ascend/descend to altitude
    #
    # @aircraft_name - name of aircraft
    # @pitch - ascend/descend
    # @altitude - altitude to ascend/descend to
    # @neg_parameter - if specified, indicates the parameter has a value out of expected range, generates a negative training example
    def __create_fly_training_example_2(self, aircraft_name, pitch, flight_level, destination):
        if pitch == "ascend":
            pitch_cmd = "climb"
        else:
            pitch_cmd = "descend"

        x = f"{aircraft_name} fly direct {destination} {pitch_cmd} to flight level {flight_level}"
        return f"{x}"


    def __create_contact_example_1(self, aircraft_name, airport_entity, channel, reason):
        if reason is None:
            reason_cmd = ""
        else:
            reason_cmd = f"for {reason} instructions"

        x = f"{aircraft_name} contact {airport_entity} {channel} {reason_cmd}"
        return f"{x}".strip()

    # Generate a training example for a contact command
    #
    # @aircraft_name - name of aircraft
    # @airport_name - name of airport to contact
    # @airport_entity - name of entity at airport to contact
    # @channel - channel to contact airport_entity
    # @neg_parameter - if specified, indicates the parameter has a value out of expected range, generates a negative training example
    def __create_contact_example_2(self, aircraft_name, airport_entity, channel):
        x = f"{aircraft_name} contact {airport_entity} {channel}"

        return f"{x}"

    def __create_contact_and_approval_example_1(self, aircraft_name, airport_entity, channel, reason, action, approval_status):
        if reason is None:
            reason_cmd = ""
        else:
            reason_cmd = f"for {reason} instructions"

        x = f"{aircraft_name} {action} {approval_status} contact {airport_entity} {channel} {reason_cmd}"
        return f"{x}".strip()

    # Generate a training example for a contact command with an action approval/disapproval
    #
    # @aircraft_name - name of aircraft
    # @airport_name - name of airport to contact
    # @airport_entity - name of entity at airport to contact
    # @channel - channel to contact airport_entity
    # @reason - reason to contact airport_entity
    # @action - action that has been approved/not_approved
    # @approval_status - either approved or not_approved
    # @neg_parameter - if specified, indicates the parameter has a value out of expected range, generates a negative training example
    def __create_contact_and_approval_example_2(self, aircraft_name, airport_entity, channel, action, approval_status):
        x = f"{aircraft_name} {action} {approval_status} contact {airport_entity} {channel}"

        return f"{x}"

    def __generate_fly_training_examples(self, count, neg_prop=0.0):
        training_examples = {}
        parameters = ["aircraft_name", "aircraft_number", "heading", "pitch", "flight_level", "speed_restriction", "destination", "altitude"]
        aircraft_names = set()
        known_locations = set()

        #Determine the number of negative examples to generate
        neg_examples = int(count * neg_prop)
        #Select random values to use during iteration for generating negative examples
        neg_example_iteration_values = random.sample(range(count), neg_examples)

        for i in range(count):
            negative_example = False
            label=1
            if i in neg_example_iteration_values:
                negative_example = True
                label=0

            parameter_values = {}
            for parameter in parameters:
                parameter_values[parameter] = self.__generate_parameter_value(parameter)

            #Randomly select a fly training example type
            j = np.random.randint(0, 2)
            if j == 0:
                if negative_example:
                    ranged_parameters = ["heading", "flight_level", "speed_restriction"]
                    neg_parameter = np.random.choice(ranged_parameters)
                    parameter_values[neg_parameter] = self.__generate_parameter_value(neg_parameter, neg_example=True)

                training_example = self.__create_fly_training_example_1(
                    f'{parameter_values["aircraft_name"]} {parameter_values["aircraft_number"]}',
                    parameter_values["heading"],
                    parameter_values["pitch"],
                    parameter_values["flight_level"],
                    parameter_values["speed_restriction"])
            elif j == 1:
                if negative_example:
                    neg_parameter = "flight_level"
                    parameter_values[neg_parameter] = self.__generate_parameter_value(neg_parameter, neg_example=True)

                training_example = self.__create_fly_training_example_2(
                        f'{parameter_values["aircraft_name"]} {parameter_values["aircraft_number"]}',
                        parameter_values["pitch"],
                        parameter_values["flight_level"],
                        parameter_values["destination"])

            training_examples[training_example] = label
            aircraft_names.add(parameter_values["aircraft_name"])
            known_locations.add(parameter_values["destination"])
            #elif j == 2:
            #    training_examples.append(create_fly_training_example_3(f"{aircraft_name} {aircraft_number}", pitch, altitude))

        return training_examples, aircraft_names, known_locations

    def __generate_contact_examples(self, count, neg_prop=0.0):
        training_examples = {}
        parameters = ["aircraft_name", "aircraft_number", "airport_name", "airport_entity", "reason", "channel", "action", "approval_status"]
        aircraft_names = set()
        airport_entities = set()

        #Determine the number of negative examples to generate
        neg_examples = int(count * neg_prop)
        #Select random values to use during iteration for generating negative examples
        neg_example_iteration_values = random.sample(range(count), neg_examples)

        for i in range(count):
            negative_example = False
            label=1
            if i in neg_example_iteration_values:
                negative_example = True
                label=0

            parameter_values = {}
            for parameter in parameters:
                parameter_values[parameter] = self.__generate_parameter_value(parameter)

            if negative_example:
                neg_parameter = "channel"
                parameter_values[neg_parameter] = self.__generate_parameter_value(neg_parameter, neg_example=True)

            j = np.random.randint(0, 4)
            if j == 0:
                training_example = self.__create_contact_example_1(
                    f'{parameter_values["aircraft_name"]} {parameter_values["aircraft_number"]}',
                    f'{parameter_values["airport_name"]} {parameter_values["airport_entity"]}',
                    parameter_values["channel"],
                    parameter_values["reason"])
            elif j == 1:
                training_example = self.__create_contact_example_2(
                    f'{parameter_values["aircraft_name"]} {parameter_values["aircraft_number"]}',
                    f'{parameter_values["airport_name"]} {parameter_values["airport_entity"]}',
                    parameter_values["channel"])
            elif j == 2:
                training_example = self.__create_contact_and_approval_example_1(
                    f'{parameter_values["aircraft_name"]} {parameter_values["aircraft_number"]}',
                    f'{parameter_values["airport_name"]} {parameter_values["airport_entity"]}',
                    parameter_values["channel"],
                    parameter_values["reason"],
                    parameter_values["action"],
                    parameter_values["approval_status"])
            elif j == 3:
                training_example = self.__create_contact_and_approval_example_2(
                    f'{parameter_values["aircraft_name"]} {parameter_values["aircraft_number"]}',
                    f'{parameter_values["airport_name"]} {parameter_values["airport_entity"]}',
                    parameter_values["channel"],
                    parameter_values["action"],
                    parameter_values["approval_status"])

            training_examples[training_example] = label
            aircraft_names.add(parameter_values["aircraft_name"])
            airport_entities.add(f'{parameter_values["airport_name"]} {parameter_values["airport_entity"]}')

        return training_examples, aircraft_names, airport_entities

    def generate(self, to_file=True, filename=None, num_fly_train_examples=None, num_contact_train_examples=None, neg_prop=0.0):
        if num_fly_train_examples is None:
            fly_training_data, fly_aircraft_names, known_locations = self.__generate_fly_training_examples(DEFAULT_FLY_TRAINING_EXAMPLES, neg_prop)
        else:
            fly_training_data, fly_aircraft_names, known_locations = self.__generate_fly_training_examples(num_fly_train_examples, neg_prop)

        if num_contact_train_examples is None:
            contact_training_data, contact_aircraft_names, airport_entities = self.__generate_contact_examples(DEFAULT_CONTACT_TRAINING_EXAMPLES, neg_prop)
        else:
            contact_training_data, contact_aircraft_names, airport_entities = self.__generate_contact_examples(num_contact_train_examples, neg_prop)

        all_aircraft_names = fly_aircraft_names.union(contact_aircraft_names)

        if to_file:
            aircraft_name_file = open(GENERATED_KNOWN_AIRCRAFT_NAME_FILE, "w")
            for aircraft_name in all_aircraft_names:
                aircraft_name_file.write(f"{aircraft_name}\n")
            aircraft_name_file.close()

            known_location_file = open(GENERATED_KNOWN_LOCATIONS_FILE, "w")
            for known_location in known_locations:
                known_location_file.write(f"{known_location}\n")
            known_location_file.close()

            airport_entities_file = open(GENERATED_KNOWN_AIRPORT_ENTITY_FILE, "w")
            for airport_entitie in airport_entities:
                airport_entities_file.write(f"{airport_entitie}\n")
            airport_entities_file.close()

            if filename is None:
                training_data_file = open(GENERATED_TRAINING_DATA_FILE, "w")
            else:
                training_data_file = open(filename, "w")

            label_matrix_file = open(GENERATED_LABEL_MATRIX_FILE, "w")
            label_matrix_file.write("fly,contact\n")
            for example in fly_training_data.keys():
                training_data_file.write(f"{example}\n")
                label_matrix_file.write(f"{fly_training_data[example]},0\n")

            for example in contact_training_data.keys():
                training_data_file.write(f"{example}\n")
                label_matrix_file.write(f"0,{contact_training_data[example]}\n")

            training_data_file.close()
            label_matrix_file.close()
        else:
            return fly_training_data, contact_training_data

        return

    # Method to generate a random value for a parameter.
    #
    # @neg_example - If neg_example is true, then generate a parameter value that is outside of the expected range,
    #                only implemented for ranged parameters.
    # @parameter_name - a string that contains the parameter for which to generate a corresponding value.
    def __generate_parameter_value(self, parameter_name, neg_example=False):
        if parameter_name == "aircraft_name":
            parameter = np.random.choice(AIRCRAFT_NAMES)
        elif parameter_name == "aircraft_number":
            parameter = np.random.randint(0, 999)
        elif parameter_name == "heading":
            if not neg_example:
                parameter = np.random.randint(0, 360)
            else:
                parameter = np.random.randint(359, 720)
        elif parameter_name == "pitch":
            parameter = np.random.choice(PITCH)
        elif parameter_name == "flight_level":
            if not neg_example:
                parameter = np.random.randint(FLIGHT_LEVEL_MIN, FLIGHT_LEVEL_MAX)
            else:
                if np.random.randint(0,2):
                    parameter = np.random.randint(0, FLIGHT_LEVEL_MIN)
                else:
                    parameter = np.random.randint(FLIGHT_LEVEL_MAX, 2 * FLIGHT_LEVEL_MAX)
        elif parameter_name == "speed_restriction":
            if not neg_example:
                if (np.random.randint(0, 10)) == 0:
                    parameter = 0
                else:
                    parameter = np.random.randint(SPEED_RESTRICTION_MIN, SPEED_RESTRICTION_MAX)
            else:
                if np.random.randint(0, 2):
                    parameter = np.random.randint(0, SPEED_RESTRICTION_MIN)
                else:
                    parameter = np.random.randint(SPEED_RESTRICTION_MAX, SPEED_RESTRICTION_MAX * 2)
        elif parameter_name == "destination":
            parameter = np.random.choice(DESTINATIONS)
        elif parameter_name == "altitude":
            parameter = np.random.randint(ALTITUDE_MIN, ALTITUDE_MAX)
        elif parameter_name == 'channel':
            if not neg_example:
                parameter = np.random.randint(CHANNEL_MIN, CHANNEL_MAX) + (0.025 * np.random.randint(0, 39))
            else:
                if np.random.randint(0, 2):
                    parameter = np.random.randint(0, CHANNEL_MIN) + (0.025 * np.random.randint(0, 39))
                else:
                    parameter = np.random.randint(CHANNEL_MAX, 999) + (0.025 * np.random.randint(0, 39))
        elif parameter_name == 'airport_name':
            parameter = np.random.choice(AIRPORT_NAMES)
        elif parameter_name == 'airport_entity':
            parameter = np.random.choice(AIRPORT_ENTITIES)
        elif parameter_name == 'reason':
            parameter = np.random.choice(CONTACT_REASONS)
        elif parameter_name == 'action':
            parameter = np.random.choice(ACTIONS)
        elif parameter_name == 'approval_status':
            parameter = np.random.choice(APPROVAL_STATUSES)
        return parameter
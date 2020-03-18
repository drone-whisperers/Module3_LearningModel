import numpy as np
import random
import os.path

MY_PATH = os.path.abspath(os.path.dirname(__file__))
GENERATED_TRAINING_DATA_FILE_NAME = os.path.join(MY_PATH, "TrainingData/generated.training.data.txt")
DEFAULT_FLY_TRAINING_EXAMPLES = 25000
DEFAULT_CONTACT_TRAINING_EXAMPLES = 25000

AIRCRAFT_NAMES = ["big jet", "airbus", "cessna", "boeing", "lockheed", "supermarine spitfire"]
AIRPORT_NAMES = ["laguardia", "kennedy", "pearson", "bush", "phoenix"]
AIRPORT_ENTITIES = ["ground", "tower", "radar"]
CONTACT_REASONS = ["taxi", "further", "takeoff", "startup", "next", "landing", "navigation"]
DESTINATIONS = ["bonny", "clyde", "sunny", "cali"]
ACTIONS = ["take off", "start up", "taxi", "land", "ils approach"]
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

    # Generate a training example for a fly command
    #
    # Ex. airbus 396 fly heading 68 ascend to flight level 44 speed 179 knots
    #  or airbus 396 fly heading 68 ascend to flight level 44 no speed restrictions
    #
    # @aircraft_name - name of aircraft
    # @heading - heading to change to (1 - 359 degrees)
    # @pitch - ascend/descend
    # @flight_level - height to ascend/descend to (measured as a multiple of feet)
    # @speed_restriction - specifies a speed restriction, if ommitted then no speed restriction
    # @neg_parameter - if specified, indicates the parameter has a value out of expected range, generates a negative training example
    def __create_fly_training_example_1(self, aircraft_name, heading, pitch, flight_level, speed_restriction, neg_parameter=None):
        if speed_restriction == 0 and not neg_parameter == "speed_restriction":
            speed_cmd = "no speed restrictions"
        else:
            speed_cmd = f"speed {speed_restriction} knots"

        x = f"{aircraft_name} fly heading {heading} {pitch} to flight level {flight_level} {speed_cmd}"

        if neg_parameter is None:
            y = f"fly ({heading}, {pitch}, {flight_level}, {speed_restriction})"
        else:
            y = f"{neg_parameter} value out of expected range"
        return f"{x} \t {y}"

    # Generate a training example for a fly command
    #
    # Ex. airbus 975 fly direct bonny climb to flight level 163

    # @aircraft_name - name of aircraft
    # @pitch - ascend/descend
    # @altitude - altitude to ascend/descend to
    # @neg_parameter - if specified, indicates the parameter has a value out of expected range, generates a negative training example
    def __create_fly_training_example_2(self, aircraft_name, pitch, flight_level, destination, neg_parameter=None):
        if pitch == "ascend":
            pitch_cmd = "climb"
        else:
            pitch_cmd = "descend"

        x = f"{aircraft_name} fly direct {destination} {pitch_cmd} to flight level {flight_level}"
        if neg_parameter is None:
            y = f"fly ({destination}, {pitch}, {flight_level}, 0)"
        else:
            y = f"{neg_parameter} value out of expected range"
        return f"{x} \t {y}"

    # Generate a training example for a fly command
    #
    # Ex. lockheed 975 fly heading 84
    #
    # @aircraft_name - name of aircraft
    # @heading - heading to change to (1 - 359 degrees)
    # @pitch - ascend/descend
    # @flight_level - height to ascend/descend to (measured as a multiple of feet)
    # @speed_restriction - specifies a speed restriction, if ommitted then no speed restriction
    # @neg_parameter - if specified, indicates the parameter has a value out of expected range, generates a negative training example
    def __create_fly_training_example_3(self, aircraft_name, heading, neg_parameter=None):
        x = f"{aircraft_name} fly heading {heading}"

        if neg_parameter is None:
            y = f"fly ({heading}, 0, 0, 0)"
        else:
            y = f"{neg_parameter} value out of expected range"
        return f"{x} \t {y}"

    # Generate a training example for a fly command
    #
    # Ex. boeing 966 fly descend to flight level 49
    #
    # @aircraft_name - name of aircraft
    # @heading - heading to change to (1 - 359 degrees)
    # @pitch - ascend/descend
    # @flight_level - height to ascend/descend to (measured as a multiple of feet)
    # @speed_restriction - specifies a speed restriction, if ommitted then no speed restriction
    # @neg_parameter - if specified, indicates the parameter has a value out of expected range, generates a negative training example
    def __create_fly_training_example_4(self, aircraft_name, pitch, flight_level, neg_parameter=None):
        x = f"{aircraft_name} fly {pitch} to flight level {flight_level}"

        if neg_parameter is None:
            y = f"fly (0, {pitch}, {flight_level}, 0)"
        else:
            y = f"{neg_parameter} value out of expected range"
        return f"{x} \t {y}"

    # Generate a training example for a fly command
    #
    # Ex. boeing 518 fly speed 48 knots
    #
    # @aircraft_name - name of aircraft
    # @heading - heading to change to (1 - 359 degrees)
    # @pitch - ascend/descend
    # @flight_level - height to ascend/descend to (measured as a multiple of feet)
    # @speed_restriction - specifies a speed restriction, if ommitted then no speed restriction
    # @neg_parameter - if specified, indicates the parameter has a value out of expected range, generates a negative training example
    def __create_fly_training_example_5(self, aircraft_name, speed_restriction, neg_parameter=None):
        if speed_restriction == 0 and not neg_parameter == "speed_restriction":
            speed_cmd = "no speed restrictions"
        else:
            speed_cmd = f"speed {speed_restriction} knots"

        x = f"{aircraft_name} fly {speed_cmd}"

        if neg_parameter is None:
            y = f"fly (0, 0, 0, {speed_restriction})"
        else:
            y = f"{neg_parameter} value out of expected range"
        return f"{x} \t {y}"

    # Generate a training example for a fly command
    #
    # Ex. cessna 416 fly heading 519 speed 184 knots
    #
    # @aircraft_name - name of aircraft
    # @heading - heading to change to (1 - 359 degrees)
    # @pitch - ascend/descend
    # @flight_level - height to ascend/descend to (measured as a multiple of feet)
    # @speed_restriction - specifies a speed restriction, if ommitted then no speed restriction
    # @neg_parameter - if specified, indicates the parameter has a value out of expected range, generates a negative training example
    def __create_fly_training_example_6(self, aircraft_name, heading, speed_restriction, neg_parameter=None):
        if speed_restriction == 0 and not neg_parameter == "speed_restriction":
            speed_cmd = "no speed restrictions"
        else:
            speed_cmd = f"speed {speed_restriction} knots"

        x = f"{aircraft_name} fly heading {heading} {speed_cmd}"

        if neg_parameter is None:
            y = f"fly ({heading}, 0, 0, {speed_restriction})"
        else:
            y = f"{neg_parameter} value out of expected range"
        return f"{x} \t {y}"

    # Generate a training example for a fly command
    #
    # Ex. boeing 203 fly ascend to flight level 197 speed 161 knots
    #
    # @aircraft_name - name of aircraft
    # @heading - heading to change to (1 - 359 degrees)
    # @pitch - ascend/descend
    # @flight_level - height to ascend/descend to (measured as a multiple of feet)
    # @speed_restriction - specifies a speed restriction, if ommitted then no speed restriction
    # @neg_parameter - if specified, indicates the parameter has a value out of expected range, generates a negative training example
    def __create_fly_training_example_7(self, aircraft_name, pitch, flight_level, speed_restriction, neg_parameter=None):
        if speed_restriction == 0 and not neg_parameter == "speed_restriction":
            speed_cmd = "no speed restrictions"
        else:
            speed_cmd = f"speed {speed_restriction} knots"

        x = f"{aircraft_name} fly {pitch} to flight level {flight_level} {speed_cmd}"

        if neg_parameter is None:
            y = f"fly (0, {pitch}, {flight_level}, {speed_restriction})"
        else:
            y = f"{neg_parameter} value out of expected range"
        return f"{x} \t {y}"


    # Generate a training example for a contact command with a  reason
    #
    # @aircraft_name - name of aircraft
    # @airport_name - name of airport to contact
    # @airport_entity - name of entity at airport to contact
    # @channel - channel to contact airport_entity
    # @reason - reason to contact airport_entity
    # @neg_parameter - if specified, indicates the parameter has a value out of expected range, generates a negative training example
    def __create_contact_example_1(self, aircraft_name, airport_name, airport_entity, channel, reason, neg_parameter=None):
        reason_cmd = f"for {reason} instructions"

        x = f"{aircraft_name} contact {airport_name} {airport_entity} {channel} {reason_cmd}"

        if neg_parameter is None:
            y = f"contact ({channel}, {reason})"
        else:
            y = f"{neg_parameter} value out of expected range"
        return f"{x} \t {y}"

    # Generate a training example for a contact command
    #
    # @aircraft_name - name of aircraft
    # @airport_name - name of airport to contact
    # @airport_entity - name of entity at airport to contact
    # @channel - channel to contact airport_entity
    # @neg_parameter - if specified, indicates the parameter has a value out of expected range, generates a negative training example
    def __create_contact_example_2(self, aircraft_name, airport_name, airport_entity, channel, neg_parameter=None):
        x = f"{aircraft_name} contact {airport_name} {airport_entity} {channel}"

        if neg_parameter is None:
            y = f"contact ({channel}, none)"
        else:
            y = f"{neg_parameter} value out of expected range"
        return f"{x} \t {y}"

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
    def __create_contact_and_approval_example_1(self, aircraft_name, airport_name, airport_entity, channel, reason, action, approval_status, neg_parameter=None):
        reason_cmd = f"for {reason} instructions"

        x = f"{aircraft_name} {action} {approval_status} contact {airport_name} {airport_entity} {channel} {reason_cmd}"
        if neg_parameter is None:
            y = f"{approval_status.replace(' ', '_')} ({action.replace(' ', '_')}) ; contact ({channel}, {reason})"
        else:
            y = f"{neg_parameter} value out of expected range"
        return f"{x} \t {y}"

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
    def __create_contact_and_approval_example_2(self, aircraft_name, airport_name, airport_entity, channel, action, approval_status, neg_parameter=None):
        x = f"{aircraft_name} {action} {approval_status} contact {airport_name} {airport_entity} {channel}"

        if neg_parameter is None:
            y = f"{approval_status.replace(' ', '_')} ({action.replace(' ', '_')}) , contact ({channel}, none)"
        else:
            y = f"{neg_parameter} value out of expected range"

        return f"{x} \t {y}"

    # Generate training data for the fly command.
    #
    # @count - Number of training examples to generate
    # @neg_prop - proportion of negative examples to generate (parameter out of expected range). For a fly command,
    #             the ranged paramters are ["heading", "flight_level", "speed_restriction"]
    def __generate_fly_training_examples(self, count, neg_prop=0.0):
        training_examples = []
        parameters = ["aircraft_name", "aircraft_number", "heading", "pitch", "flight_level", "speed_restriction", "destination", "altitude"]

        #Select the random iteration values to generate negative examples
        neg_example_iteration_values = random.sample(range(count), int(count * neg_prop))

        for i in range(count):
            parameter_values = {}
            neg_parameter = None
            for parameter in parameters:
                parameter_values[parameter] = self.__generate_parameter_value(parameter)

            j = np.random.randint(0, 7)
            if j == 0:
                if i in neg_example_iteration_values:
                    ranged_parameters = ["heading", "flight_level", "speed_restriction"]
                    neg_parameter = np.random.choice(ranged_parameters)
                    parameter_values[neg_parameter] = self.__generate_parameter_value(neg_parameter, neg_example=True)

                training_examples.append(
                    self.__create_fly_training_example_1(
                        f"{parameter_values['aircraft_name']} {parameter_values['aircraft_number']}",
                        parameter_values['heading'],
                        parameter_values['pitch'],
                        parameter_values['flight_level'],
                        parameter_values['speed_restriction'],
                        neg_parameter or None
                    ))
            elif j == 1:
                if i in neg_example_iteration_values:
                    neg_parameter = "flight_level"
                    parameter_values[neg_parameter] = self.__generate_parameter_value(neg_parameter, neg_example=True)

                training_examples.append(
                    self.__create_fly_training_example_2(
                        f"{parameter_values['aircraft_name']} {parameter_values['aircraft_number']}",
                        parameter_values['pitch'],
                        parameter_values['flight_level'],
                        parameter_values['destination'],
                        neg_parameter or None
                    ))
            elif j == 2:
                if i in neg_example_iteration_values:
                    neg_parameter = "heading"
                    parameter_values[neg_parameter] = self.__generate_parameter_value(neg_parameter, neg_example=True)

                training_examples.append(
                    self.__create_fly_training_example_3(
                        f"{parameter_values['aircraft_name']} {parameter_values['aircraft_number']}",
                        parameter_values['heading'],
                        neg_parameter or None
                    ))
            elif j==3:
                if i in neg_example_iteration_values:
                    neg_parameter = "flight_level"
                    parameter_values[neg_parameter] = self.__generate_parameter_value(neg_parameter, neg_example=True)

                training_examples.append(
                    self.__create_fly_training_example_4(
                        f"{parameter_values['aircraft_name']} {parameter_values['aircraft_number']}",
                        parameter_values['pitch'],
                        parameter_values['flight_level'],
                        neg_parameter or None
                    ))
            elif j==4:
                if i in neg_example_iteration_values:
                    neg_parameter = "speed_restriction"
                    parameter_values[neg_parameter] = self.__generate_parameter_value(neg_parameter, neg_example=True)

                training_examples.append(
                    self.__create_fly_training_example_5(
                        f"{parameter_values['aircraft_name']} {parameter_values['aircraft_number']}",
                        parameter_values['speed_restriction'],
                        neg_parameter or None
                    ))
            elif j==5:
                if i in neg_example_iteration_values:
                    ranged_parameters = ["heading", "speed_restriction"]
                    neg_parameter = np.random.choice(ranged_parameters)
                    parameter_values[neg_parameter] = self.__generate_parameter_value(neg_parameter, neg_example=True)

                training_examples.append(
                    self.__create_fly_training_example_6(
                        f"{parameter_values['aircraft_name']} {parameter_values['aircraft_number']}",
                        parameter_values['heading'],
                        parameter_values['speed_restriction'],
                        neg_parameter or None
                    ))
            elif j==6:
                if i in neg_example_iteration_values:
                    ranged_parameters = ["flight_level", "speed_restriction"]
                    neg_parameter = np.random.choice(ranged_parameters)
                    parameter_values[neg_parameter] = self.__generate_parameter_value(neg_parameter, neg_example=True)

                training_examples.append(
                    self.__create_fly_training_example_7(
                        f"{parameter_values['aircraft_name']} {parameter_values['aircraft_number']}",
                        parameter_values['pitch'],
                        parameter_values['flight_level'],
                        parameter_values['speed_restriction'],
                        neg_parameter or None
                    ))

        return training_examples

    # Generate training data for the contact command.
    #
    # @count - Number of training examples to generate
    # @neg_prop - proportion of negative examples to generate (parameter out of expected range). For the contact command,
    #             the ranged parameters are ["channel"].
    def __generate_contact_examples(self, count, neg_prop=0.0):
        training_examples = []
        parameters = ["aircraft_name", "aircraft_number", "airport_name", "airport_entity", "reason", "channel", "action", "approval_status"]

        #Select the random iteration values to generate negative examples
        neg_example_iteration_values = random.sample(range(count), int(count * neg_prop))

        for i in range(count):
            parameter_values = {}
            neg_parameter = None

            for parameter in parameters:
                parameter_values[parameter] = self.__generate_parameter_value(parameter)

            if i in neg_example_iteration_values:
                neg_parameter = "channel"
                parameter_values[neg_parameter] = self.__generate_parameter_value(neg_parameter, neg_example=True)

            j = np.random.randint(0, 4)
            if j == 0:
                training_examples.append(
                    self.__create_contact_example_1(
                        f"{parameter_values['aircraft_name']} {parameter_values['aircraft_number']}",
                        parameter_values['airport_name'],
                        parameter_values['airport_entity'],
                        parameter_values['channel'],
                        parameter_values['reason'],
                        neg_parameter or None
                    ))
            elif j==1:
                training_examples.append(
                    self.__create_contact_example_2(
                        f"{parameter_values['aircraft_name']} {parameter_values['aircraft_number']}",
                        parameter_values['airport_name'],
                        parameter_values['airport_entity'],
                        parameter_values['channel'],
                        neg_parameter or None
                    ))
            elif j==2:
                training_examples.append(
                    self.__create_contact_and_approval_example_1(
                        parameter_values['aircraft_name'],
                        parameter_values['airport_name'],
                        parameter_values['airport_entity'],
                        parameter_values['channel'],
                        parameter_values['reason'],
                        parameter_values['action'],
                        parameter_values['approval_status'],
                        neg_parameter or None
                    ))
            elif j == 3:
                training_examples.append(
                    self.__create_contact_and_approval_example_2(
                        parameter_values['aircraft_name'],
                        parameter_values['airport_name'],
                        parameter_values['airport_entity'],
                        parameter_values['channel'],
                        parameter_values['action'],
                        parameter_values['approval_status'],
                        neg_parameter or None
                    ))
        return training_examples

    # Generate training data. Default behaviour is to output to a file, can return training data directly.
    #
    # @to_file - If to_file is true, then training data is output to a file, otherwise returns data directly.
    # @filename - Overwrite default training data output file name.
    # @num_fly_train_examples - number of training examples to generate for fly commands.
    # @num_contact_train_examples - number of training examples to generate for contact commands.
    # @neg_prop - proportion of negative examples to generate (parameter out of expected range)
    def generate(self, to_file=True, filename=None, num_fly_train_examples=None, num_contact_train_examples=None, neg_prop=0.0):
        if num_fly_train_examples is None:
            training_examples = self.__generate_fly_training_examples(DEFAULT_FLY_TRAINING_EXAMPLES, neg_prop)
        else:
            training_examples = self.__generate_fly_training_examples(num_fly_train_examples, neg_prop)

        if num_contact_train_examples is None:
            training_examples = training_examples + self.__generate_contact_examples(DEFAULT_CONTACT_TRAINING_EXAMPLES, neg_prop)
        else:
            training_examples = training_examples + self.__generate_contact_examples(num_contact_train_examples, neg_prop)

        if to_file:
            if filename is None:
                f = open(GENERATED_TRAINING_DATA_FILE_NAME, "w")
            else:
                f = open(filename, "w")

            for i in range(len(training_examples)):
                f.write(f"{training_examples[i]}\n")

            f.close()
        else:
            return training_examples

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
                parameter = np.random.randint(1, 360)
            else:
                parameter = np.random.randint(360, 720)
        elif parameter_name == "pitch":
            parameter = np.random.choice(PITCH)
        elif parameter_name == "flight_level":
            if not neg_example:
                parameter = np.random.randint(FLIGHT_LEVEL_MIN, FLIGHT_LEVEL_MAX + 1)
            else:
                if np.random.randint(0,2):
                    parameter = np.random.randint(0, FLIGHT_LEVEL_MIN)
                else:
                    parameter = np.random.randint(FLIGHT_LEVEL_MAX + 1, 2 * FLIGHT_LEVEL_MAX)
        elif parameter_name == "speed_restriction":
            if not neg_example:
                if (np.random.randint(0, 10)) == 0:
                    parameter = 0
                else:
                    parameter = np.random.randint(SPEED_RESTRICTION_MIN, SPEED_RESTRICTION_MAX + 1)
            else:
                if np.random.randint(0, 2):
                    parameter = np.random.randint(0, SPEED_RESTRICTION_MIN)
                else:
                    parameter = np.random.randint(SPEED_RESTRICTION_MAX + 1, SPEED_RESTRICTION_MAX * 2)
        elif parameter_name == "destination":
            parameter = np.random.choice(DESTINATIONS)
        elif parameter_name == "altitude":
            parameter = np.random.randint(ALTITUDE_MIN, ALTITUDE_MAX)
        elif parameter_name == 'channel':
            if not neg_example:
                parameter = np.random.randint(CHANNEL_MIN, CHANNEL_MAX + 1) + (0.025 * np.random.randint(0, 39))
            else:
                if np.random.randint(0, 2):
                    parameter = np.random.randint(0, CHANNEL_MIN) + (0.025 * np.random.randint(0, 39))
                else:
                    parameter = np.random.randint(CHANNEL_MAX + 1, 999) + (0.025 * np.random.randint(0, 39))
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
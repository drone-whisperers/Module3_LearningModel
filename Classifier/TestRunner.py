from Classifier import Classifier
from TrainingDataGenerator import TrainingDataGenerator
import os

MY_PATH = os.path.abspath(os.path.dirname(__file__))
GENERATED_TEST_DATA_FILE_NAME = os.path.join(MY_PATH, "TrainingData/generated.test.data.txt")

classifier = Classifier()
training_data_generator = TrainingDataGenerator()

#Delete any previously saved test data files, only keep the most recent
if os.path.isfile(GENERATED_TEST_DATA_FILE_NAME):
    os.remove(GENERATED_TEST_DATA_FILE_NAME)

#Generate the fly and contact command examples
fly_cmd_examples, contact_cmd_examples = training_data_generator.generate(to_file=True, filename=GENERATED_TEST_DATA_FILE_NAME, num_fly_train_examples=500, num_contact_train_examples=500)

#Put both collections of examples in a labelled map, the label is used for verification against predictions
cmd_examples = {'fly':fly_cmd_examples, 'contact':contact_cmd_examples}

cmd_class_test_results = {}
for cmd_class in cmd_examples.keys():
    correct_translation = 0
    incorrect_translation = 0
    incorrect_translations = {}
    examples = cmd_examples[cmd_class]

    for example in examples:
        classification = classifier.classify_command(example, print_translation=False)

        if cmd_class in classification:
            correct_translation += 1
        else:
            incorrect_translation += 1
            incorrect_translations[example] = {"expected_translation": cmd_class, "actual_translation": classification}

    cmd_class_test_results[cmd_class] = {'correct_translations':correct_translation, 'incorrect_translation':incorrect_translation, 'incorrect_translations':incorrect_translations}

print(f"Fly Commands: {cmd_class_test_results['fly']}")
print(f"Contact Commands: {cmd_class_test_results['contact']}")





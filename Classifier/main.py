from Classifier import Classifier
from TrainingDataGenerator import TrainingDataGenerator

classifier = Classifier()
classifier.init_classifier()

training_data_generator = TrainingDataGenerator()
fly_cmd_examples, contact_cmd_examples = training_data_generator.generate(to_file=False, num_fly_train_examples=500, num_contact_train_examples=500)
cmd_examples = {'fly':fly_cmd_examples, 'contact':contact_cmd_examples}

cmd_class_test_results = {}
for cmd_class in cmd_examples.keys():
    correct_translation = 0
    incorrect_translation = 0
    incorrect_translations = {}
    examples = cmd_examples[cmd_class]

    for example in examples:
        classification = classifier.classify_sentence(example, print_translation=False)

        #Caseless comparison and strip leading/trailing whitespace
        if cmd_class in classification:
            correct_translation += 1
        else:
            incorrect_translation += 1
            incorrect_translations[example] = {"expected_translation": cmd_class, "actual_translation": classification}

    cmd_class_test_results[cmd_class] = {'correct_translations':correct_translation, 'incorrect_translation':incorrect_translation, 'incorrect_translations':incorrect_translations}

print(cmd_class_test_results)
print("Hello World")




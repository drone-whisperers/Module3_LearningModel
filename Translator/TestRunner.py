from Translator import Translator
from TrainingDataGenerator import TrainingDataGenerator

translator = Translator()

#Generate new data for testing
trainingDataGenerator = TrainingDataGenerator()
fly_cmd_examples = trainingDataGenerator.generate(to_file=False, num_fly_train_examples=5000, num_contact_train_examples=0, neg_prop=0.1)
contact_cmd_examples = trainingDataGenerator.generate(to_file=False, num_fly_train_examples=0, num_contact_train_examples=5000, neg_prop=0.1)
cmd_examples = {'fly':fly_cmd_examples, 'contact':contact_cmd_examples}

cmd_class_test_results = {}
for cmd_class in cmd_examples.keys():
    correct_translation = 0
    incorrect_translation = 0
    incorrect_translations = {}
    examples = cmd_examples[cmd_class]
    for example in examples:
        cmd, expected_translation = example.split("\t")
        actual_translation = translator.translate_command(cmd, print_translation=False)

        #Caseless comparison and strip leading/trailing whitespace
        if actual_translation.lower().strip() == expected_translation.lower().strip():
            correct_translation += 1
        else:
            incorrect_translation += 1
            incorrect_translations[cmd] = {"expected_translation": expected_translation, "actual_translation": actual_translation}

    cmd_class_test_results[cmd_class] = {'correct_translations':correct_translation, 'incorrect_translation':incorrect_translation, 'incorrect_translations':incorrect_translations}

print(f"Fly Commands: {cmd_class_test_results['fly']}")
print(f"Contact Commands: {cmd_class_test_results['contact']}")

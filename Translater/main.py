from Translater import Translater
from TrainingDataGenerator import TrainingDataGenerator

translater = Translater()
translater.init_translater()

#Generate new data for testing
trainingDataGenerator = TrainingDataGenerator()
fly_cmd_examples = trainingDataGenerator.generate(to_file=False, fly_training_examples=1000, contact_training_examples=0)
contact_cmd_examples = trainingDataGenerator.generate(to_file=False, fly_training_examples=0, contact_training_examples=1000)
cmd_examples = {'fly':fly_cmd_examples, 'contact':contact_cmd_examples}

cmd_class_test_results = {}
for cmd_class in cmd_examples.keys():
    correct_translation = 0
    incorrect_translation = 0
    incorrect_translations = {}
    examples = cmd_examples[cmd_class]
    for example in examples:
        cmd, expected_translation = example.split("\t")
        actual_translation = translater.translate_sentence(cmd, print_translation=False)

        #Caseless comparison and strip leading/trailing whitespace
        if actual_translation.lower().strip() == expected_translation.lower().strip():
            correct_translation += 1
        else:
            incorrect_translation += 1
            incorrect_translations[expected_translation] = actual_translation

    cmd_class_test_results[cmd_class] = {'correct_translations':correct_translation, 'incorrect_translation':incorrect_translation, 'incorrect_translations':incorrect_translations}

print(f"Fly Commands: {cmd_class_test_results['fly']}")
print(f"Contact Commands: {cmd_class_test_results['contact']}")

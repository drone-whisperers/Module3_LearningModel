# Module3_LearningModel

Environment Requirements:
  - python 3.7
  - need the following packages installed (pip3 install...)
    - tensorflow
    - keras
    - pandas
    - scikit-learn
    
Before you run...
  Classifier
  - What resources are required to run the Classifier?
    - Trained models (located in Module3_LearningModel/Classifier/Model):
      - contact.command.classifier.pickle
      - fly.command.classifier.pickle
    - Encoder (located in Module3_LearningModel/Classifier/Model)
      - encoder.pickle
    - Training Data (located in Module3_LearningModel/Classifier/TrainingData)
      - various generated.x.x.x files
        - If these do not exist, by instantiating the Classifier, it will create these by:
          - generating training data using Module3_LearningModel/Classifier/TrainingDataGenerator.py (output as generated.x.x.x)
          - creating an encoder based on the generated training data

  Translator
  - What resources are required to run the Translator?
    - Trained models (located in Module3_LearningModel/Translator/Model):
      - encoder.model
      - decoder.model
      - note: you will notice "best.weights.val.acc.h5" and "best.weights.val.loss.h5", these are artifacts of training the  model and used to save a snapshot of the most optimal model that is obtained during training
        - this is relevant because when you specify x epochs to train a model for, the default behaviour is to keep the model in the state from the last epoch, this is not always the best model
    - Training Data (located in Module3_LearningModel/Translator/TrainingData)
      - generated.training.data.txt
        - If this does not exist, by instantiating the Translator, it will create these by:
          - generating training data using Module3_LearningModel/Translator/TrainingDataGenerator.py (output as generated.training.data.txt)
    - The glove word embeddings file
      - this is used in the encoding of input for the LSTM networks that are implemented in the Translator.
      - this can be downloaded using http://nlp.stanford.edu/data/glove.6B.zip
        - extract this zip and copy 'globe.6B.100d.txt' to 'Module3_LearningModel/Translator/TrainingData'
        
How to run...
  - If you simply want to validate everything is set up properly, run the TestRunner.py script for both the Classifier and Translator
    - This script will:
      - instantiate the model (which ensures the training data is supplied if it does not exist)
      - generate a set of test data (output as generated.test.data.txt in the TrainingData folder)
      - run the model for each examply in the test data and output a very simply summary
  - If you'd like to run the application such that you can query the system to translate commands from the command line
    - run Module3_LearningModel/Server/run_server.py
    - run Module3_LearningModel/Server/run_client.py
    - from the run_client.py terminal, send ATC commands, these will be sent to the server instance, which will query the system
      - The system is currently configured to bypass the Classifier and sends the input directly to the Translator for a translation
        - part of the fun will be getting the classifier to work as intended to gate nonsensical input as expected!
  - If you want to train a brand new model, delete all of the files mentioned above and simply instantiate either a Classifier or Translatore
    - translator = Translator() or classifier = Classifier()
  

    

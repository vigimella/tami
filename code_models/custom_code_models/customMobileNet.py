from tensorflow.keras.layers import Dense, Flatten, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import Adam
import tensorflow


class MobNetCustom:

    def __init__(self, num_classes, img_size, channels, weights='imagenet', learning_rate=0.01, name="CustomMobileNet",
                 include_top=False):
        self.name = name
        self.learning_rate = learning_rate
        self.weights = weights
        self.include_top = include_top
        self.num_classes = num_classes
        self.input_width_height = img_size
        self.channels = channels
        self.input_type = 'images'

    def build(self):

        base_model = None
        output = None

        if self.include_top:
            if self.input_width_height != 224 or self.channels != 3:
                print("IF include_top=True, input_shape MUST be (224,224,3), exiting...")
                exit()
            else:
                if self.name == "CustomMobileNet":
                    MobileNetV3 = tensorflow.keras.applications.MobileNetV3Small(weights=self.weights, include_top=False, classes=self.num_classes)
                else:
                    print("Invalid name, accepted 'CustomMobileNet', exiting...")
                    exit()
                output = MobileNetV3.output
        else:
            inputs = Input(shape=(self.input_width_height, self.input_width_height, self.channels))
            if self.name == "CustomMobileNet":
                MobileNetV3 = tensorflow.keras.applications.MobileNetV3Small(weights=self.weights, include_top=False, classes=self.num_classes)
            else:
                print("Invalid name, accepted 'CustomMobileNet', exiting...")
                exit()
            flatten = Flatten(name='my_flatten')
            f_dropout = Dropout(0.2)
            output_layer = Dense(self.num_classes, activation='softmax', name='my_predictions')
            s_dropout = Dropout(0.2)
            output = output_layer(flatten(MobileNetV3.output))

        input_layer = MobileNetV3.input

        model = Model(input_layer, output)
        # model.summary(line_length=50)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(self.learning_rate),
                      metrics=['acc', Precision(name="prec"), Recall(name="rec"), AUC(name='auc')])
        return model

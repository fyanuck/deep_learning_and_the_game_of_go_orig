from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.agent.predict import DeepLearningAgent
from dlgo.networks.alphago import alphago_model

from keras.callbacks import ModelCheckpoint
import h5py

from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Conv2D

rows, cols = 19, 19
num_classes = rows * cols
num_games = 10

encoder = AlphaGoEncoder()
processor = GoDataProcessor(encoder=encoder.name())
generator = processor.load_go_data('train', num_games, use_generator=True)
test_generator = processor.load_go_data('test', num_games, use_generator=True)

input_shape = (encoder.num_planes, rows, cols)

# Модель на 5 (или 6) слоёв вместо 12 (или 13)
num_filters=192,
first_kernel_size=5,
other_kernel_size=3
alphago_sl_policy = Sequential()
alphago_sl_policy.add(
         Conv2D(num_filters, first_kernel_size, input_shape=input_shape, padding='same',
               data_format='channels_first', activation='relu'))
for i in range(2, 5):  # <4>
        alphago_sl_policy.add(
            Conv2D(num_filters, other_kernel_size, padding='same',
                   data_format='channels_first', activation='relu'))
alphago_sl_policy.add(
        Conv2D(filters=1, kernel_size=1, padding='same',
            data_format='channels_first', activation='softmax'))
alphago_sl_policy.add(Flatten())
    
# alphago_sl_policy = alphago_model(input_shape, is_policy_net=True)

alphago_sl_policy.compile('sgd', 'categorical_crossentropy', metrics=['accuracy'])
# end::alphago_sl_model[]

# tag::alphago_sl_train[]
epochs = 5
batch_size = 128
alphago_sl_policy.fit_generator(
    generator=generator.generate(batch_size, num_classes),
    epochs=epochs,
    steps_per_epoch=generator.get_num_samples() / batch_size,
    validation_data=test_generator.generate(batch_size, num_classes),
    validation_steps=test_generator.get_num_samples() / batch_size,
    callbacks=[ModelCheckpoint('alphago_sl_policy_{epoch}.h5')]
)

alphago_sl_agent = DeepLearningAgent(alphago_sl_policy, encoder)

with h5py.File('alphago_sl_policy.h5', 'w') as sl_agent_out:
    alphago_sl_agent.serialize(sl_agent_out)
# end::alphago_sl_train[]

alphago_sl_policy.evaluate_generator(
    generator=test_generator.generate(batch_size, num_classes),
    steps=test_generator.get_num_samples() / batch_size
)
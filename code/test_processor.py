from dlgo.data.processor import GoDataProcessor
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.agent.predict import DeepLearningAgent
from dlgo.networks.alphago import alphago_model   # strong policy network
from dlgo.networks import medium

from keras.callbacks import ModelCheckpoint
import h5py

print('This is test_processor.py!')

rows, cols = 19, 19
num_classes = rows * cols
num_games = 100

encoder = AlphaGoEncoder()
processor = GoDataProcessor(encoder=encoder.name())
# generator = processor.load_go_data('train', num_games, use_generator=True)
generator = processor.load_go_data('train', num_games)
test_generator = processor.load_go_data('test', num_games, use_generator=True)

input_shape = (encoder.num_planes, rows, cols)
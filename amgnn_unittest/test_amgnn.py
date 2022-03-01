import wandb
from unittest import TestCase
from model.amgnn import model, NeuralNetwork
from utils.config import read_config
from pathlib import Path


class Testmodel(TestCase):
    # Initialise wandb
    test_configuration = read_config(Path("amgnn_unittest") / "test_configs")
    wandb.init(mode="offline", config=test_configuration)

    def test_build(self):
        m = model()
        m.build()
        self.assertIsInstance(m.network, NeuralNetwork)

    def test_train(self):
        self.fail()

    def test_validate(self):
        self.fail()

    def test_predict(self):
        self.fail()

    def test_load_data(self):
        # model.load_data()
        self.fail()

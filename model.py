from config import Config
from utils import get_regularized_model

def create_model(): 
  return get_regularized_model(
    input_shape=Config.input_shape,
    num_classes=Config.num_classes,
    dropout_rate=Config.dropout_rate,
    l2_regularization=Config.l2_regularization
  )

if __name__ == "__main__":
    model = create_model()
    model.summary()

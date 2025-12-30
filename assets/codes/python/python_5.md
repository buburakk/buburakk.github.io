# Time Series Regression with PyTorch

```python
from IPython.display import Image, display
from matplotlib.animation import PillowWriter
from matplotlib import animation
from matplotlib import pyplot as plt
from numpy import pi as PI
import matplotlib as mpl
import numpy as np
import torch
```

```python
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Matplotlib version: {mpl.__version__}")
```

    PyTorch version: 2.9.0+cpu
    NumPy version: 2.0.2
    Matplotlib version: 3.10.0


```python
mpl.rcParams["figure.figsize"] = (8, 6)
mpl.rcParams["figure.dpi"] = 90
mpl.rcParams["axes.spines.top"] = True
mpl.rcParams["axes.spines.right"] = True
mpl.rcParams["axes.spines.bottom"] = True
mpl.rcParams["axes.spines.left"] = True
mpl.rcParams["axes.grid"] = True
mpl.rcParams["axes.titlesize"] = 16
mpl.rcParams["axes.linewidth"] = 1.25
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["xtick.minor.size"] = 4
mpl.rcParams["xtick.minor.width"] = 1
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["xtick.major.width"] = 1.25
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.minor.size"] = 4
mpl.rcParams["ytick.minor.width"] = 1
mpl.rcParams["ytick.major.size"] = 8
mpl.rcParams["ytick.major.width"] = 1.25
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["legend.fontsize"] = 14
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "Liberation Serif"
mpl.rcParams["animation.html"] = "jshtml" # For animations
```

```python
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  torch.set_default_device(device) # Creates all tensors on the same device
  print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
  device = torch.device("cpu")
  torch.set_default_device(device) # Creates all tensors on the same device
  print("No GPU available!")
```

    No GPU available!


```python
class CreateDataset(torch.utils.data.Dataset):
  def __init__(self, tensor_list):
    # Initialize a dataset
    assert isinstance(tensor_list, list), "'tensor_list' must be of 'list' type!"
    assert isinstance(tensor_list[0], torch.Tensor), "The first element of 'tensor_list' must be of 'torch.Tensor' type!"
    assert isinstance(tensor_list[1], torch.Tensor), "The second element of 'tensor_list' must be of 'torch.Tensor' type!"
    self.x = tensor_list[0]
    self.y = tensor_list[1]
    self.length = self.x.shape[0]

  def __len__(self):
    # Get the number of elements in entire dataset
    return self.length

  def __getitem__(self, index):
    return self.x[index], self.y[index]


def split_dataset(dataset, split_ratios=None, seed=7):
  if split_ratios is not None:
    generator_seed = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, split_ratios, generator_seed)
    return train_dataset, validation_dataset, test_dataset
  return dataset


def level(size_n, base_level=0.0):
  return np.full((size_n,), base_level)


def trend(time_steps, m=1.0, n=0.0):
  return m * time_steps + n


def seasonality(time_steps, func, period=None, amplitude=1.0, phase_shift=0.0, vertical_shift=0.0):
  if period is not None:
    time_steps = (time_steps + phase_shift) % period
    return amplitude * func(time_steps) + vertical_shift
  return amplitude * func(time_steps + phase_shift) + vertical_shift


def noise(size_n, noise_level=1.0, mu=0.0, sigma=1.0, seed_n=666):
  np.random.seed(seed_n)
  return noise_level * np.random.normal(mu, sigma, size=(size_n,))


def pattern(time_steps):
  return 0.1 * time_steps * np.cos(time_steps)
```

```python
class FCNN(torch.nn.Module):
  def __init__(self, model_name, input_size, output_size, hiddens, weights, biases, batchnorms, activations, dropouts):
    # Initialize a custom fully-connected neural network
    super(FCNN, self).__init__()
    assert isinstance(model_name, str), \
    "'model_name' must be of 'str' type!"
    assert len(hiddens) + 1 == len(weights), \
    "Number of hidden layers must match the number of 'weights' units/tensors!"
    assert len(hiddens) + 1 == len(biases), \
    "Number of hidden layers must match the number of 'bias' units/scalars!"
    assert len(hiddens) == len(batchnorms), \
    "Number of hidden layers must match the number of 'batch normalization' units!"
    assert len(hiddens) == len(activations), \
    "Number of hidden layers must match the number of 'activation' functions!"
    assert len(hiddens) == len(dropouts), \
    "Number of hidden layers must match the number of 'dropout' units!"
    self._epoch_number = None
    self._training_loss_array = None
    self._validation_loss_array = None
    self._test_loss_array = None
    self._predictions_array = None
    self._epoch_data_dict = dict()
    self._model_name = model_name
    self._output_size = output_size
    self._weights = weights
    self._biases = biases
    self._batchnorms = batchnorms
    self._activations = activations
    self._dropouts = dropouts
    self._layers_size = [input_size]
    self._layers_size.extend(hiddens)
    self._layers = torch.nn.ModuleList()
    # Build a model with given specifications
    for index in range(len(self._layers_size) - 1):
      layer = torch.nn.Linear(self._layers_size[index], self._layers_size[index + 1])
      if self._weights[index]:
        self._weights[index](layer.weight)
      if self._biases[index]:
        self._biases[index](layer.bias)
      self._layers.append(layer)
      if self._batchnorms[index]:
        self._layers.append(torch.nn.BatchNorm1d(self._layers_size[index + 1]))
      if self._dropouts[index]:
        self._layers.append(torch.nn.Dropout(self._dropouts[index]))
      self._layers.append(self._activations[index]())
    self._layers.append(torch.nn.Linear(self._layers_size[-1], self._output_size))

  def _training_loss(self, training_loader, criterion, optimizer):
    # Training loop for a given model
    self.train()
    training_loss = 0.0
    for x_train, y_train in training_loader:
      y_hat_train = self.forward(x_train)
      train_loss = criterion(y_hat_train, y_train)
      train_loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      training_loss += train_loss.item()
    # Calculate the average training loss
    training_loss /= len(training_loader)
    return training_loss

  def _validation_loss(self, validation_loader, criterion):
    # Validation loop for a given model
    self.eval()
    validation_loss = 0.0
    with torch.no_grad():
      for x_val, y_val in validation_loader:
        y_hat_val = self.forward(x_val)
        val_loss = criterion(y_hat_val, y_val)
        validation_loss += val_loss.item()
    # Calculate the average validation loss
    validation_loss /= len(validation_loader)
    return validation_loss

  def _test_loss(self, test_loader, criterion):
    # Test loop for a given model
    self.eval()
    test_loss = 0.0
    with torch.no_grad():
      for x_test, y_test in test_loader:
        y_hat_test = self.forward(x_test)
        test_loss = criterion(y_hat_test, y_test)
        test_loss += test_loss.item()
    # Calculate the average test loss
    test_loss /= len(test_loader)
    return test_loss

  def _plot_loss(self, epoch, x_padding, y_padding, title_padding, figure_size=(8, 6), figure_dpi=90, line_width=3):
    fig, ax = plt.subplots(figsize=figure_size, dpi=figure_dpi)
    ax.plot(range(1, epoch + 1), self._training_loss_array[:epoch + 1, :], "r-", linewidth=line_width, label="Training loss")
    if self._validation_loss_array is not None:
      ax.plot(range(1, epoch + 1), self._validation_loss_array[:epoch + 1, :], "g-", linewidth=line_width, label="Validation loss")
    if self._test_loss_array is not None:
      ax.plot(range(1, epoch + 1), self._test_loss_array[:epoch + 1, :], "b-", linewidth=line_width, label="Test loss")
    ax.set_xlim(left=0)
    ax.set_xlabel("Epoch", labelpad=x_padding)
    ax.set_ylabel("Loss", labelpad=y_padding, rotation="horizontal")
    ax.set_yscale("log")
    ax.set_title("Loss vs. Epoch", pad=title_padding)
    ax.grid()
    ax.legend(title=self._model_name, title_fontsize=16,
              loc="best", edgecolor="black",
              fancybox=False, shadow=True, borderaxespad=1)
    plt.tight_layout()
    plt.show()

  def _plot_prediction(self, x, y, epoch, x_padding, y_padding, title_padding, figure_size=(8, 6), figure_dpi=90, marker_size=3, line_width=3):
    fig, ax = plt.subplots(figsize=figure_size, dpi=figure_dpi)
    ax.plot(x.numpy(), y.numpy(), "ro", markersize=marker_size, label="Data")
    ax.plot(x.numpy(), self._predictions_array[epoch - 1, :, :], "b-", linewidth=line_width, label="Prediction")
    ax.set_xlabel(r"$x$", labelpad=x_padding)
    ax.set_ylabel(r"$y$", labelpad=y_padding, rotation="horizontal")
    ax.set_title("Data vs. Prediction", pad=title_padding)
    ax.grid()
    ax.legend(title=self._model_name, title_fontsize=16,
              loc="best", edgecolor="black",
              fancybox=False, shadow=True, borderaxespad=1)
    plt.tight_layout()
    plt.show()

  def get_summary(self):
    # Get model information
    print(f"'Model structure': {self._get_name()}\n")
    for name, parameter in self.named_parameters():
      print(f"'Layer name': {name} - 'Layer size': {tuple(parameter.size())}\n")

  def forward(self, x):
    # Forward pass for a given input
    for layer in self._layers:
      x = layer(x)
    return x

  def trainer(self, criterion, optimizer, epochs, training_loader, validation_loader=None, test_loader=None, x=None, each_epoch=False):
    self._epoch_number = epochs
    self._training_loss_array = np.empty(shape=(self._epoch_number, 1))
    if validation_loader is not None:
      self._validation_loss_array = np.empty(shape=(self._epoch_number, 1))
    if test_loader is not None:
      self._test_loss_array = np.empty(shape=(self._epoch_number, 1))
    if x is not None:
      self._predictions_array = np.empty(shape=(self._epoch_number, x.shape[0], self._output_size))
    for epoch in range(1, self._epoch_number + 1):
      loss_info_string = f"Epoch number: [{epoch} / {self._epoch_number}] - "
      training_loss = self._training_loss(training_loader, criterion, optimizer)
      self._training_loss_array[epoch - 1, :] = training_loss
      loss_info_string += f"Training error/loss: {training_loss:.6e}"
      if validation_loader is not None:
        validation_loss = self._validation_loss(validation_loader, criterion)
        self._validation_loss_array[epoch - 1, :] = validation_loss
        loss_info_string += f" - Validation error/loss: {validation_loss:.6e}"
      if test_loader is not None:
        test_loss = self._test_loss(test_loader, criterion)
        self._test_loss_array[epoch - 1, :] = test_loss
        loss_info_string += f" - Test error/loss: {test_loss:.6e}"
      if x is not None:
        epoch_prediction = self.predictor(x)
        self._predictions_array[epoch - 1, :, :] = epoch_prediction.numpy()
      print(loss_info_string)
      if each_epoch:
        self._plot_loss(epoch)
    self._epoch_data_dict["losses"] = dict()
    self._epoch_data_dict["losses"]["training"] = self._training_loss_array
    self._epoch_data_dict["losses"]["validation"] = self._validation_loss_array
    self._epoch_data_dict["losses"]["test"] = self._test_loss_array
    self._epoch_data_dict["predictions"] = self._predictions_array
    return self._epoch_data_dict

  def predictor(self, x):
    # Predict after training for a given model
    self.eval()
    with torch.no_grad():
      x = self.forward(x)
    return x

  def get_loss_plot(self, x_padding, y_padding, title_padding, figure_size=(8, 6), figure_dpi=90):
    self._plot_loss(self._epoch_number, x_padding, y_padding, title_padding, figure_size=figure_size, figure_dpi=figure_dpi)

  def get_prediction_plot(self, x, y, x_padding, y_padding, title_padding, figure_size=(8, 6), figure_dpi=90):
    self._plot_prediction(x, y, self._epoch_number, x_padding, y_padding, title_padding, figure_size=figure_size, figure_dpi=figure_dpi)
```

```python
SEED = 7
DATASIZE = 500
EPOCH = 500
STEP = 10
```

# Problem #1

```python
x1 = np.linspace(-10, 10, DATASIZE)
y1 = pattern(x1) + noise(DATASIZE, 0.05, 1.0, 2.0, SEED)

x1 = torch.from_numpy(x1).to(torch.float32).reshape((DATASIZE, 1))
y1 = torch.from_numpy(y1).to(torch.float32).reshape((DATASIZE, 1))

ds_1 = CreateDataset([x1, y1])
ds_loader_1 = torch.utils.data.DataLoader(ds_1, batch_size=32, shuffle=True)

print(f"'x1' data: {tuple(x1.shape)}",
      f"'y1' data: {tuple(y1.shape)}",
      sep="\n\n")
```

    'x1' data: (500, 1)
    
    'y1' data: (500, 1)


```python
torch.manual_seed(SEED)

def set_weight(weights):
  return torch.nn.init.xavier_uniform_(weights)


def set_bias(biases):
  return torch.nn.init.zeros_(biases)


def set_activation():
  return torch.nn.ReLU()


model_1 = FCNN("model_1", 1, 1, [64, 64], 3 * [set_weight], 3 * [set_bias], 2 * [False], 2 * [set_activation], 2 * [False])

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model_1.parameters())

model_1.get_summary()
```

    'Model structure': FCNN
    
    'Layer name': _layers.0.weight - 'Layer size': (64, 1)
    
    'Layer name': _layers.0.bias - 'Layer size': (64,)
    
    'Layer name': _layers.2.weight - 'Layer size': (64, 64)
    
    'Layer name': _layers.2.bias - 'Layer size': (64,)
    
    'Layer name': _layers.4.weight - 'Layer size': (1, 64)
    
    'Layer name': _layers.4.bias - 'Layer size': (1,)
    


```python
print(*model_1.parameters(), sep="\n\n")
```

    Parameter containing:
    tensor([[ 0.1241],
            [ 0.1202],
            [-0.1879],
            [-0.0947],
            [-0.2840],
            [ 0.1633],
            [ 0.2627],
            [ 0.0994],
            [-0.1880],
            [ 0.1603],
            [ 0.2521],
            [-0.0622],
            [-0.0498],
            [ 0.1052],
            [ 0.0391],
            [-0.1004],
            [-0.2821],
            [-0.2130],
            [ 0.0165],
            [-0.0163],
            [ 0.2509],
            [ 0.0104],
            [-0.0782],
            [-0.2969],
            [-0.2793],
            [ 0.1262],
            [ 0.2706],
            [-0.0046],
            [-0.2716],
            [ 0.1079],
            [ 0.2475],
            [-0.2011],
            [-0.0800],
            [ 0.0256],
            [-0.1733],
            [-0.2817],
            [ 0.1330],
            [-0.2493],
            [-0.0763],
            [-0.1613],
            [ 0.2285],
            [ 0.0587],
            [ 0.1409],
            [-0.2057],
            [ 0.2560],
            [ 0.0271],
            [-0.1685],
            [-0.0199],
            [-0.2151],
            [-0.0140],
            [ 0.1697],
            [-0.1366],
            [ 0.2568],
            [ 0.0928],
            [-0.0532],
            [ 0.0949],
            [-0.0560],
            [-0.0706],
            [-0.1993],
            [ 0.1770],
            [ 0.1824],
            [ 0.1040],
            [ 0.1797],
            [ 0.3035]], requires_grad=True)
    
    Parameter containing:
    tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           requires_grad=True)
    
    Parameter containing:
    tensor([[ 0.1086,  0.1079,  0.1447,  ..., -0.0754,  0.0668,  0.0211],
            [ 0.0175, -0.1901,  0.0325,  ...,  0.1395,  0.0915,  0.2050],
            [-0.0075, -0.1651,  0.0024,  ..., -0.0792,  0.0365, -0.0140],
            ...,
            [-0.1062,  0.1632, -0.2158,  ...,  0.0649,  0.1436, -0.0524],
            [ 0.0419, -0.0125, -0.1059,  ...,  0.0913, -0.0647,  0.1942],
            [ 0.1136,  0.0144, -0.0577,  ...,  0.0840, -0.1260,  0.0895]],
           requires_grad=True)
    
    Parameter containing:
    tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           requires_grad=True)
    
    Parameter containing:
    tensor([[-0.0446, -0.0026, -0.1054,  0.0096,  0.0254, -0.0886, -0.0745, -0.1141,
             -0.0805,  0.0287,  0.0142,  0.0023, -0.0504, -0.0342,  0.0716,  0.1024,
             -0.1212, -0.0362, -0.0054, -0.0358, -0.1028, -0.1174, -0.0329,  0.0999,
              0.0005,  0.0276,  0.0168, -0.0369,  0.0915, -0.0400, -0.0980, -0.1194,
              0.0973,  0.1082,  0.0269, -0.1165,  0.0537, -0.1077,  0.0440,  0.0954,
              0.0343,  0.0860, -0.0398, -0.0898, -0.0995,  0.0852,  0.1061,  0.0706,
              0.0998,  0.0195, -0.0999, -0.0177, -0.0283, -0.0607, -0.0675,  0.1095,
              0.0691,  0.1035,  0.0693, -0.0351, -0.0461,  0.0361, -0.1248,  0.0597]],
           requires_grad=True)
    
    Parameter containing:
    tensor([0.0903], requires_grad=True)


```python
history_1 = model_1.trainer(criterion, optimizer, EPOCH, training_loader=ds_loader_1, x=x1)
```

    Epoch number: [1 / 500] - Training error/loss: 1.924362e-01
    Epoch number: [2 / 500] - Training error/loss: 1.868809e-01
    Epoch number: [3 / 500] - Training error/loss: 1.870384e-01
    Epoch number: [4 / 500] - Training error/loss: 1.855559e-01
    Epoch number: [5 / 500] - Training error/loss: 1.830311e-01
    Epoch number: [6 / 500] - Training error/loss: 1.809341e-01
    Epoch number: [7 / 500] - Training error/loss: 1.828153e-01
    Epoch number: [8 / 500] - Training error/loss: 1.823456e-01
    Epoch number: [9 / 500] - Training error/loss: 1.813765e-01
    Epoch number: [10 / 500] - Training error/loss: 1.782762e-01
    Epoch number: [11 / 500] - Training error/loss: 1.822846e-01
    Epoch number: [12 / 500] - Training error/loss: 1.820278e-01
    Epoch number: [13 / 500] - Training error/loss: 1.810666e-01
    Epoch number: [14 / 500] - Training error/loss: 1.812315e-01
    Epoch number: [15 / 500] - Training error/loss: 1.822809e-01
    Epoch number: [16 / 500] - Training error/loss: 1.776678e-01
    Epoch number: [17 / 500] - Training error/loss: 1.751117e-01
    Epoch number: [18 / 500] - Training error/loss: 1.819030e-01
    Epoch number: [19 / 500] - Training error/loss: 1.802780e-01
    Epoch number: [20 / 500] - Training error/loss: 1.783305e-01
    Epoch number: [21 / 500] - Training error/loss: 1.763279e-01
    Epoch number: [22 / 500] - Training error/loss: 1.798556e-01
    Epoch number: [23 / 500] - Training error/loss: 1.763251e-01
    Epoch number: [24 / 500] - Training error/loss: 1.784959e-01
    Epoch number: [25 / 500] - Training error/loss: 1.834880e-01
    Epoch number: [26 / 500] - Training error/loss: 1.746162e-01
    Epoch number: [27 / 500] - Training error/loss: 1.710437e-01
    Epoch number: [28 / 500] - Training error/loss: 1.738584e-01
    Epoch number: [29 / 500] - Training error/loss: 1.737272e-01
    Epoch number: [30 / 500] - Training error/loss: 1.690527e-01
    Epoch number: [31 / 500] - Training error/loss: 1.711113e-01
    Epoch number: [32 / 500] - Training error/loss: 1.741120e-01
    Epoch number: [33 / 500] - Training error/loss: 1.687132e-01
    Epoch number: [34 / 500] - Training error/loss: 1.648987e-01
    Epoch number: [35 / 500] - Training error/loss: 1.596595e-01
    Epoch number: [36 / 500] - Training error/loss: 1.598570e-01
    Epoch number: [37 / 500] - Training error/loss: 1.545532e-01
    Epoch number: [38 / 500] - Training error/loss: 1.528327e-01
    Epoch number: [39 / 500] - Training error/loss: 1.529963e-01
    Epoch number: [40 / 500] - Training error/loss: 1.464519e-01
    Epoch number: [41 / 500] - Training error/loss: 1.424019e-01
    Epoch number: [42 / 500] - Training error/loss: 1.416512e-01
    Epoch number: [43 / 500] - Training error/loss: 1.355640e-01
    Epoch number: [44 / 500] - Training error/loss: 1.308721e-01
    Epoch number: [45 / 500] - Training error/loss: 1.260099e-01
    Epoch number: [46 / 500] - Training error/loss: 1.219294e-01
    Epoch number: [47 / 500] - Training error/loss: 1.173226e-01
    Epoch number: [48 / 500] - Training error/loss: 1.085919e-01
    Epoch number: [49 / 500] - Training error/loss: 9.876544e-02
    Epoch number: [50 / 500] - Training error/loss: 9.271034e-02
    Epoch number: [51 / 500] - Training error/loss: 8.722509e-02
    Epoch number: [52 / 500] - Training error/loss: 8.697208e-02
    Epoch number: [53 / 500] - Training error/loss: 7.965059e-02
    Epoch number: [54 / 500] - Training error/loss: 7.090824e-02
    Epoch number: [55 / 500] - Training error/loss: 6.546269e-02
    Epoch number: [56 / 500] - Training error/loss: 5.843074e-02
    Epoch number: [57 / 500] - Training error/loss: 5.629589e-02
    Epoch number: [58 / 500] - Training error/loss: 4.926013e-02
    Epoch number: [59 / 500] - Training error/loss: 4.721285e-02
    Epoch number: [60 / 500] - Training error/loss: 4.217337e-02
    Epoch number: [61 / 500] - Training error/loss: 4.133642e-02
    Epoch number: [62 / 500] - Training error/loss: 3.949868e-02
    Epoch number: [63 / 500] - Training error/loss: 3.152538e-02
    Epoch number: [64 / 500] - Training error/loss: 3.138415e-02
    Epoch number: [65 / 500] - Training error/loss: 2.768438e-02
    Epoch number: [66 / 500] - Training error/loss: 2.535244e-02
    Epoch number: [67 / 500] - Training error/loss: 2.392487e-02
    Epoch number: [68 / 500] - Training error/loss: 2.228531e-02
    Epoch number: [69 / 500] - Training error/loss: 2.097305e-02
    Epoch number: [70 / 500] - Training error/loss: 2.025279e-02
    Epoch number: [71 / 500] - Training error/loss: 2.334073e-02
    Epoch number: [72 / 500] - Training error/loss: 1.890667e-02
    Epoch number: [73 / 500] - Training error/loss: 1.734166e-02
    Epoch number: [74 / 500] - Training error/loss: 1.685531e-02
    Epoch number: [75 / 500] - Training error/loss: 1.707965e-02
    Epoch number: [76 / 500] - Training error/loss: 1.744741e-02
    Epoch number: [77 / 500] - Training error/loss: 1.759307e-02
    Epoch number: [78 / 500] - Training error/loss: 1.703263e-02
    Epoch number: [79 / 500] - Training error/loss: 1.680807e-02
    Epoch number: [80 / 500] - Training error/loss: 1.704687e-02
    Epoch number: [81 / 500] - Training error/loss: 1.499294e-02
    Epoch number: [82 / 500] - Training error/loss: 1.631077e-02
    Epoch number: [83 / 500] - Training error/loss: 1.465833e-02
    Epoch number: [84 / 500] - Training error/loss: 1.567140e-02
    Epoch number: [85 / 500] - Training error/loss: 1.400114e-02
    Epoch number: [86 / 500] - Training error/loss: 1.425650e-02
    Epoch number: [87 / 500] - Training error/loss: 1.388464e-02
    Epoch number: [88 / 500] - Training error/loss: 1.489464e-02
    Epoch number: [89 / 500] - Training error/loss: 1.516789e-02
    Epoch number: [90 / 500] - Training error/loss: 1.435348e-02
    Epoch number: [91 / 500] - Training error/loss: 1.503136e-02
    Epoch number: [92 / 500] - Training error/loss: 1.464830e-02
    Epoch number: [93 / 500] - Training error/loss: 1.419786e-02
    Epoch number: [94 / 500] - Training error/loss: 1.371645e-02
    Epoch number: [95 / 500] - Training error/loss: 1.491373e-02
    Epoch number: [96 / 500] - Training error/loss: 1.676372e-02
    Epoch number: [97 / 500] - Training error/loss: 1.399158e-02
    Epoch number: [98 / 500] - Training error/loss: 1.387991e-02
    Epoch number: [99 / 500] - Training error/loss: 1.547366e-02
    Epoch number: [100 / 500] - Training error/loss: 1.385535e-02
    Epoch number: [101 / 500] - Training error/loss: 1.370253e-02
    Epoch number: [102 / 500] - Training error/loss: 1.410731e-02
    Epoch number: [103 / 500] - Training error/loss: 1.340075e-02
    Epoch number: [104 / 500] - Training error/loss: 1.375729e-02
    Epoch number: [105 / 500] - Training error/loss: 1.445577e-02
    Epoch number: [106 / 500] - Training error/loss: 1.431580e-02
    Epoch number: [107 / 500] - Training error/loss: 1.396726e-02
    Epoch number: [108 / 500] - Training error/loss: 1.603924e-02
    Epoch number: [109 / 500] - Training error/loss: 1.440712e-02
    Epoch number: [110 / 500] - Training error/loss: 1.459979e-02
    Epoch number: [111 / 500] - Training error/loss: 1.668325e-02
    Epoch number: [112 / 500] - Training error/loss: 1.846349e-02
    Epoch number: [113 / 500] - Training error/loss: 1.690279e-02
    Epoch number: [114 / 500] - Training error/loss: 1.362475e-02
    Epoch number: [115 / 500] - Training error/loss: 1.345646e-02
    Epoch number: [116 / 500] - Training error/loss: 1.374974e-02
    Epoch number: [117 / 500] - Training error/loss: 1.430541e-02
    Epoch number: [118 / 500] - Training error/loss: 1.578044e-02
    Epoch number: [119 / 500] - Training error/loss: 1.434608e-02
    Epoch number: [120 / 500] - Training error/loss: 1.338916e-02
    Epoch number: [121 / 500] - Training error/loss: 1.420573e-02
    Epoch number: [122 / 500] - Training error/loss: 1.426197e-02
    Epoch number: [123 / 500] - Training error/loss: 1.293091e-02
    Epoch number: [124 / 500] - Training error/loss: 1.404649e-02
    Epoch number: [125 / 500] - Training error/loss: 1.440381e-02
    Epoch number: [126 / 500] - Training error/loss: 1.367690e-02
    Epoch number: [127 / 500] - Training error/loss: 1.414773e-02
    Epoch number: [128 / 500] - Training error/loss: 1.411429e-02
    Epoch number: [129 / 500] - Training error/loss: 1.364457e-02
    Epoch number: [130 / 500] - Training error/loss: 1.388572e-02
    Epoch number: [131 / 500] - Training error/loss: 1.307571e-02
    Epoch number: [132 / 500] - Training error/loss: 1.286522e-02
    Epoch number: [133 / 500] - Training error/loss: 1.332783e-02
    Epoch number: [134 / 500] - Training error/loss: 1.475282e-02
    Epoch number: [135 / 500] - Training error/loss: 1.279732e-02
    Epoch number: [136 / 500] - Training error/loss: 1.289749e-02
    Epoch number: [137 / 500] - Training error/loss: 1.283748e-02
    Epoch number: [138 / 500] - Training error/loss: 1.345250e-02
    Epoch number: [139 / 500] - Training error/loss: 1.363409e-02
    Epoch number: [140 / 500] - Training error/loss: 1.387267e-02
    Epoch number: [141 / 500] - Training error/loss: 1.364036e-02
    Epoch number: [142 / 500] - Training error/loss: 1.280542e-02
    Epoch number: [143 / 500] - Training error/loss: 1.371127e-02
    Epoch number: [144 / 500] - Training error/loss: 1.259807e-02
    Epoch number: [145 / 500] - Training error/loss: 1.447544e-02
    Epoch number: [146 / 500] - Training error/loss: 1.389803e-02
    Epoch number: [147 / 500] - Training error/loss: 1.448960e-02
    Epoch number: [148 / 500] - Training error/loss: 1.247607e-02
    Epoch number: [149 / 500] - Training error/loss: 1.243515e-02
    Epoch number: [150 / 500] - Training error/loss: 1.231652e-02
    Epoch number: [151 / 500] - Training error/loss: 1.208399e-02
    Epoch number: [152 / 500] - Training error/loss: 1.348281e-02
    Epoch number: [153 / 500] - Training error/loss: 1.372378e-02
    Epoch number: [154 / 500] - Training error/loss: 1.521284e-02
    Epoch number: [155 / 500] - Training error/loss: 1.521102e-02
    Epoch number: [156 / 500] - Training error/loss: 1.639814e-02
    Epoch number: [157 / 500] - Training error/loss: 1.266934e-02
    Epoch number: [158 / 500] - Training error/loss: 1.363807e-02
    Epoch number: [159 / 500] - Training error/loss: 1.445205e-02
    Epoch number: [160 / 500] - Training error/loss: 1.344962e-02
    Epoch number: [161 / 500] - Training error/loss: 1.256364e-02
    Epoch number: [162 / 500] - Training error/loss: 1.357063e-02
    Epoch number: [163 / 500] - Training error/loss: 1.312385e-02
    Epoch number: [164 / 500] - Training error/loss: 1.294239e-02
    Epoch number: [165 / 500] - Training error/loss: 1.298909e-02
    Epoch number: [166 / 500] - Training error/loss: 1.212962e-02
    Epoch number: [167 / 500] - Training error/loss: 1.189294e-02
    Epoch number: [168 / 500] - Training error/loss: 1.365924e-02
    Epoch number: [169 / 500] - Training error/loss: 1.255396e-02
    Epoch number: [170 / 500] - Training error/loss: 1.335012e-02
    Epoch number: [171 / 500] - Training error/loss: 1.381582e-02
    Epoch number: [172 / 500] - Training error/loss: 1.222875e-02
    Epoch number: [173 / 500] - Training error/loss: 1.260941e-02
    Epoch number: [174 / 500] - Training error/loss: 1.262180e-02
    Epoch number: [175 / 500] - Training error/loss: 1.247269e-02
    Epoch number: [176 / 500] - Training error/loss: 1.353841e-02
    Epoch number: [177 / 500] - Training error/loss: 1.285955e-02
    Epoch number: [178 / 500] - Training error/loss: 1.249268e-02
    Epoch number: [179 / 500] - Training error/loss: 1.393851e-02
    Epoch number: [180 / 500] - Training error/loss: 1.347699e-02
    Epoch number: [181 / 500] - Training error/loss: 1.409947e-02
    Epoch number: [182 / 500] - Training error/loss: 1.372918e-02
    Epoch number: [183 / 500] - Training error/loss: 1.551862e-02
    Epoch number: [184 / 500] - Training error/loss: 1.454837e-02
    Epoch number: [185 / 500] - Training error/loss: 1.264190e-02
    Epoch number: [186 / 500] - Training error/loss: 1.258616e-02
    Epoch number: [187 / 500] - Training error/loss: 1.309815e-02
    Epoch number: [188 / 500] - Training error/loss: 1.392391e-02
    Epoch number: [189 / 500] - Training error/loss: 1.400711e-02
    Epoch number: [190 / 500] - Training error/loss: 1.236910e-02
    Epoch number: [191 / 500] - Training error/loss: 1.319049e-02
    Epoch number: [192 / 500] - Training error/loss: 1.313514e-02
    Epoch number: [193 / 500] - Training error/loss: 1.269836e-02
    Epoch number: [194 / 500] - Training error/loss: 1.253246e-02
    Epoch number: [195 / 500] - Training error/loss: 1.375320e-02
    Epoch number: [196 / 500] - Training error/loss: 1.614478e-02
    Epoch number: [197 / 500] - Training error/loss: 1.529745e-02
    Epoch number: [198 / 500] - Training error/loss: 1.409330e-02
    Epoch number: [199 / 500] - Training error/loss: 1.323337e-02
    Epoch number: [200 / 500] - Training error/loss: 1.233608e-02
    Epoch number: [201 / 500] - Training error/loss: 1.272739e-02
    Epoch number: [202 / 500] - Training error/loss: 1.233682e-02
    Epoch number: [203 / 500] - Training error/loss: 1.190797e-02
    Epoch number: [204 / 500] - Training error/loss: 1.269128e-02
    Epoch number: [205 / 500] - Training error/loss: 1.320339e-02
    Epoch number: [206 / 500] - Training error/loss: 1.239403e-02
    Epoch number: [207 / 500] - Training error/loss: 1.230710e-02
    Epoch number: [208 / 500] - Training error/loss: 1.353394e-02
    Epoch number: [209 / 500] - Training error/loss: 1.229883e-02
    Epoch number: [210 / 500] - Training error/loss: 1.210877e-02
    Epoch number: [211 / 500] - Training error/loss: 1.320526e-02
    Epoch number: [212 / 500] - Training error/loss: 1.468561e-02
    Epoch number: [213 / 500] - Training error/loss: 1.313768e-02
    Epoch number: [214 / 500] - Training error/loss: 1.432484e-02
    Epoch number: [215 / 500] - Training error/loss: 1.379420e-02
    Epoch number: [216 / 500] - Training error/loss: 1.299084e-02
    Epoch number: [217 / 500] - Training error/loss: 1.235224e-02
    Epoch number: [218 / 500] - Training error/loss: 1.373758e-02
    Epoch number: [219 / 500] - Training error/loss: 1.235347e-02
    Epoch number: [220 / 500] - Training error/loss: 1.189203e-02
    Epoch number: [221 / 500] - Training error/loss: 1.365636e-02
    Epoch number: [222 / 500] - Training error/loss: 1.515991e-02
    Epoch number: [223 / 500] - Training error/loss: 1.198538e-02
    Epoch number: [224 / 500] - Training error/loss: 1.318909e-02
    Epoch number: [225 / 500] - Training error/loss: 1.237298e-02
    Epoch number: [226 / 500] - Training error/loss: 1.170654e-02
    Epoch number: [227 / 500] - Training error/loss: 1.133279e-02
    Epoch number: [228 / 500] - Training error/loss: 1.177013e-02
    Epoch number: [229 / 500] - Training error/loss: 1.229640e-02
    Epoch number: [230 / 500] - Training error/loss: 1.243780e-02
    Epoch number: [231 / 500] - Training error/loss: 1.171138e-02
    Epoch number: [232 / 500] - Training error/loss: 1.211117e-02
    Epoch number: [233 / 500] - Training error/loss: 1.187857e-02
    Epoch number: [234 / 500] - Training error/loss: 1.247283e-02
    Epoch number: [235 / 500] - Training error/loss: 1.158453e-02
    Epoch number: [236 / 500] - Training error/loss: 1.180417e-02
    Epoch number: [237 / 500] - Training error/loss: 1.173916e-02
    Epoch number: [238 / 500] - Training error/loss: 1.466625e-02
    Epoch number: [239 / 500] - Training error/loss: 1.425666e-02
    Epoch number: [240 / 500] - Training error/loss: 1.189627e-02
    Epoch number: [241 / 500] - Training error/loss: 1.291031e-02
    Epoch number: [242 / 500] - Training error/loss: 1.266165e-02
    Epoch number: [243 / 500] - Training error/loss: 1.170021e-02
    Epoch number: [244 / 500] - Training error/loss: 1.225691e-02
    Epoch number: [245 / 500] - Training error/loss: 1.212902e-02
    Epoch number: [246 / 500] - Training error/loss: 1.145702e-02
    Epoch number: [247 / 500] - Training error/loss: 1.134012e-02
    Epoch number: [248 / 500] - Training error/loss: 1.176216e-02
    Epoch number: [249 / 500] - Training error/loss: 1.117608e-02
    Epoch number: [250 / 500] - Training error/loss: 1.216802e-02
    Epoch number: [251 / 500] - Training error/loss: 1.387591e-02
    Epoch number: [252 / 500] - Training error/loss: 1.410601e-02
    Epoch number: [253 / 500] - Training error/loss: 1.225448e-02
    Epoch number: [254 / 500] - Training error/loss: 1.268728e-02
    Epoch number: [255 / 500] - Training error/loss: 1.188859e-02
    Epoch number: [256 / 500] - Training error/loss: 1.188947e-02
    Epoch number: [257 / 500] - Training error/loss: 1.153248e-02
    Epoch number: [258 / 500] - Training error/loss: 1.138354e-02
    Epoch number: [259 / 500] - Training error/loss: 1.169592e-02
    Epoch number: [260 / 500] - Training error/loss: 1.165211e-02
    Epoch number: [261 / 500] - Training error/loss: 1.190658e-02
    Epoch number: [262 / 500] - Training error/loss: 1.168643e-02
    Epoch number: [263 / 500] - Training error/loss: 1.198896e-02
    Epoch number: [264 / 500] - Training error/loss: 1.242310e-02
    Epoch number: [265 / 500] - Training error/loss: 1.243116e-02
    Epoch number: [266 / 500] - Training error/loss: 1.154807e-02
    Epoch number: [267 / 500] - Training error/loss: 1.164925e-02
    Epoch number: [268 / 500] - Training error/loss: 1.225104e-02
    Epoch number: [269 / 500] - Training error/loss: 1.234348e-02
    Epoch number: [270 / 500] - Training error/loss: 1.244462e-02
    Epoch number: [271 / 500] - Training error/loss: 1.136528e-02
    Epoch number: [272 / 500] - Training error/loss: 1.168478e-02
    Epoch number: [273 / 500] - Training error/loss: 1.128560e-02
    Epoch number: [274 / 500] - Training error/loss: 1.160637e-02
    Epoch number: [275 / 500] - Training error/loss: 1.097080e-02
    Epoch number: [276 / 500] - Training error/loss: 1.250693e-02
    Epoch number: [277 / 500] - Training error/loss: 1.219782e-02
    Epoch number: [278 / 500] - Training error/loss: 1.262545e-02
    Epoch number: [279 / 500] - Training error/loss: 1.180975e-02
    Epoch number: [280 / 500] - Training error/loss: 1.104793e-02
    Epoch number: [281 / 500] - Training error/loss: 1.160328e-02
    Epoch number: [282 / 500] - Training error/loss: 1.143032e-02
    Epoch number: [283 / 500] - Training error/loss: 1.148824e-02
    Epoch number: [284 / 500] - Training error/loss: 1.126155e-02
    Epoch number: [285 / 500] - Training error/loss: 1.247409e-02
    Epoch number: [286 / 500] - Training error/loss: 1.171191e-02
    Epoch number: [287 / 500] - Training error/loss: 1.225177e-02
    Epoch number: [288 / 500] - Training error/loss: 1.283848e-02
    Epoch number: [289 / 500] - Training error/loss: 1.389999e-02
    Epoch number: [290 / 500] - Training error/loss: 1.378103e-02
    Epoch number: [291 / 500] - Training error/loss: 1.151987e-02
    Epoch number: [292 / 500] - Training error/loss: 1.254756e-02
    Epoch number: [293 / 500] - Training error/loss: 1.184688e-02
    Epoch number: [294 / 500] - Training error/loss: 1.041057e-02
    Epoch number: [295 / 500] - Training error/loss: 1.222134e-02
    Epoch number: [296 / 500] - Training error/loss: 1.220486e-02
    Epoch number: [297 / 500] - Training error/loss: 1.141259e-02
    Epoch number: [298 / 500] - Training error/loss: 1.157308e-02
    Epoch number: [299 / 500] - Training error/loss: 1.226562e-02
    Epoch number: [300 / 500] - Training error/loss: 1.215542e-02
    Epoch number: [301 / 500] - Training error/loss: 1.156072e-02
    Epoch number: [302 / 500] - Training error/loss: 1.143800e-02
    Epoch number: [303 / 500] - Training error/loss: 1.155642e-02
    Epoch number: [304 / 500] - Training error/loss: 1.267260e-02
    Epoch number: [305 / 500] - Training error/loss: 1.263688e-02
    Epoch number: [306 / 500] - Training error/loss: 1.173410e-02
    Epoch number: [307 / 500] - Training error/loss: 1.207188e-02
    Epoch number: [308 / 500] - Training error/loss: 1.161162e-02
    Epoch number: [309 / 500] - Training error/loss: 1.270101e-02
    Epoch number: [310 / 500] - Training error/loss: 1.161172e-02
    Epoch number: [311 / 500] - Training error/loss: 1.106482e-02
    Epoch number: [312 / 500] - Training error/loss: 1.151313e-02
    Epoch number: [313 / 500] - Training error/loss: 1.266449e-02
    Epoch number: [314 / 500] - Training error/loss: 1.110808e-02
    Epoch number: [315 / 500] - Training error/loss: 1.163675e-02
    Epoch number: [316 / 500] - Training error/loss: 1.146141e-02
    Epoch number: [317 / 500] - Training error/loss: 1.080787e-02
    Epoch number: [318 / 500] - Training error/loss: 1.157406e-02
    Epoch number: [319 / 500] - Training error/loss: 1.081401e-02
    Epoch number: [320 / 500] - Training error/loss: 1.156695e-02
    Epoch number: [321 / 500] - Training error/loss: 1.161698e-02
    Epoch number: [322 / 500] - Training error/loss: 1.274659e-02
    Epoch number: [323 / 500] - Training error/loss: 1.322542e-02
    Epoch number: [324 / 500] - Training error/loss: 1.207377e-02
    Epoch number: [325 / 500] - Training error/loss: 1.085151e-02
    Epoch number: [326 / 500] - Training error/loss: 1.107943e-02
    Epoch number: [327 / 500] - Training error/loss: 1.159762e-02
    Epoch number: [328 / 500] - Training error/loss: 1.163030e-02
    Epoch number: [329 / 500] - Training error/loss: 1.181719e-02
    Epoch number: [330 / 500] - Training error/loss: 1.176558e-02
    Epoch number: [331 / 500] - Training error/loss: 1.278651e-02
    Epoch number: [332 / 500] - Training error/loss: 1.102351e-02
    Epoch number: [333 / 500] - Training error/loss: 1.077067e-02
    Epoch number: [334 / 500] - Training error/loss: 1.083937e-02
    Epoch number: [335 / 500] - Training error/loss: 1.081312e-02
    Epoch number: [336 / 500] - Training error/loss: 1.160176e-02
    Epoch number: [337 / 500] - Training error/loss: 1.191605e-02
    Epoch number: [338 / 500] - Training error/loss: 1.094522e-02
    Epoch number: [339 / 500] - Training error/loss: 1.121743e-02
    Epoch number: [340 / 500] - Training error/loss: 1.131193e-02
    Epoch number: [341 / 500] - Training error/loss: 1.077028e-02
    Epoch number: [342 / 500] - Training error/loss: 1.197210e-02
    Epoch number: [343 / 500] - Training error/loss: 1.137432e-02
    Epoch number: [344 / 500] - Training error/loss: 1.216969e-02
    Epoch number: [345 / 500] - Training error/loss: 1.070880e-02
    Epoch number: [346 / 500] - Training error/loss: 1.089478e-02
    Epoch number: [347 / 500] - Training error/loss: 1.162626e-02
    Epoch number: [348 / 500] - Training error/loss: 1.092782e-02
    Epoch number: [349 / 500] - Training error/loss: 1.105237e-02
    Epoch number: [350 / 500] - Training error/loss: 1.079419e-02
    Epoch number: [351 / 500] - Training error/loss: 1.122704e-02
    Epoch number: [352 / 500] - Training error/loss: 1.059140e-02
    Epoch number: [353 / 500] - Training error/loss: 1.087166e-02
    Epoch number: [354 / 500] - Training error/loss: 1.207216e-02
    Epoch number: [355 / 500] - Training error/loss: 1.211617e-02
    Epoch number: [356 / 500] - Training error/loss: 1.214302e-02
    Epoch number: [357 / 500] - Training error/loss: 1.091517e-02
    Epoch number: [358 / 500] - Training error/loss: 1.186461e-02
    Epoch number: [359 / 500] - Training error/loss: 1.115993e-02
    Epoch number: [360 / 500] - Training error/loss: 1.161312e-02
    Epoch number: [361 / 500] - Training error/loss: 1.124713e-02
    Epoch number: [362 / 500] - Training error/loss: 1.181147e-02
    Epoch number: [363 / 500] - Training error/loss: 1.120291e-02
    Epoch number: [364 / 500] - Training error/loss: 1.108081e-02
    Epoch number: [365 / 500] - Training error/loss: 1.179022e-02
    Epoch number: [366 / 500] - Training error/loss: 1.299096e-02
    Epoch number: [367 / 500] - Training error/loss: 1.103589e-02
    Epoch number: [368 / 500] - Training error/loss: 1.071431e-02
    Epoch number: [369 / 500] - Training error/loss: 1.113578e-02
    Epoch number: [370 / 500] - Training error/loss: 1.063243e-02
    Epoch number: [371 / 500] - Training error/loss: 1.038702e-02
    Epoch number: [372 / 500] - Training error/loss: 1.058817e-02
    Epoch number: [373 / 500] - Training error/loss: 1.077582e-02
    Epoch number: [374 / 500] - Training error/loss: 1.056378e-02
    Epoch number: [375 / 500] - Training error/loss: 1.055113e-02
    Epoch number: [376 / 500] - Training error/loss: 1.103782e-02
    Epoch number: [377 / 500] - Training error/loss: 1.132542e-02
    Epoch number: [378 / 500] - Training error/loss: 1.224043e-02
    Epoch number: [379 / 500] - Training error/loss: 1.132971e-02
    Epoch number: [380 / 500] - Training error/loss: 1.147979e-02
    Epoch number: [381 / 500] - Training error/loss: 1.178598e-02
    Epoch number: [382 / 500] - Training error/loss: 1.263413e-02
    Epoch number: [383 / 500] - Training error/loss: 1.475518e-02
    Epoch number: [384 / 500] - Training error/loss: 1.238094e-02
    Epoch number: [385 / 500] - Training error/loss: 1.097872e-02
    Epoch number: [386 / 500] - Training error/loss: 1.064888e-02
    Epoch number: [387 / 500] - Training error/loss: 1.088837e-02
    Epoch number: [388 / 500] - Training error/loss: 1.192452e-02
    Epoch number: [389 / 500] - Training error/loss: 1.104497e-02
    Epoch number: [390 / 500] - Training error/loss: 1.100849e-02
    Epoch number: [391 / 500] - Training error/loss: 1.317352e-02
    Epoch number: [392 / 500] - Training error/loss: 1.150596e-02
    Epoch number: [393 / 500] - Training error/loss: 1.107219e-02
    Epoch number: [394 / 500] - Training error/loss: 1.160546e-02
    Epoch number: [395 / 500] - Training error/loss: 1.075867e-02
    Epoch number: [396 / 500] - Training error/loss: 1.055551e-02
    Epoch number: [397 / 500] - Training error/loss: 1.080373e-02
    Epoch number: [398 / 500] - Training error/loss: 1.192894e-02
    Epoch number: [399 / 500] - Training error/loss: 1.086032e-02
    Epoch number: [400 / 500] - Training error/loss: 1.173494e-02
    Epoch number: [401 / 500] - Training error/loss: 1.148641e-02
    Epoch number: [402 / 500] - Training error/loss: 1.125392e-02
    Epoch number: [403 / 500] - Training error/loss: 1.066779e-02
    Epoch number: [404 / 500] - Training error/loss: 1.292774e-02
    Epoch number: [405 / 500] - Training error/loss: 1.101648e-02
    Epoch number: [406 / 500] - Training error/loss: 1.090027e-02
    Epoch number: [407 / 500] - Training error/loss: 1.042637e-02
    Epoch number: [408 / 500] - Training error/loss: 1.120747e-02
    Epoch number: [409 / 500] - Training error/loss: 1.108934e-02
    Epoch number: [410 / 500] - Training error/loss: 1.151278e-02
    Epoch number: [411 / 500] - Training error/loss: 1.139615e-02
    Epoch number: [412 / 500] - Training error/loss: 1.169912e-02
    Epoch number: [413 / 500] - Training error/loss: 1.107944e-02
    Epoch number: [414 / 500] - Training error/loss: 1.124935e-02
    Epoch number: [415 / 500] - Training error/loss: 1.186205e-02
    Epoch number: [416 / 500] - Training error/loss: 1.120446e-02
    Epoch number: [417 / 500] - Training error/loss: 1.112145e-02
    Epoch number: [418 / 500] - Training error/loss: 1.089051e-02
    Epoch number: [419 / 500] - Training error/loss: 1.076342e-02
    Epoch number: [420 / 500] - Training error/loss: 1.126175e-02
    Epoch number: [421 / 500] - Training error/loss: 1.149528e-02
    Epoch number: [422 / 500] - Training error/loss: 1.081340e-02
    Epoch number: [423 / 500] - Training error/loss: 9.902584e-03
    Epoch number: [424 / 500] - Training error/loss: 1.058649e-02
    Epoch number: [425 / 500] - Training error/loss: 1.212359e-02
    Epoch number: [426 / 500] - Training error/loss: 1.229831e-02
    Epoch number: [427 / 500] - Training error/loss: 1.222834e-02
    Epoch number: [428 / 500] - Training error/loss: 1.218267e-02
    Epoch number: [429 / 500] - Training error/loss: 1.073961e-02
    Epoch number: [430 / 500] - Training error/loss: 1.225280e-02
    Epoch number: [431 / 500] - Training error/loss: 1.103556e-02
    Epoch number: [432 / 500] - Training error/loss: 1.083013e-02
    Epoch number: [433 / 500] - Training error/loss: 1.189215e-02
    Epoch number: [434 / 500] - Training error/loss: 1.178755e-02
    Epoch number: [435 / 500] - Training error/loss: 1.163123e-02
    Epoch number: [436 / 500] - Training error/loss: 1.052749e-02
    Epoch number: [437 / 500] - Training error/loss: 1.066163e-02
    Epoch number: [438 / 500] - Training error/loss: 1.223170e-02
    Epoch number: [439 / 500] - Training error/loss: 1.249766e-02
    Epoch number: [440 / 500] - Training error/loss: 1.093399e-02
    Epoch number: [441 / 500] - Training error/loss: 1.073420e-02
    Epoch number: [442 / 500] - Training error/loss: 1.088882e-02
    Epoch number: [443 / 500] - Training error/loss: 1.155460e-02
    Epoch number: [444 / 500] - Training error/loss: 1.161424e-02
    Epoch number: [445 / 500] - Training error/loss: 1.108368e-02
    Epoch number: [446 / 500] - Training error/loss: 1.176476e-02
    Epoch number: [447 / 500] - Training error/loss: 1.085203e-02
    Epoch number: [448 / 500] - Training error/loss: 1.074561e-02
    Epoch number: [449 / 500] - Training error/loss: 1.024523e-02
    Epoch number: [450 / 500] - Training error/loss: 1.067982e-02
    Epoch number: [451 / 500] - Training error/loss: 1.093154e-02
    Epoch number: [452 / 500] - Training error/loss: 1.138392e-02
    Epoch number: [453 / 500] - Training error/loss: 1.134967e-02
    Epoch number: [454 / 500] - Training error/loss: 1.181470e-02
    Epoch number: [455 / 500] - Training error/loss: 1.143552e-02
    Epoch number: [456 / 500] - Training error/loss: 1.104622e-02
    Epoch number: [457 / 500] - Training error/loss: 1.025088e-02
    Epoch number: [458 / 500] - Training error/loss: 1.121290e-02
    Epoch number: [459 / 500] - Training error/loss: 1.141868e-02
    Epoch number: [460 / 500] - Training error/loss: 1.059795e-02
    Epoch number: [461 / 500] - Training error/loss: 1.093176e-02
    Epoch number: [462 / 500] - Training error/loss: 1.010071e-02
    Epoch number: [463 / 500] - Training error/loss: 1.074534e-02
    Epoch number: [464 / 500] - Training error/loss: 1.075328e-02
    Epoch number: [465 / 500] - Training error/loss: 1.149180e-02
    Epoch number: [466 / 500] - Training error/loss: 1.241420e-02
    Epoch number: [467 / 500] - Training error/loss: 1.064605e-02
    Epoch number: [468 / 500] - Training error/loss: 1.123335e-02
    Epoch number: [469 / 500] - Training error/loss: 1.070721e-02
    Epoch number: [470 / 500] - Training error/loss: 1.206172e-02
    Epoch number: [471 / 500] - Training error/loss: 1.241415e-02
    Epoch number: [472 / 500] - Training error/loss: 1.047835e-02
    Epoch number: [473 / 500] - Training error/loss: 1.050262e-02
    Epoch number: [474 / 500] - Training error/loss: 1.064969e-02
    Epoch number: [475 / 500] - Training error/loss: 1.019282e-02
    Epoch number: [476 / 500] - Training error/loss: 1.044055e-02
    Epoch number: [477 / 500] - Training error/loss: 1.013787e-02
    Epoch number: [478 / 500] - Training error/loss: 1.068264e-02
    Epoch number: [479 / 500] - Training error/loss: 1.103005e-02
    Epoch number: [480 / 500] - Training error/loss: 1.145803e-02
    Epoch number: [481 / 500] - Training error/loss: 1.083715e-02
    Epoch number: [482 / 500] - Training error/loss: 1.039445e-02
    Epoch number: [483 / 500] - Training error/loss: 1.111938e-02
    Epoch number: [484 / 500] - Training error/loss: 1.053598e-02
    Epoch number: [485 / 500] - Training error/loss: 1.108802e-02
    Epoch number: [486 / 500] - Training error/loss: 1.084541e-02
    Epoch number: [487 / 500] - Training error/loss: 1.030881e-02
    Epoch number: [488 / 500] - Training error/loss: 1.093579e-02
    Epoch number: [489 / 500] - Training error/loss: 1.126138e-02
    Epoch number: [490 / 500] - Training error/loss: 1.097811e-02
    Epoch number: [491 / 500] - Training error/loss: 1.032683e-02
    Epoch number: [492 / 500] - Training error/loss: 1.008702e-02
    Epoch number: [493 / 500] - Training error/loss: 1.093923e-02
    Epoch number: [494 / 500] - Training error/loss: 1.134186e-02
    Epoch number: [495 / 500] - Training error/loss: 1.067571e-02
    Epoch number: [496 / 500] - Training error/loss: 1.057307e-02
    Epoch number: [497 / 500] - Training error/loss: 1.107341e-02
    Epoch number: [498 / 500] - Training error/loss: 1.098424e-02
    Epoch number: [499 / 500] - Training error/loss: 1.043103e-02
    Epoch number: [500 / 500] - Training error/loss: 1.138922e-02


```python
model_1.get_loss_plot(x_padding=10, y_padding=25, title_padding=15)
```


    
![Figure](assets/codes/python/python_5_files/output_13_0.png)
    


```python
model_1.get_prediction_plot(x1, y1, x_padding=10, y_padding=15, title_padding=15)
```


    
![Figure](assets/codes/python/python_5_files/output_14_0.png)
    


```python
H = 1

fig, ax = plt.subplots(figsize=(8, 6), dpi=90)

data_plot, = ax.plot([], [], "ro", markersize=2, label="Data")
pred_plot, = ax.plot([], [], "b-", linewidth=3, label="Prediction")

ax.set_xlim(torch.min(x1).item() - H, torch.max(x1).item() + H)
ax.set_xlabel(r"$x_{1}$", labelpad=10)
ax.set_ylim(torch.min(y1).item() - H, torch.max(y1).item() + H)
ax.set_ylabel(r"$y_{1}$", labelpad=15, rotation="horizontal")
ax.grid()
ax.legend(title="model_1", title_fontsize=16,
          loc="best", edgecolor="black",
          fancybox=False, shadow=True, borderaxespad=1)

plt.subplots_adjust(top=0.93, bottom=0.12)
plt.close()


def run_animation(i):
  data_plot.set_data(x1.numpy(), y1.numpy())
  pred_plot.set_data(x1.numpy(), history_1["predictions"][i + 1, :, :])
  ax.set_title(f"Iteration number: {i + 1}", pad=10)
  return data_plot, pred_plot


ani = animation.FuncAnimation(fig, run_animation, frames=range(0, EPOCH, STEP), interval=250, blit=True)
writer = PillowWriter(fps=15, bitrate=900)
ani.save("animation.gif", writer=writer)
display(Image("animation.gif"))
```


![Figure](assets/codes/python/python_5_files/output_15_0.gif)


# Problem #2

```python
x2 = np.linspace(-10, 10, DATASIZE)
y2 = seasonality(x2, pattern, period=11) + noise(DATASIZE, 0.05, 1.0, 2.0, SEED)

x2 = torch.from_numpy(x2).to(torch.float32).reshape((DATASIZE, 1))
y2 = torch.from_numpy(y2).to(torch.float32).reshape((DATASIZE, 1))

ds_2 = CreateDataset([x2, y2])
ds_loader_2 = torch.utils.data.DataLoader(ds_2, batch_size=32, shuffle=True)

print(f"'x2' data: {tuple(x2.shape)}",
      f"'y2' data: {tuple(y2.shape)}",
      sep="\n\n")
```

    'x2' data: (500, 1)
    
    'y2' data: (500, 1)


```python
torch.manual_seed(SEED)

def set_weight(weights):
  return torch.nn.init.xavier_uniform_(weights)


def set_bias(biases):
  return torch.nn.init.zeros_(biases)


def set_activation():
  return torch.nn.ReLU()


model_2 = FCNN("model_2", 1, 1, [64, 64], 3 * [set_weight], 3 * [set_bias], 2 * [False], 2 * [set_activation], 2 * [False])

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model_2.parameters())

model_2.get_summary()
```

    'Model structure': FCNN
    
    'Layer name': _layers.0.weight - 'Layer size': (64, 1)
    
    'Layer name': _layers.0.bias - 'Layer size': (64,)
    
    'Layer name': _layers.2.weight - 'Layer size': (64, 64)
    
    'Layer name': _layers.2.bias - 'Layer size': (64,)
    
    'Layer name': _layers.4.weight - 'Layer size': (1, 64)
    
    'Layer name': _layers.4.bias - 'Layer size': (1,)
    


```python
print(*model_2.parameters(), sep="\n\n")
```

    Parameter containing:
    tensor([[ 0.1241],
            [ 0.1202],
            [-0.1879],
            [-0.0947],
            [-0.2840],
            [ 0.1633],
            [ 0.2627],
            [ 0.0994],
            [-0.1880],
            [ 0.1603],
            [ 0.2521],
            [-0.0622],
            [-0.0498],
            [ 0.1052],
            [ 0.0391],
            [-0.1004],
            [-0.2821],
            [-0.2130],
            [ 0.0165],
            [-0.0163],
            [ 0.2509],
            [ 0.0104],
            [-0.0782],
            [-0.2969],
            [-0.2793],
            [ 0.1262],
            [ 0.2706],
            [-0.0046],
            [-0.2716],
            [ 0.1079],
            [ 0.2475],
            [-0.2011],
            [-0.0800],
            [ 0.0256],
            [-0.1733],
            [-0.2817],
            [ 0.1330],
            [-0.2493],
            [-0.0763],
            [-0.1613],
            [ 0.2285],
            [ 0.0587],
            [ 0.1409],
            [-0.2057],
            [ 0.2560],
            [ 0.0271],
            [-0.1685],
            [-0.0199],
            [-0.2151],
            [-0.0140],
            [ 0.1697],
            [-0.1366],
            [ 0.2568],
            [ 0.0928],
            [-0.0532],
            [ 0.0949],
            [-0.0560],
            [-0.0706],
            [-0.1993],
            [ 0.1770],
            [ 0.1824],
            [ 0.1040],
            [ 0.1797],
            [ 0.3035]], requires_grad=True)
    
    Parameter containing:
    tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           requires_grad=True)
    
    Parameter containing:
    tensor([[ 0.1086,  0.1079,  0.1447,  ..., -0.0754,  0.0668,  0.0211],
            [ 0.0175, -0.1901,  0.0325,  ...,  0.1395,  0.0915,  0.2050],
            [-0.0075, -0.1651,  0.0024,  ..., -0.0792,  0.0365, -0.0140],
            ...,
            [-0.1062,  0.1632, -0.2158,  ...,  0.0649,  0.1436, -0.0524],
            [ 0.0419, -0.0125, -0.1059,  ...,  0.0913, -0.0647,  0.1942],
            [ 0.1136,  0.0144, -0.0577,  ...,  0.0840, -0.1260,  0.0895]],
           requires_grad=True)
    
    Parameter containing:
    tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           requires_grad=True)
    
    Parameter containing:
    tensor([[-0.0446, -0.0026, -0.1054,  0.0096,  0.0254, -0.0886, -0.0745, -0.1141,
             -0.0805,  0.0287,  0.0142,  0.0023, -0.0504, -0.0342,  0.0716,  0.1024,
             -0.1212, -0.0362, -0.0054, -0.0358, -0.1028, -0.1174, -0.0329,  0.0999,
              0.0005,  0.0276,  0.0168, -0.0369,  0.0915, -0.0400, -0.0980, -0.1194,
              0.0973,  0.1082,  0.0269, -0.1165,  0.0537, -0.1077,  0.0440,  0.0954,
              0.0343,  0.0860, -0.0398, -0.0898, -0.0995,  0.0852,  0.1061,  0.0706,
              0.0998,  0.0195, -0.0999, -0.0177, -0.0283, -0.0607, -0.0675,  0.1095,
              0.0691,  0.1035,  0.0693, -0.0351, -0.0461,  0.0361, -0.1248,  0.0597]],
           requires_grad=True)
    
    Parameter containing:
    tensor([0.0903], requires_grad=True)


```python
history_2 = model_2.trainer(criterion, optimizer, EPOCH, training_loader=ds_loader_2, x=x2)
```

    Epoch number: [1 / 500] - Training error/loss: 2.094440e-01
    Epoch number: [2 / 500] - Training error/loss: 2.057643e-01
    Epoch number: [3 / 500] - Training error/loss: 2.035001e-01
    Epoch number: [4 / 500] - Training error/loss: 1.969204e-01
    Epoch number: [5 / 500] - Training error/loss: 1.939010e-01
    Epoch number: [6 / 500] - Training error/loss: 1.898171e-01
    Epoch number: [7 / 500] - Training error/loss: 1.873090e-01
    Epoch number: [8 / 500] - Training error/loss: 1.854770e-01
    Epoch number: [9 / 500] - Training error/loss: 1.826644e-01
    Epoch number: [10 / 500] - Training error/loss: 1.782108e-01
    Epoch number: [11 / 500] - Training error/loss: 1.749256e-01
    Epoch number: [12 / 500] - Training error/loss: 1.709157e-01
    Epoch number: [13 / 500] - Training error/loss: 1.701546e-01
    Epoch number: [14 / 500] - Training error/loss: 1.673563e-01
    Epoch number: [15 / 500] - Training error/loss: 1.666601e-01
    Epoch number: [16 / 500] - Training error/loss: 1.579833e-01
    Epoch number: [17 / 500] - Training error/loss: 1.551153e-01
    Epoch number: [18 / 500] - Training error/loss: 1.577996e-01
    Epoch number: [19 / 500] - Training error/loss: 1.552422e-01
    Epoch number: [20 / 500] - Training error/loss: 1.500153e-01
    Epoch number: [21 / 500] - Training error/loss: 1.447868e-01
    Epoch number: [22 / 500] - Training error/loss: 1.486967e-01
    Epoch number: [23 / 500] - Training error/loss: 1.424458e-01
    Epoch number: [24 / 500] - Training error/loss: 1.393066e-01
    Epoch number: [25 / 500] - Training error/loss: 1.360716e-01
    Epoch number: [26 / 500] - Training error/loss: 1.311374e-01
    Epoch number: [27 / 500] - Training error/loss: 1.296511e-01
    Epoch number: [28 / 500] - Training error/loss: 1.290844e-01
    Epoch number: [29 / 500] - Training error/loss: 1.316009e-01
    Epoch number: [30 / 500] - Training error/loss: 1.237151e-01
    Epoch number: [31 / 500] - Training error/loss: 1.233340e-01
    Epoch number: [32 / 500] - Training error/loss: 1.240143e-01
    Epoch number: [33 / 500] - Training error/loss: 1.245363e-01
    Epoch number: [34 / 500] - Training error/loss: 1.190573e-01
    Epoch number: [35 / 500] - Training error/loss: 1.190783e-01
    Epoch number: [36 / 500] - Training error/loss: 1.123771e-01
    Epoch number: [37 / 500] - Training error/loss: 1.089667e-01
    Epoch number: [38 / 500] - Training error/loss: 1.034687e-01
    Epoch number: [39 / 500] - Training error/loss: 1.032711e-01
    Epoch number: [40 / 500] - Training error/loss: 1.008041e-01
    Epoch number: [41 / 500] - Training error/loss: 9.767268e-02
    Epoch number: [42 / 500] - Training error/loss: 9.797303e-02
    Epoch number: [43 / 500] - Training error/loss: 9.330617e-02
    Epoch number: [44 / 500] - Training error/loss: 9.175345e-02
    Epoch number: [45 / 500] - Training error/loss: 8.603678e-02
    Epoch number: [46 / 500] - Training error/loss: 8.446789e-02
    Epoch number: [47 / 500] - Training error/loss: 8.060648e-02
    Epoch number: [48 / 500] - Training error/loss: 7.838903e-02
    Epoch number: [49 / 500] - Training error/loss: 7.420027e-02
    Epoch number: [50 / 500] - Training error/loss: 7.181432e-02
    Epoch number: [51 / 500] - Training error/loss: 6.871245e-02
    Epoch number: [52 / 500] - Training error/loss: 6.826819e-02
    Epoch number: [53 / 500] - Training error/loss: 6.574958e-02
    Epoch number: [54 / 500] - Training error/loss: 6.407738e-02
    Epoch number: [55 / 500] - Training error/loss: 5.744290e-02
    Epoch number: [56 / 500] - Training error/loss: 5.661064e-02
    Epoch number: [57 / 500] - Training error/loss: 5.500454e-02
    Epoch number: [58 / 500] - Training error/loss: 4.991859e-02
    Epoch number: [59 / 500] - Training error/loss: 4.749751e-02
    Epoch number: [60 / 500] - Training error/loss: 4.678405e-02
    Epoch number: [61 / 500] - Training error/loss: 4.659496e-02
    Epoch number: [62 / 500] - Training error/loss: 4.585077e-02
    Epoch number: [63 / 500] - Training error/loss: 3.930996e-02
    Epoch number: [64 / 500] - Training error/loss: 4.350106e-02
    Epoch number: [65 / 500] - Training error/loss: 4.002179e-02
    Epoch number: [66 / 500] - Training error/loss: 3.606008e-02
    Epoch number: [67 / 500] - Training error/loss: 3.519978e-02
    Epoch number: [68 / 500] - Training error/loss: 3.405049e-02
    Epoch number: [69 / 500] - Training error/loss: 3.105877e-02
    Epoch number: [70 / 500] - Training error/loss: 3.172713e-02
    Epoch number: [71 / 500] - Training error/loss: 3.213524e-02
    Epoch number: [72 / 500] - Training error/loss: 3.232847e-02
    Epoch number: [73 / 500] - Training error/loss: 2.923583e-02
    Epoch number: [74 / 500] - Training error/loss: 3.126162e-02
    Epoch number: [75 / 500] - Training error/loss: 3.159613e-02
    Epoch number: [76 / 500] - Training error/loss: 3.033361e-02
    Epoch number: [77 / 500] - Training error/loss: 2.838037e-02
    Epoch number: [78 / 500] - Training error/loss: 2.931574e-02
    Epoch number: [79 / 500] - Training error/loss: 3.060131e-02
    Epoch number: [80 / 500] - Training error/loss: 3.062069e-02
    Epoch number: [81 / 500] - Training error/loss: 2.776314e-02
    Epoch number: [82 / 500] - Training error/loss: 2.845834e-02
    Epoch number: [83 / 500] - Training error/loss: 2.797208e-02
    Epoch number: [84 / 500] - Training error/loss: 2.941487e-02
    Epoch number: [85 / 500] - Training error/loss: 2.500168e-02
    Epoch number: [86 / 500] - Training error/loss: 2.623464e-02
    Epoch number: [87 / 500] - Training error/loss: 2.463551e-02
    Epoch number: [88 / 500] - Training error/loss: 2.727174e-02
    Epoch number: [89 / 500] - Training error/loss: 2.662466e-02
    Epoch number: [90 / 500] - Training error/loss: 2.685143e-02
    Epoch number: [91 / 500] - Training error/loss: 2.531828e-02
    Epoch number: [92 / 500] - Training error/loss: 2.507360e-02
    Epoch number: [93 / 500] - Training error/loss: 2.497226e-02
    Epoch number: [94 / 500] - Training error/loss: 2.564469e-02
    Epoch number: [95 / 500] - Training error/loss: 2.616997e-02
    Epoch number: [96 / 500] - Training error/loss: 2.815245e-02
    Epoch number: [97 / 500] - Training error/loss: 2.742019e-02
    Epoch number: [98 / 500] - Training error/loss: 2.659543e-02
    Epoch number: [99 / 500] - Training error/loss: 2.488089e-02
    Epoch number: [100 / 500] - Training error/loss: 2.381690e-02
    Epoch number: [101 / 500] - Training error/loss: 2.467619e-02
    Epoch number: [102 / 500] - Training error/loss: 2.571482e-02
    Epoch number: [103 / 500] - Training error/loss: 2.359570e-02
    Epoch number: [104 / 500] - Training error/loss: 2.436644e-02
    Epoch number: [105 / 500] - Training error/loss: 2.617952e-02
    Epoch number: [106 / 500] - Training error/loss: 2.395081e-02
    Epoch number: [107 / 500] - Training error/loss: 2.405164e-02
    Epoch number: [108 / 500] - Training error/loss: 2.370910e-02
    Epoch number: [109 / 500] - Training error/loss: 2.185602e-02
    Epoch number: [110 / 500] - Training error/loss: 2.376907e-02
    Epoch number: [111 / 500] - Training error/loss: 2.328549e-02
    Epoch number: [112 / 500] - Training error/loss: 2.625599e-02
    Epoch number: [113 / 500] - Training error/loss: 2.559275e-02
    Epoch number: [114 / 500] - Training error/loss: 2.232242e-02
    Epoch number: [115 / 500] - Training error/loss: 2.293899e-02
    Epoch number: [116 / 500] - Training error/loss: 2.230242e-02
    Epoch number: [117 / 500] - Training error/loss: 2.368368e-02
    Epoch number: [118 / 500] - Training error/loss: 2.673245e-02
    Epoch number: [119 / 500] - Training error/loss: 2.479154e-02
    Epoch number: [120 / 500] - Training error/loss: 2.095383e-02
    Epoch number: [121 / 500] - Training error/loss: 2.124329e-02
    Epoch number: [122 / 500] - Training error/loss: 2.109145e-02
    Epoch number: [123 / 500] - Training error/loss: 2.025047e-02
    Epoch number: [124 / 500] - Training error/loss: 2.079745e-02
    Epoch number: [125 / 500] - Training error/loss: 2.199195e-02
    Epoch number: [126 / 500] - Training error/loss: 2.206681e-02
    Epoch number: [127 / 500] - Training error/loss: 2.106249e-02
    Epoch number: [128 / 500] - Training error/loss: 2.137157e-02
    Epoch number: [129 / 500] - Training error/loss: 1.988963e-02
    Epoch number: [130 / 500] - Training error/loss: 2.157079e-02
    Epoch number: [131 / 500] - Training error/loss: 1.900863e-02
    Epoch number: [132 / 500] - Training error/loss: 2.139199e-02
    Epoch number: [133 / 500] - Training error/loss: 2.022335e-02
    Epoch number: [134 / 500] - Training error/loss: 2.068844e-02
    Epoch number: [135 / 500] - Training error/loss: 1.926370e-02
    Epoch number: [136 / 500] - Training error/loss: 1.869851e-02
    Epoch number: [137 / 500] - Training error/loss: 1.890662e-02
    Epoch number: [138 / 500] - Training error/loss: 1.835360e-02
    Epoch number: [139 / 500] - Training error/loss: 1.800355e-02
    Epoch number: [140 / 500] - Training error/loss: 1.810444e-02
    Epoch number: [141 / 500] - Training error/loss: 1.726598e-02
    Epoch number: [142 / 500] - Training error/loss: 1.858974e-02
    Epoch number: [143 / 500] - Training error/loss: 1.907146e-02
    Epoch number: [144 / 500] - Training error/loss: 1.818518e-02
    Epoch number: [145 / 500] - Training error/loss: 1.853344e-02
    Epoch number: [146 / 500] - Training error/loss: 1.792493e-02
    Epoch number: [147 / 500] - Training error/loss: 1.907823e-02
    Epoch number: [148 / 500] - Training error/loss: 1.753397e-02
    Epoch number: [149 / 500] - Training error/loss: 1.662142e-02
    Epoch number: [150 / 500] - Training error/loss: 1.620722e-02
    Epoch number: [151 / 500] - Training error/loss: 1.605294e-02
    Epoch number: [152 / 500] - Training error/loss: 1.708760e-02
    Epoch number: [153 / 500] - Training error/loss: 1.652984e-02
    Epoch number: [154 / 500] - Training error/loss: 1.742683e-02
    Epoch number: [155 / 500] - Training error/loss: 1.733892e-02
    Epoch number: [156 / 500] - Training error/loss: 1.884621e-02
    Epoch number: [157 / 500] - Training error/loss: 1.792439e-02
    Epoch number: [158 / 500] - Training error/loss: 1.601688e-02
    Epoch number: [159 / 500] - Training error/loss: 1.637873e-02
    Epoch number: [160 / 500] - Training error/loss: 1.575991e-02
    Epoch number: [161 / 500] - Training error/loss: 1.562750e-02
    Epoch number: [162 / 500] - Training error/loss: 1.693650e-02
    Epoch number: [163 / 500] - Training error/loss: 1.652350e-02
    Epoch number: [164 / 500] - Training error/loss: 1.556768e-02
    Epoch number: [165 / 500] - Training error/loss: 1.566975e-02
    Epoch number: [166 / 500] - Training error/loss: 1.388058e-02
    Epoch number: [167 / 500] - Training error/loss: 1.378274e-02
    Epoch number: [168 / 500] - Training error/loss: 1.480310e-02
    Epoch number: [169 / 500] - Training error/loss: 1.385960e-02
    Epoch number: [170 / 500] - Training error/loss: 1.521473e-02
    Epoch number: [171 / 500] - Training error/loss: 1.427231e-02
    Epoch number: [172 / 500] - Training error/loss: 1.468363e-02
    Epoch number: [173 / 500] - Training error/loss: 1.451597e-02
    Epoch number: [174 / 500] - Training error/loss: 1.342722e-02
    Epoch number: [175 / 500] - Training error/loss: 1.303486e-02
    Epoch number: [176 / 500] - Training error/loss: 1.312166e-02
    Epoch number: [177 / 500] - Training error/loss: 1.345800e-02
    Epoch number: [178 / 500] - Training error/loss: 1.292246e-02
    Epoch number: [179 / 500] - Training error/loss: 1.375657e-02
    Epoch number: [180 / 500] - Training error/loss: 1.438917e-02
    Epoch number: [181 / 500] - Training error/loss: 1.349882e-02
    Epoch number: [182 / 500] - Training error/loss: 1.243082e-02
    Epoch number: [183 / 500] - Training error/loss: 1.373899e-02
    Epoch number: [184 / 500] - Training error/loss: 1.315850e-02
    Epoch number: [185 / 500] - Training error/loss: 1.244815e-02
    Epoch number: [186 / 500] - Training error/loss: 1.312837e-02
    Epoch number: [187 / 500] - Training error/loss: 1.395761e-02
    Epoch number: [188 / 500] - Training error/loss: 1.339056e-02
    Epoch number: [189 / 500] - Training error/loss: 1.326432e-02
    Epoch number: [190 / 500] - Training error/loss: 1.243608e-02
    Epoch number: [191 / 500] - Training error/loss: 1.339563e-02
    Epoch number: [192 / 500] - Training error/loss: 1.442817e-02
    Epoch number: [193 / 500] - Training error/loss: 1.259333e-02
    Epoch number: [194 / 500] - Training error/loss: 1.347822e-02
    Epoch number: [195 / 500] - Training error/loss: 1.442154e-02
    Epoch number: [196 / 500] - Training error/loss: 1.693420e-02
    Epoch number: [197 / 500] - Training error/loss: 1.579481e-02
    Epoch number: [198 / 500] - Training error/loss: 1.334440e-02
    Epoch number: [199 / 500] - Training error/loss: 1.313810e-02
    Epoch number: [200 / 500] - Training error/loss: 1.301367e-02
    Epoch number: [201 / 500] - Training error/loss: 1.282853e-02
    Epoch number: [202 / 500] - Training error/loss: 1.221118e-02
    Epoch number: [203 / 500] - Training error/loss: 1.208467e-02
    Epoch number: [204 / 500] - Training error/loss: 1.182760e-02
    Epoch number: [205 / 500] - Training error/loss: 1.224557e-02
    Epoch number: [206 / 500] - Training error/loss: 1.331175e-02
    Epoch number: [207 / 500] - Training error/loss: 1.283170e-02
    Epoch number: [208 / 500] - Training error/loss: 1.310790e-02
    Epoch number: [209 / 500] - Training error/loss: 1.217873e-02
    Epoch number: [210 / 500] - Training error/loss: 1.195403e-02
    Epoch number: [211 / 500] - Training error/loss: 1.277447e-02
    Epoch number: [212 / 500] - Training error/loss: 1.293059e-02
    Epoch number: [213 / 500] - Training error/loss: 1.378753e-02
    Epoch number: [214 / 500] - Training error/loss: 1.586750e-02
    Epoch number: [215 / 500] - Training error/loss: 1.652759e-02
    Epoch number: [216 / 500] - Training error/loss: 1.521993e-02
    Epoch number: [217 / 500] - Training error/loss: 1.259289e-02
    Epoch number: [218 / 500] - Training error/loss: 1.274848e-02
    Epoch number: [219 / 500] - Training error/loss: 1.207152e-02
    Epoch number: [220 / 500] - Training error/loss: 1.129713e-02
    Epoch number: [221 / 500] - Training error/loss: 1.173519e-02
    Epoch number: [222 / 500] - Training error/loss: 1.230402e-02
    Epoch number: [223 / 500] - Training error/loss: 1.179446e-02
    Epoch number: [224 / 500] - Training error/loss: 1.173015e-02
    Epoch number: [225 / 500] - Training error/loss: 1.176792e-02
    Epoch number: [226 / 500] - Training error/loss: 1.309248e-02
    Epoch number: [227 / 500] - Training error/loss: 1.187959e-02
    Epoch number: [228 / 500] - Training error/loss: 1.213496e-02
    Epoch number: [229 / 500] - Training error/loss: 1.275789e-02
    Epoch number: [230 / 500] - Training error/loss: 1.274257e-02
    Epoch number: [231 / 500] - Training error/loss: 1.192227e-02
    Epoch number: [232 / 500] - Training error/loss: 1.238869e-02
    Epoch number: [233 / 500] - Training error/loss: 1.141657e-02
    Epoch number: [234 / 500] - Training error/loss: 1.200998e-02
    Epoch number: [235 / 500] - Training error/loss: 1.143051e-02
    Epoch number: [236 / 500] - Training error/loss: 1.248485e-02
    Epoch number: [237 / 500] - Training error/loss: 1.173120e-02
    Epoch number: [238 / 500] - Training error/loss: 1.499219e-02
    Epoch number: [239 / 500] - Training error/loss: 1.535236e-02
    Epoch number: [240 / 500] - Training error/loss: 1.279423e-02
    Epoch number: [241 / 500] - Training error/loss: 1.401076e-02
    Epoch number: [242 / 500] - Training error/loss: 1.410945e-02
    Epoch number: [243 / 500] - Training error/loss: 1.256392e-02
    Epoch number: [244 / 500] - Training error/loss: 1.322307e-02
    Epoch number: [245 / 500] - Training error/loss: 1.156908e-02
    Epoch number: [246 / 500] - Training error/loss: 1.206929e-02
    Epoch number: [247 / 500] - Training error/loss: 1.115169e-02
    Epoch number: [248 / 500] - Training error/loss: 1.189623e-02
    Epoch number: [249 / 500] - Training error/loss: 1.109377e-02
    Epoch number: [250 / 500] - Training error/loss: 1.179701e-02
    Epoch number: [251 / 500] - Training error/loss: 1.276834e-02
    Epoch number: [252 / 500] - Training error/loss: 1.256159e-02
    Epoch number: [253 / 500] - Training error/loss: 1.121064e-02
    Epoch number: [254 / 500] - Training error/loss: 1.182496e-02
    Epoch number: [255 / 500] - Training error/loss: 1.137387e-02
    Epoch number: [256 / 500] - Training error/loss: 1.146302e-02
    Epoch number: [257 / 500] - Training error/loss: 1.222388e-02
    Epoch number: [258 / 500] - Training error/loss: 1.130473e-02
    Epoch number: [259 / 500] - Training error/loss: 1.208415e-02
    Epoch number: [260 / 500] - Training error/loss: 1.247436e-02
    Epoch number: [261 / 500] - Training error/loss: 1.198110e-02
    Epoch number: [262 / 500] - Training error/loss: 1.124135e-02
    Epoch number: [263 / 500] - Training error/loss: 1.240857e-02
    Epoch number: [264 / 500] - Training error/loss: 1.216023e-02
    Epoch number: [265 / 500] - Training error/loss: 1.239682e-02
    Epoch number: [266 / 500] - Training error/loss: 1.142285e-02
    Epoch number: [267 / 500] - Training error/loss: 1.147906e-02
    Epoch number: [268 / 500] - Training error/loss: 1.196623e-02
    Epoch number: [269 / 500] - Training error/loss: 1.244227e-02
    Epoch number: [270 / 500] - Training error/loss: 1.196153e-02
    Epoch number: [271 / 500] - Training error/loss: 1.146195e-02
    Epoch number: [272 / 500] - Training error/loss: 1.187643e-02
    Epoch number: [273 / 500] - Training error/loss: 1.162208e-02
    Epoch number: [274 / 500] - Training error/loss: 1.211013e-02
    Epoch number: [275 / 500] - Training error/loss: 1.146346e-02
    Epoch number: [276 / 500] - Training error/loss: 1.318129e-02
    Epoch number: [277 / 500] - Training error/loss: 1.199836e-02
    Epoch number: [278 / 500] - Training error/loss: 1.236561e-02
    Epoch number: [279 / 500] - Training error/loss: 1.204277e-02
    Epoch number: [280 / 500] - Training error/loss: 1.121381e-02
    Epoch number: [281 / 500] - Training error/loss: 1.190842e-02
    Epoch number: [282 / 500] - Training error/loss: 1.167143e-02
    Epoch number: [283 / 500] - Training error/loss: 1.115657e-02
    Epoch number: [284 / 500] - Training error/loss: 1.139227e-02
    Epoch number: [285 / 500] - Training error/loss: 1.328175e-02
    Epoch number: [286 / 500] - Training error/loss: 1.195156e-02
    Epoch number: [287 / 500] - Training error/loss: 1.195399e-02
    Epoch number: [288 / 500] - Training error/loss: 1.201256e-02
    Epoch number: [289 / 500] - Training error/loss: 1.261940e-02
    Epoch number: [290 / 500] - Training error/loss: 1.574087e-02
    Epoch number: [291 / 500] - Training error/loss: 1.189059e-02
    Epoch number: [292 / 500] - Training error/loss: 1.267333e-02
    Epoch number: [293 / 500] - Training error/loss: 1.247416e-02
    Epoch number: [294 / 500] - Training error/loss: 1.099006e-02
    Epoch number: [295 / 500] - Training error/loss: 1.252911e-02
    Epoch number: [296 / 500] - Training error/loss: 1.139640e-02
    Epoch number: [297 / 500] - Training error/loss: 1.161008e-02
    Epoch number: [298 / 500] - Training error/loss: 1.184551e-02
    Epoch number: [299 / 500] - Training error/loss: 1.286820e-02
    Epoch number: [300 / 500] - Training error/loss: 1.252431e-02
    Epoch number: [301 / 500] - Training error/loss: 1.226646e-02
    Epoch number: [302 / 500] - Training error/loss: 1.088813e-02
    Epoch number: [303 / 500] - Training error/loss: 1.101484e-02
    Epoch number: [304 / 500] - Training error/loss: 1.237748e-02
    Epoch number: [305 / 500] - Training error/loss: 1.283454e-02
    Epoch number: [306 / 500] - Training error/loss: 1.152327e-02
    Epoch number: [307 / 500] - Training error/loss: 1.124965e-02
    Epoch number: [308 / 500] - Training error/loss: 1.163669e-02
    Epoch number: [309 / 500] - Training error/loss: 1.210968e-02
    Epoch number: [310 / 500] - Training error/loss: 1.180582e-02
    Epoch number: [311 / 500] - Training error/loss: 1.159052e-02
    Epoch number: [312 / 500] - Training error/loss: 1.203281e-02
    Epoch number: [313 / 500] - Training error/loss: 1.300073e-02
    Epoch number: [314 / 500] - Training error/loss: 1.089521e-02
    Epoch number: [315 / 500] - Training error/loss: 1.179647e-02
    Epoch number: [316 / 500] - Training error/loss: 1.210627e-02
    Epoch number: [317 / 500] - Training error/loss: 1.116997e-02
    Epoch number: [318 / 500] - Training error/loss: 1.213691e-02
    Epoch number: [319 / 500] - Training error/loss: 1.138907e-02
    Epoch number: [320 / 500] - Training error/loss: 1.130370e-02
    Epoch number: [321 / 500] - Training error/loss: 1.101631e-02
    Epoch number: [322 / 500] - Training error/loss: 1.236447e-02
    Epoch number: [323 / 500] - Training error/loss: 1.148273e-02
    Epoch number: [324 / 500] - Training error/loss: 1.137669e-02
    Epoch number: [325 / 500] - Training error/loss: 1.094495e-02
    Epoch number: [326 / 500] - Training error/loss: 1.135578e-02
    Epoch number: [327 / 500] - Training error/loss: 1.158716e-02
    Epoch number: [328 / 500] - Training error/loss: 1.098402e-02
    Epoch number: [329 / 500] - Training error/loss: 1.223202e-02
    Epoch number: [330 / 500] - Training error/loss: 1.465936e-02
    Epoch number: [331 / 500] - Training error/loss: 1.365259e-02
    Epoch number: [332 / 500] - Training error/loss: 1.284732e-02
    Epoch number: [333 / 500] - Training error/loss: 1.127020e-02
    Epoch number: [334 / 500] - Training error/loss: 1.099221e-02
    Epoch number: [335 / 500] - Training error/loss: 1.067060e-02
    Epoch number: [336 / 500] - Training error/loss: 1.176849e-02
    Epoch number: [337 / 500] - Training error/loss: 1.277574e-02
    Epoch number: [338 / 500] - Training error/loss: 1.141682e-02
    Epoch number: [339 / 500] - Training error/loss: 1.105825e-02
    Epoch number: [340 / 500] - Training error/loss: 1.104356e-02
    Epoch number: [341 / 500] - Training error/loss: 1.079635e-02
    Epoch number: [342 / 500] - Training error/loss: 1.241563e-02
    Epoch number: [343 / 500] - Training error/loss: 1.258075e-02
    Epoch number: [344 / 500] - Training error/loss: 1.192381e-02
    Epoch number: [345 / 500] - Training error/loss: 1.091137e-02
    Epoch number: [346 / 500] - Training error/loss: 1.097054e-02
    Epoch number: [347 / 500] - Training error/loss: 1.154300e-02
    Epoch number: [348 / 500] - Training error/loss: 1.188898e-02
    Epoch number: [349 / 500] - Training error/loss: 1.134035e-02
    Epoch number: [350 / 500] - Training error/loss: 1.079174e-02
    Epoch number: [351 / 500] - Training error/loss: 1.135774e-02
    Epoch number: [352 / 500] - Training error/loss: 1.106332e-02
    Epoch number: [353 / 500] - Training error/loss: 1.122421e-02
    Epoch number: [354 / 500] - Training error/loss: 1.267859e-02
    Epoch number: [355 / 500] - Training error/loss: 1.265732e-02
    Epoch number: [356 / 500] - Training error/loss: 1.300189e-02
    Epoch number: [357 / 500] - Training error/loss: 1.167856e-02
    Epoch number: [358 / 500] - Training error/loss: 1.204530e-02
    Epoch number: [359 / 500] - Training error/loss: 1.090334e-02
    Epoch number: [360 / 500] - Training error/loss: 1.109777e-02
    Epoch number: [361 / 500] - Training error/loss: 1.091047e-02
    Epoch number: [362 / 500] - Training error/loss: 1.167718e-02
    Epoch number: [363 / 500] - Training error/loss: 1.180519e-02
    Epoch number: [364 / 500] - Training error/loss: 1.156213e-02
    Epoch number: [365 / 500] - Training error/loss: 1.227390e-02
    Epoch number: [366 / 500] - Training error/loss: 1.150655e-02
    Epoch number: [367 / 500] - Training error/loss: 1.060071e-02
    Epoch number: [368 / 500] - Training error/loss: 1.133423e-02
    Epoch number: [369 / 500] - Training error/loss: 1.116439e-02
    Epoch number: [370 / 500] - Training error/loss: 1.079402e-02
    Epoch number: [371 / 500] - Training error/loss: 1.102626e-02
    Epoch number: [372 / 500] - Training error/loss: 1.134426e-02
    Epoch number: [373 / 500] - Training error/loss: 1.137506e-02
    Epoch number: [374 / 500] - Training error/loss: 1.160421e-02
    Epoch number: [375 / 500] - Training error/loss: 1.074819e-02
    Epoch number: [376 / 500] - Training error/loss: 1.040698e-02
    Epoch number: [377 / 500] - Training error/loss: 1.102099e-02
    Epoch number: [378 / 500] - Training error/loss: 1.136258e-02
    Epoch number: [379 / 500] - Training error/loss: 1.111977e-02
    Epoch number: [380 / 500] - Training error/loss: 1.161600e-02
    Epoch number: [381 / 500] - Training error/loss: 1.094249e-02
    Epoch number: [382 / 500] - Training error/loss: 1.128996e-02
    Epoch number: [383 / 500] - Training error/loss: 1.373790e-02
    Epoch number: [384 / 500] - Training error/loss: 1.339737e-02
    Epoch number: [385 / 500] - Training error/loss: 1.113191e-02
    Epoch number: [386 / 500] - Training error/loss: 1.162121e-02
    Epoch number: [387 / 500] - Training error/loss: 1.117167e-02
    Epoch number: [388 / 500] - Training error/loss: 1.126390e-02
    Epoch number: [389 / 500] - Training error/loss: 1.183445e-02
    Epoch number: [390 / 500] - Training error/loss: 1.161867e-02
    Epoch number: [391 / 500] - Training error/loss: 1.274800e-02
    Epoch number: [392 / 500] - Training error/loss: 1.107295e-02
    Epoch number: [393 / 500] - Training error/loss: 1.092950e-02
    Epoch number: [394 / 500] - Training error/loss: 1.163839e-02
    Epoch number: [395 / 500] - Training error/loss: 1.186584e-02
    Epoch number: [396 / 500] - Training error/loss: 1.101865e-02
    Epoch number: [397 / 500] - Training error/loss: 1.130654e-02
    Epoch number: [398 / 500] - Training error/loss: 1.108949e-02
    Epoch number: [399 / 500] - Training error/loss: 1.062992e-02
    Epoch number: [400 / 500] - Training error/loss: 1.113554e-02
    Epoch number: [401 / 500] - Training error/loss: 1.121299e-02
    Epoch number: [402 / 500] - Training error/loss: 1.087901e-02
    Epoch number: [403 / 500] - Training error/loss: 1.054452e-02
    Epoch number: [404 / 500] - Training error/loss: 1.190733e-02
    Epoch number: [405 / 500] - Training error/loss: 1.091058e-02
    Epoch number: [406 / 500] - Training error/loss: 1.200692e-02
    Epoch number: [407 / 500] - Training error/loss: 1.065480e-02
    Epoch number: [408 / 500] - Training error/loss: 1.106474e-02
    Epoch number: [409 / 500] - Training error/loss: 1.171089e-02
    Epoch number: [410 / 500] - Training error/loss: 1.134280e-02
    Epoch number: [411 / 500] - Training error/loss: 1.108485e-02
    Epoch number: [412 / 500] - Training error/loss: 1.100386e-02
    Epoch number: [413 / 500] - Training error/loss: 1.154958e-02
    Epoch number: [414 / 500] - Training error/loss: 1.154727e-02
    Epoch number: [415 / 500] - Training error/loss: 1.214516e-02
    Epoch number: [416 / 500] - Training error/loss: 1.163042e-02
    Epoch number: [417 / 500] - Training error/loss: 1.115218e-02
    Epoch number: [418 / 500] - Training error/loss: 1.061485e-02
    Epoch number: [419 / 500] - Training error/loss: 1.072552e-02
    Epoch number: [420 / 500] - Training error/loss: 1.137238e-02
    Epoch number: [421 / 500] - Training error/loss: 1.245566e-02
    Epoch number: [422 / 500] - Training error/loss: 1.230339e-02
    Epoch number: [423 / 500] - Training error/loss: 1.067571e-02
    Epoch number: [424 / 500] - Training error/loss: 1.077694e-02
    Epoch number: [425 / 500] - Training error/loss: 1.227160e-02
    Epoch number: [426 / 500] - Training error/loss: 1.287466e-02
    Epoch number: [427 / 500] - Training error/loss: 1.296285e-02
    Epoch number: [428 / 500] - Training error/loss: 1.324332e-02
    Epoch number: [429 / 500] - Training error/loss: 1.113723e-02
    Epoch number: [430 / 500] - Training error/loss: 1.159765e-02
    Epoch number: [431 / 500] - Training error/loss: 1.083802e-02
    Epoch number: [432 / 500] - Training error/loss: 1.103306e-02
    Epoch number: [433 / 500] - Training error/loss: 1.148555e-02
    Epoch number: [434 / 500] - Training error/loss: 1.196020e-02
    Epoch number: [435 / 500] - Training error/loss: 1.172052e-02
    Epoch number: [436 / 500] - Training error/loss: 1.081766e-02
    Epoch number: [437 / 500] - Training error/loss: 1.078805e-02
    Epoch number: [438 / 500] - Training error/loss: 1.133307e-02
    Epoch number: [439 / 500] - Training error/loss: 1.079862e-02
    Epoch number: [440 / 500] - Training error/loss: 1.109860e-02
    Epoch number: [441 / 500] - Training error/loss: 1.061787e-02
    Epoch number: [442 / 500] - Training error/loss: 1.098529e-02
    Epoch number: [443 / 500] - Training error/loss: 1.161324e-02
    Epoch number: [444 / 500] - Training error/loss: 1.167927e-02
    Epoch number: [445 / 500] - Training error/loss: 1.155208e-02
    Epoch number: [446 / 500] - Training error/loss: 1.120596e-02
    Epoch number: [447 / 500] - Training error/loss: 1.110168e-02
    Epoch number: [448 / 500] - Training error/loss: 1.101620e-02
    Epoch number: [449 / 500] - Training error/loss: 1.042334e-02
    Epoch number: [450 / 500] - Training error/loss: 1.079572e-02
    Epoch number: [451 / 500] - Training error/loss: 1.100147e-02
    Epoch number: [452 / 500] - Training error/loss: 1.113946e-02
    Epoch number: [453 / 500] - Training error/loss: 1.118417e-02
    Epoch number: [454 / 500] - Training error/loss: 1.144973e-02
    Epoch number: [455 / 500] - Training error/loss: 1.090338e-02
    Epoch number: [456 / 500] - Training error/loss: 1.031775e-02
    Epoch number: [457 / 500] - Training error/loss: 1.078503e-02
    Epoch number: [458 / 500] - Training error/loss: 1.125078e-02
    Epoch number: [459 / 500] - Training error/loss: 1.104072e-02
    Epoch number: [460 / 500] - Training error/loss: 1.109175e-02
    Epoch number: [461 / 500] - Training error/loss: 1.067502e-02
    Epoch number: [462 / 500] - Training error/loss: 1.036995e-02
    Epoch number: [463 / 500] - Training error/loss: 1.101803e-02
    Epoch number: [464 / 500] - Training error/loss: 1.099251e-02
    Epoch number: [465 / 500] - Training error/loss: 1.154118e-02
    Epoch number: [466 / 500] - Training error/loss: 1.254920e-02
    Epoch number: [467 / 500] - Training error/loss: 1.274312e-02
    Epoch number: [468 / 500] - Training error/loss: 1.150174e-02
    Epoch number: [469 / 500] - Training error/loss: 1.107347e-02
    Epoch number: [470 / 500] - Training error/loss: 1.137410e-02
    Epoch number: [471 / 500] - Training error/loss: 1.176479e-02
    Epoch number: [472 / 500] - Training error/loss: 1.109245e-02
    Epoch number: [473 / 500] - Training error/loss: 1.037389e-02
    Epoch number: [474 / 500] - Training error/loss: 1.074958e-02
    Epoch number: [475 / 500] - Training error/loss: 1.070688e-02
    Epoch number: [476 / 500] - Training error/loss: 1.087163e-02
    Epoch number: [477 / 500] - Training error/loss: 1.027155e-02
    Epoch number: [478 / 500] - Training error/loss: 1.119446e-02
    Epoch number: [479 / 500] - Training error/loss: 1.078744e-02
    Epoch number: [480 / 500] - Training error/loss: 1.073546e-02
    Epoch number: [481 / 500] - Training error/loss: 1.056787e-02
    Epoch number: [482 / 500] - Training error/loss: 1.027339e-02
    Epoch number: [483 / 500] - Training error/loss: 1.101735e-02
    Epoch number: [484 / 500] - Training error/loss: 1.063291e-02
    Epoch number: [485 / 500] - Training error/loss: 1.057783e-02
    Epoch number: [486 / 500] - Training error/loss: 1.087796e-02
    Epoch number: [487 / 500] - Training error/loss: 1.100470e-02
    Epoch number: [488 / 500] - Training error/loss: 1.051173e-02
    Epoch number: [489 / 500] - Training error/loss: 1.034746e-02
    Epoch number: [490 / 500] - Training error/loss: 1.121718e-02
    Epoch number: [491 / 500] - Training error/loss: 1.148014e-02
    Epoch number: [492 / 500] - Training error/loss: 1.102686e-02
    Epoch number: [493 / 500] - Training error/loss: 1.211891e-02
    Epoch number: [494 / 500] - Training error/loss: 1.145676e-02
    Epoch number: [495 / 500] - Training error/loss: 1.119438e-02
    Epoch number: [496 / 500] - Training error/loss: 1.087110e-02
    Epoch number: [497 / 500] - Training error/loss: 1.090182e-02
    Epoch number: [498 / 500] - Training error/loss: 1.109425e-02
    Epoch number: [499 / 500] - Training error/loss: 1.071044e-02
    Epoch number: [500 / 500] - Training error/loss: 1.098765e-02


```python
model_2.get_loss_plot(x_padding=10, y_padding=25, title_padding=15)
```


    
![Figure](assets/codes/python/python_5_files/output_21_0.png)
    


```python
model_2.get_prediction_plot(x2, y2, x_padding=10, y_padding=25, title_padding=15)
```


    
![Figure](assets/codes/python/python_5_files/output_22_0.png)
    


```python
H = 1

fig, ax = plt.subplots(figsize=(8, 6), dpi=90)

data_plot, = ax.plot([], [], "ro", markersize=2, label="Data")
pred_plot, = ax.plot([], [], "b-", linewidth=3, label="Prediction")

ax.set_xlim(torch.min(x2).item() - H, torch.max(x2).item() + H)
ax.set_xlabel(r"$x_{2}$", labelpad=10)
ax.set_ylim(torch.min(y2).item() - H, torch.max(y2).item() + H)
ax.set_ylabel(r"$y_{2}$", labelpad=20, rotation="horizontal")
ax.grid()
ax.legend(title="model_2", title_fontsize=16,
          loc="best", edgecolor="black",
          fancybox=False, shadow=True, borderaxespad=1)

plt.subplots_adjust(top=0.93, bottom=0.12)
plt.close()


def run_animation(i):
  data_plot.set_data(x2.numpy(), y2.numpy())
  pred_plot.set_data(x2.numpy(), history_2["predictions"][i + 1, :, :])
  ax.set_title(f"Iteration number: {i + 1}", pad=10)
  return data_plot, pred_plot


ani = animation.FuncAnimation(fig, run_animation, frames=range(0, EPOCH, STEP), interval=250, blit=True)
writer = PillowWriter(fps=15, bitrate=900)
ani.save("animation.gif", writer=writer)
display(Image("animation.gif"))
```


![Figure](assets/codes/python/python_5_files/output_23_0.gif)


# Conclusion

This comparative nonlinear regression example carried out on two problems through a deep learning model via **PyTorch**. In the first problem dataset was generated as a series of points fluctuating in a periodic manner in contrast to the second problem, in which dataset points were distributed in a discontinued trend where there is an abrupt jump in the middle. For both problems the same model (i.e., same number of neurons, layers, initialized weights, biases, activation and loss function, and optimizer) was used for training until the same number of epochs. In addition, there are no batch normalization and dropout techniques applied as well. Although the loss function plots of the two similarly show a descending trend, they differ in the location where loss values plateau. For the first problem it took place at around the $300$th iteration in contrast to the second at around the $200$th.

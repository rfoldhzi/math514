import torch
import plotly.graph_objects as go
import numpy as np
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Batch Size, Input Neurons, Hidden Neurons, Output Neurons
N, D_in, H, D_out = 16, 5, 1024, 1

# Create random Tensors to hold inputs and outputs
x0 = [[float(i)] for i in range(16)]
y0 = [[x[0]**2] for x in x0]
x = torch.randn(N, D_in)
print(x)
y = torch.randn(N, D_out)

with open('xMatrix3.npy', 'rb') as f:
    numpyX = np.load(f)
with open('yMatrix3.npy', 'rb') as f:
    numpyY = np.load(f)[:, [2]]

with open('xMatrix2.npy', 'rb') as f:
    numpyXTest = np.load(f)
with open('yMatrix2.npy', 'rb') as f:
    numpyYTest = np.load(f)[:, [2]]

print(numpyX)

x = torch.from_numpy(numpyX).to(torch.float32)
x = x.to(device)
print(y)
y = torch.from_numpy(numpyY).to(torch.float32)
y = y.to(device)

xTest = torch.from_numpy(numpyXTest).to(torch.float32)
xTest = xTest.to(device)
print(y)
yTest = torch.from_numpy(numpyYTest).to(torch.float32)
yTest = yTest.to(device)

# Use the nn package to define our model
# Linear (Input -> Hidden), ReLU (Non-linearity), Linear (Hidden-> Output)
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, 4096),
    torch.nn.Tanh(),
    torch.nn.Linear(4096, H),
    torch.nn.BatchNorm1d(H),
    torch.nn.Tanh(),
    torch.nn.Linear(H, H),
    torch.nn.BatchNorm1d(H),
    torch.nn.Tanh(),
    torch.nn.Linear(H, H),
    torch.nn.BatchNorm1d(H),
    torch.nn.Tanh(),
    torch.nn.Linear(H, H),
    torch.nn.BatchNorm1d(H),
    torch.nn.Tanh(),
    torch.nn.Linear(H, D_out),
)
model.to(device)

model.load_state_dict(torch.load("modelTanhLargeDataC.pt"))
model.eval()



# Define the loss function: Mean Squared Error
# The sum of the squares of the differences between prediction and ground truth
loss_fn = torch.nn.MSELoss(reduction='sum')

# The optimizer does a lot of the work of actually calculating gradients and
# applying backpropagation through the network to update weights
learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def calcLossOfTest():
    y_pred = model(xTest)
    loss2 = loss_fn(y_pred, yTest)
    print("Test Loss", loss2.item())

startTime = time.time()
NumberOfIterations = 30000
# Perform 30000 training steps
for t in range(NumberOfIterations):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute loss and print it periodically
    loss = loss_fn(y_pred, y)
    if t % 100 == 0:
        print(t, loss.item())
        calcLossOfTest()
        if t != 0:
            timePerIter = (time.time()-startTime)/t
            iterRemaining = NumberOfIterations-t
            timeRemaining = timePerIter*iterRemaining
            print("%s Est Time: %s %.2f" % (bcolors.WARNING, bcolors.ENDC, timeRemaining))

    # Update the network weights using gradient of the loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for param in model.parameters():
    print(type(param), param.size(), param)

torch.save(model.state_dict(), "modelTanhLargeDataC.pt")

# Generate predictions for evenly spaced x-values between minx and maxx
# minx = min(list(numpyX[:, [0]]))
# maxx = max(list(numpyX[:, [0]]))
# newX = np.copy(x)
# numpyX[:, [0]] = np.linspace(minx, maxx, num=400)
# c = torch.from_numpy(np.linspace(minx, maxx, num=400)).reshape(-1, 1).float()

# d = model(torch.from_numpy(numpyX).to(torch.float32))

# # Draw the predicted functions as a line graph
# fig.add_trace(go.Scatter(x=c.flatten().numpy(), y=d.flatten().detach().numpy(), mode="lines"))
# fig.show()
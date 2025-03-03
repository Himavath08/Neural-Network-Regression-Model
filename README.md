# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

![image](https://github.com/user-attachments/assets/2dbc75fe-fb81-4299-8474-ea7d95dd4b84)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:
```
# Name:Himavath M
# Register Number:212223240053
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,10)
        self.fc2=nn.Linear(10,14)
        self.fc3=nn.Linear(14,1)
        self.relu=nn.ReLU()
        self.history = {'loss': []}


  def forward(self,x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.fc3(x)
    return x

# Initialize the Model, Loss Function, and Optimize
ai_brain=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = ai_brain(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

```
## Dataset Information
![image](https://github.com/user-attachments/assets/b0030366-08c7-4e5f-9d4f-5ab661c485c0)



## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/9167ee97-6f87-468a-9faa-917e2705c20e)


### New Sample Data Prediction
![image](https://github.com/user-attachments/assets/c71854d0-7b3a-4c62-a607-af7e755ee13e)


## RESULT

Thus, a neural network regression model for the given dataset is successfully developed.

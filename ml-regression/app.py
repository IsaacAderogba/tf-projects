# 1) imports
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd


def render_header():
    st.title("1. Regression with PyTorch")


# 2) Get number of samples, features and targets
def generate_and_plot_data():
    N = st.sidebar.selectbox(
        "Number of samples", tuple(i + 1 for i in range(100)), 19)

    X = np.random.random(N) * 10 - 5
    Y = 0.5 * X - 1 + np.random.randn(N)

    plt.scatter(X, Y)
    st.subheader("Scatterplot")
    st.pyplot()

    return N, X, Y


# 3) Create model, criterion and optimizer
def create_regression_model():
    lr = st.sidebar.selectbox(
        "Learning rate", tuple((i + 1) / 100 for i in range(100)), 5)
    model = nn.Linear(1, 1)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    return model, criterion, optimizer


# 4) Reshape the numpy data to be a tensor
def reshape_data(N, X, Y):
    X = X.reshape(N, 1)
    Y = Y.reshape(N, 1)

    inputs = torch.from_numpy(X.astype(np.float32()))
    targets = torch.from_numpy(Y.astype(np.float32()))
    return inputs, targets


# 5) train the model
def train_model(inputs, targets, model, criterion, optimizer):
    num_epochs = st.sidebar.selectbox(
        "Number of epochs", tuple(i + 1 for i in range(100)), 29)
    losses = []

    for it in range(num_epochs):
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        st.text(f'Epcoh {it+1}/{num_epochs}, Loss: {loss.item(): .4f}')

    predicted = model(inputs).detach().numpy()
    weights = model.weight.data.item()
    biases = model.bias.data.item()
    return losses, predicted, weights, biases


# 6) Plot the losses
def plot_losses(losses):
    st.subheader("Losses")
    plt.plot(losses)
    st.pyplot()

# 7) Plot the predictions


def plot_predictions(X, Y, predicted):
    st.subheader("Predictions")
    plt.scatter(X, Y, label="Original data")
    plt.plot(X, predicted, label="Fitted line")
    plt.legend()
    plt.show()
    st.pyplot()


# 8) Evaluate the model
def evaluate_model(weights, biases):
    st.subheader("Evaluation")
    df = pd.DataFrame([{weights, biases}, {0.5, -1}],
                      columns=("Weights", "Biases"))
    df.rename(index={0: "Predicted", 1: "Actual"}, inplace=True)
    st.table(df)


def main():
    render_header()

    N, X, Y = generate_and_plot_data()

    inputs, targets = reshape_data(N, X, Y)

    model, criterion, optimizer = create_regression_model()

    losses, predicted, weights, biases = train_model(
        inputs, targets, model, criterion, optimizer)

    plot_losses(losses)
    plot_predictions(X, Y, predicted)
    evaluate_model(weights, biases)


main()

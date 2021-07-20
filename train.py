from os import path
import torch
from torch.utils.data import DataLoader, random_split

from model import ConvNet, OneDimConv
import dataset


SAVE_DIR = './data/control_data'
MODEL_SAVE_DIR = './data/models'
MODEL_SAVE_NAME = 'conv_model_1.pth'

# For reproducability
torch.manual_seed(4)

# Create train and test splits from the custom dataset
train_dataset, test_dataset = random_split(
    dataset.SynthSoundsDataset(SAVE_DIR), [29000, 1000]
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

LEARNING_RATE = 0.01
NUM_EPOCHS = 100
SAMPLE_DIFF_TOLERANCE = 0.04

conv_model = OneDimConv()
# conv_model = ConvNet()

# Loss and optimizer
# Mean squared error loss function, Adam optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(conv_model.parameters(), lr=LEARNING_RATE)
# Create a learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, .5)

def train_loop():
    # Train the model
    # Keep lists of the loss to track progress
    loss_list = []
    test_loss_list = []
    print('Training Started')
    for epoch in range(NUM_EPOCHS):
        conv_model.train() # Set model to trainable gradients
        loss_epoch = 0
        print(f'Epoch {epoch}')
        for i, (control_labels, sample) in enumerate(train_loader):
            # Run the forward pass
            outputs = conv_model(sample)
            loss = criterion(outputs, control_labels)
            loss_epoch += loss.item()
            # Backprop and perform Adam optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i % 100 == 0):
                print(f'Batch #{i}')

        scheduler.step()
        loss_list.append(loss_epoch)

        print('Testing Model')
        conv_model.eval()
        test_losses = 0
        correct = 0
        total_test_num = len(test_loader)
        with torch.no_grad(): # Disable gradient calculation
            for i, (control_labels, sample) in enumerate(test_loader):
                # Run the forward pass
                outputs = conv_model(sample)
                loss = criterion(outputs, control_labels)
                loss_scalar = loss.item()
                if (loss_scalar <= SAMPLE_DIFF_TOLERANCE): # Measure of correctness
                    correct += 1
                test_losses += loss_scalar
                if (i % 100 == 0):
                    print(loss_scalar)
                    print(control_labels - outputs)

        test_loss_list.append(test_losses / total_test_num)
        print('Total Loss History', loss_list)
        print(f'Test Loss Total: {test_losses}')
        print(f'% Tolerable: {round(correct / total_test_num * 100)}')
        print('Average Test Loss History', test_loss_list)

        # Stop training and don't save new state if the test loss increases (early stopping)
        if (len(test_loss_list) >= 3 and test_loss_list[-1] > test_loss_list[-3]):
            print('Stopping training due to test loss increase')
            return

        torch.save(conv_model.state_dict(), path.join(MODEL_SAVE_DIR, MODEL_SAVE_NAME))

train_loop()
"""Example of neural networks on MNIST."""
import torchvision
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import logging
logger = logging.getLogger(__name__)


class LeNet5(nn.Module):
    # require 32*32 pixels
    def __init__(self, n_classes=10):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        # Since NLL or KL divergence uses log-probability as input, we need to use log_softmax
        probs = F.log_softmax(logits, dim=1)
        return probs


class LogisticRegression(nn.Module):
    # require 32*32 pixels
    def __init__(self, in_features=120, n_classes=2):
        super(LogisticRegression, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=n_classes),
        )

    def forward(self, x):
        logits = self.classifier(x)
        # Since NLL or KL divergence uses log-probability as input, we need to use log_softmax
        probs = F.log_softmax(logits, dim=1)
        return probs


def train(model, train_loader, criterion, optimizer, device):
    """Training for one epoch."""
    train_loss = 0
    model.train(True)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # Initialize grad
        output = model(data.to(device))  # feed model
        loss = criterion(output.squeeze(), target.squeeze().to(device))   # calculate loss
        loss.backward()  # back propagation
        optimizer.step()  # update parameters
        train_loss += loss.item()  # sum up training loss
    return train_loss / len(train_loader)


def test(model, test_loader, criterion, device):
    """Test error on the test dataset given a trained model."""
    model.train(False)
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.to(device))
            test_loss += criterion(output.squeeze(), target.squeeze().to(device)).item()  # sum up batch loss
    test_loss = test_loss / len(test_loader)
    return test_loss


def myCustomLoss(my_outputs, my_labels):
    '''Misclassification error.'''
    # specifying the batch size
    my_batch_size = my_outputs.size()[0]

    if my_outputs.dim() == 2:
        my_outputs = torch.argmax(my_outputs, dim=1)
    if my_labels.dim() == 2:
        my_labels = torch.argmax(my_labels, dim=1)
    # returning the results
    return torch.sum(my_outputs != my_labels)/my_batch_size


def CELoss(my_outputs, targets):
    """Cross entropy loss or negative log likelihood, automatically chosen based on the hard/soft label."""
    if my_outputs.shape == targets.shape:
        return nn.KLDivLoss(reduction='batchmean')(my_outputs, targets)
    else:
        return nn.NLLLoss()(my_outputs, targets)


def run(train_loader, test_loader, model, criterion=CELoss, criterion2=myCustomLoss, lr=0.001, num_epochs=100,
        device='cpu'):
    """Model training procedure."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.5)
    qtl = max(num_epochs // 10, 1)
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, device=device)
        test_loss = test(model, test_loader, criterion2, device=device)
        scheduler.step()
        if epoch % qtl == 0:
            logger.debug('Train({})[{:.0f}%]: Loss: {:.4f}; Test error:{:.4f}'.format(
                epoch, 100. * epoch / num_epochs, train_loss, test_loss))

    return train_loss, test_loss, model


if __name__ == '__main__':
    pass

import torch
from tqdm import tqdm

# Let's visualize some of the images
import matplotlib.pyplot as plt
import numpy as np

from gradcam import VisualizeCam

class DataModel(object):
  def __init__(self, image_data, criterion, optimizer, schedular, num_of_epochs = 10, cal_misclassified = False, expected_accuracy = 95):
    super(DataModel, self).__init__()
    self.train_losses = []
    self.train_acc = []
    self.test_losses = []
    self.test_acc = []
    self.misclassified = []
    self.cal_misclassified = cal_misclassified
    self.EPOCHS = num_of_epochs
    self.can_exit = False
    self.model = None
    self.img_data = image_data
    self.expected_accuracy = expected_accuracy
    self.criterion = criterion
    self.optimizer = optimizer
    self.schedular = schedular

  def train(self, device, train_loader, epoch):
    self.model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
      # get samples
      data, target = data.to(device), target.to(device)

      # Init
      self.optimizer.zero_grad()
      # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
      # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

      # Predict
      y_pred = self.model(data)

      loss = self.criterion(y_pred, target)
      self.train_losses.append(loss)

      # Backpropagation
      loss.backward()
      self.optimizer.step()

      # Update pbar-tqdm
      pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()
      processed += len(data)
      pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    self.train_acc.append(100. * correct/processed)

  def test(self, device, test_loader):
      self.model.eval()
      test_loss = 0
      correct = 0
      with torch.no_grad():
          for data, target in test_loader:
              data, target = data.to(device), target.to(device)
              output = self.model(data)

              loss = self.criterion(output, target)

              test_loss += loss.item()  # sum up batch loss
              pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
              if self.cal_misclassified == True:
                for i in range(len(pred)):
                    if pred[i] != target[i] and len(self.misclassified) < 25:
                        self.misclassified.append([data[i], pred[i], target[i]])
              correct += pred.eq(target.view_as(pred)).sum().item()

      test_loss /= len(test_loader.dataset)
      self.test_losses.append(test_loss)

      accuracy = 100. * correct / len(test_loader.dataset)
      print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
          test_loss, correct, len(test_loader.dataset), accuracy))
      
      self.schedular.step(test_loss)

      self.test_acc.append(accuracy)
      if accuracy > self.expected_accuracy:
        self.can_exit = True
      
  def run_model(self, net_model, device):
    self.model = net_model.to(device)

    for epoch in range(self.EPOCHS):
        if self.can_exit:
          print("****Required Accuracy is acheived****")
          break
        print("EPOCH:", epoch + 1)
        self.misclassified = []
        self.train(device, self.img_data.trainloader, epoch)
        self.test(device, self.img_data.testloader)

  def plot_matrix(self, matrix_data, matrix):
      fig = plt.figure(figsize=(10, 10))
      
      plt.title(matrix)
      plt.xlabel('Epoch')
      plt.ylabel(matrix)

      plt_tuple = ()
      legend_tuple = ()
  
      plt_tuple = plt_tuple + (plt.plot(matrix_data)[0], )
      legend_tuple = legend_tuple

      plt.legend(plt_tuple, legend_tuple)

      fig.savefig(f'val_%s_change.png' % (matrix.lower()))

  def plot_test_train_accuracy(self):
      fig = plt.figure(figsize=(10, 10))
      host = fig.add_subplot(111)

      par1 = host.twinx()

      host.set_xlabel("EPOCHS")
      host.set_ylabel("Train Accuracy")
      par1.set_ylabel("Test Accuracy")


      color1 = plt.cm.viridis(0)
      color2 = plt.cm.viridis(0.5)

      print(self.train_acc)
      print(self.test_acc)
      p1, = (host.plot(self.train_acc, color=color1, label="Train Accuracy")[0], )
      p2, = (par1.plot(self.test_acc, color=color2, label="Test Accuracy")[0], )

      lns = [p1, p2]
      host.legend(handles=lns, loc='best')

      host.yaxis.label.set_color(color1)
      par1.yaxis.label.set_color(color2)

      plt.savefig(f'val_%s_change.png' % ("traintestaccuracy"), bbox_inches='tight')

  def plot_loss_accuracy(self):
      fig = plt.figure(figsize=(10, 10))
      host = fig.add_subplot(111)

      par1 = host.twinx()

      host.set_xlabel("Distance")
      host.set_ylabel("Loss Graph")
      par1.set_ylabel("Test Accuracy")


      color1 = plt.cm.viridis(0)
      color2 = plt.cm.viridis(0.5)

      p1, = (host.plot(self.test_losses, color=color1, label="Loss Graph")[0], )
      p2, = (par1.plot(self.test_acc, color=color2, label="Test Accuracy")[0], )

      lns = [p1, p2]
      host.legend(handles=lns, loc='best')

      host.yaxis.label.set_color(color1)
      par1.yaxis.label.set_color(color2)

      plt.savefig(f'val_%s_change.png' % ("lossaccuracy"), bbox_inches='tight')

  def plot_misclassified(self):
    fig = plt.figure(figsize = (10,10))

    for i in range(len(self.misclassified)):
        sub = fig.add_subplot(5, 5, i+1)
        misclassified_transpose = np.transpose(self.misclassified[i][0].cpu().numpy(), (1, 2, 0))
        plt.imshow(misclassified_transpose.squeeze(),cmap='gray',interpolation='none')
        
        sub.set_title("Pred={}, Act={}".format(str(self.img_data.classes[self.misclassified[i][1].data]),str(self.img_data.classes[self.misclassified[i][2].data])))
        
    plt.tight_layout()

    plt.show()

  def plot_loss_graph(self):
    self.plot_matrix(self.test_losses, "Loss Graph")

  def plot_test_accuracy_graph(self):
    self.plot_matrix(self.test_acc, "Validation Accuracy")

  def plot_train_accuracy_graph(self):
    self.plot_matrix(self.train_acc, "Validation Accuracy")
  
  def plot_GRADcam(self, target_layers):
    viz_cam = VisualizeCam(self.model, self.img_data.classes, target_layers)

    num_img = len(self.misclassified)
    incorrect_pred_imgs = []
    image_for_gradcam = []
    for i in range(num_img):
      incorrect_pred_imgs.append(torch.as_tensor(self.misclassified[i][0]))
      image_for_gradcam.append(self.misclassified[i])
      if i == 25:
          break
    viz_cam(torch.stack(incorrect_pred_imgs), image_for_gradcam, target_layers, metric="incorrect")
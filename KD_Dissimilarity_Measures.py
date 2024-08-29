import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import random
import numpy as np
import json
import time
import sys
import os

seed = 42

# Set seed for Python's random number generator
random.seed(seed)

# Set seed for NumPy
np.random.seed(seed)

# Set seed for PyTorch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define custom ResNet model with two heads
class CustomResNet(nn.Module):
    def __init__(self, base_model):
        super(CustomResNet, self).__init__()
        self.base = nn.Sequential(*list(base_model.children())[:-1])  # Exclude the last FC layer
        self.fc1 = nn.Linear(base_model.fc.in_features, 10)  # First head
        self.hidden1 = nn.Linear(base_model.fc.in_features, 512)  # First hidden layer for second head
        self.hidden2 = nn.Linear(512, 256)  # Second hidden layer for second head
        self.fc2 = nn.Linear(256, 10)  # Second head output layer

    def forward(self, x):
        x = self.base(x)
        x = torch.flatten(x, 1)
        out1 = self.fc1(x)
        hidden = F.relu(self.hidden1(x))
        hidden = F.relu(self.hidden2(hidden))
        out2 = self.fc2(hidden)
        return out1, out2








# Define distillation methods and other utility functions
def jensen_shannon_divergence(p, q):
    sumPQ = torch.add(p, q)
    tensor_mean = torch.mul(sumPQ, 0.5)
    comp1 = nn.KLDivLoss(reduction='sum',log_target=False)(tensor_mean.log(),p)
    comp2 = nn.KLDivLoss(reduction='sum',log_target=False)(tensor_mean.log(),q)
    sum_comp = comp1 + comp2
    jsd = torch.mul(sum_comp, 0.5)
    return jsd



def triangular_distance(x, y):
    
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if y.dim() == 1:
        y = y.unsqueeze(0)

    combined = x + y
    combined = torch.where(combined == 0, torch.ones_like(combined), combined)
    product = 2 * x * y
    distance = 1 - torch.sum(product / (combined),dim=1)
    return torch.sum(distance)

def compute_entropies(tensor):
    tensor = torch.where(tensor == 0, torch.ones_like(tensor), tensor)
    tensor_log = torch.log(tensor)
    tensor_mul = tensor * tensor_log
    entropy = -torch.sum(tensor_mul, dim=1)
    return entropy

def compute_complexity(tensor):
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    
    entropies = compute_entropies(tensor)
    complexity = torch.exp(entropies)
    return complexity

def compute_sed(tensor1, tensor2):
    
    complexity1 = compute_complexity(tensor1)
    complexity2 = compute_complexity(tensor2)
    tensor_mean = (tensor1 + tensor2) / 2
    complexity12 = compute_complexity(tensor_mean)
    geometric_mean = torch.sqrt(complexity1 * complexity2)
    
    
    sed = torch.sum((complexity12 / geometric_mean) - 1)
    return sed

def compute_msed(tensors):
    complexities = [compute_complexity(tensor) for tensor in tensors]
    mean_tensor = torch.mean(torch.stack(tensors), 0)
    complexity_mean = compute_complexity(mean_tensor)
    geometric_mean = torch.prod(torch.stack(complexities), 0) ** (1 / len(complexities))
    msed = (1 / (len(complexities) - 1)) * (torch.sum(complexity_mean / geometric_mean-1) )
    return msed







# Load the value of temperature from JSON

alphas = [ 0,0.2,0.5,0.8]
currentClient = 1
remoteClients = [2, 3]

dataset = "iid_cifar"
dosum_values = [True,False]

with open('alpha3.json') as f:
    data = json.load(f)
    temperatures = data['temperature']
    modes = data['mode']
    #dosum_values = data['dosum']
    for dosum in dosum_values:
        if dosum == True:
            distillation_case = 'sum'
        else:
            distillation_case = 'avg'
        
        for mode in modes:
            print(f'******Mode******: {mode}')
            print()
            for temperature in temperatures:
                print(f'******Temperature******: {temperature}')
                for alpha in alphas:
                    print(f'******Alpha******: {alpha}')
                    # Define transformations
                    transform_train = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

                    transform_val = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

                    # Load datasets
                    trainset_head1 = torchvision.datasets.ImageFolder(root=f'data/cifar10_{currentClient}/train', transform=transform_train)
                    valset_head1 = torchvision.datasets.ImageFolder(root=f'data/cifar10_{currentClient}/val', transform=transform_val)
                    trainset_head2 = torchvision.datasets.ImageFolder(root=f'data/cifar10_{currentClient}/train', transform=transform_train)
                    valset_head2 = torchvision.datasets.ImageFolder(root=f'data/cifar10_{currentClient}/val', transform=transform_val)

                    # Define batch size and workers
                    batch_size = 128
                    worker = 6

                    # Create data loaders
                    trainloader_head1 = DataLoader(trainset_head1, batch_size=batch_size, shuffle=True, num_workers=worker)
                    valloader_head1 = DataLoader(valset_head1, batch_size=batch_size, shuffle=False, num_workers=worker)
                    trainloader_head2 = DataLoader(trainset_head2, batch_size=batch_size, shuffle=True, num_workers=worker)
                    valloader_head2 = DataLoader(valset_head2, batch_size=batch_size, shuffle=False, num_workers=worker)

                    # Define student model
                    base_model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
                    student_model = CustomResNet(base_model)
                    student_model.to(device)

                    # Define loss function
                    criterion = nn.CrossEntropyLoss()

                    # Define optimizer and scheduler
                    optimizer = optim.SGD(student_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

                    # Define training parameters
                    num_epochs = 50
                    patience = 5
                    best_val_loss = float('inf')
                    no_improvement_count = 0

                
                    # Train the first head if model does not exist
                    if not os.path.exists(f'./{dataset}/first_head_client{currentClient}.pth'):
                        for epoch in range(num_epochs):
                            student_model.train()
                            running_train_loss = 0.0
                            correct_train = 0
                            total_train = 0
                            with tqdm(trainloader_head1, unit="batch") as tepoch:
                                for inputs, labels in tepoch:
                                    inputs, labels = inputs.to(device), labels.to(device)
                                    optimizer.zero_grad()
                                    outputs1, _ = student_model(inputs)
                                    loss1 = criterion(outputs1, labels)
                                    loss1.backward()
                                    optimizer.step()
                                    running_train_loss += loss1.item()
                                    _, predicted1 = torch.max(outputs1, 1)
                                    total_train += labels.size(0)
                                    correct_train += (predicted1 == labels).sum().item()
                                    tepoch.set_postfix(loss=running_train_loss / total_train, accuracy=100. * correct_train / total_train)
                            scheduler.step()
                            student_model.eval()
                            running_val_loss = 0.0
                            correct_val = 0
                            total_val = 0
                            with torch.no_grad():
                                for inputs, labels in valloader_head1:
                                    inputs, labels = inputs.to(device), labels.to(device)
                                    outputs1, _ = student_model(inputs)
                                    loss1 = criterion(outputs1, labels)
                                    running_val_loss += loss1.item()
                                    _, predicted1 = torch.max(outputs1, 1)
                                    total_val += labels.size(0)
                                    correct_val += (predicted1 == labels).sum().item()
                            avg_val_loss = running_val_loss / len(valloader_head1)
                            val_accuracy = 100 * correct_val / total_val
                            print(f"Epoch [{epoch + 1}/{num_epochs}] - Head 1, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
                            if avg_val_loss < best_val_loss:
                                best_val_loss = avg_val_loss
                                no_improvement_count = 0
                                torch.save({'model_state_dict': student_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'./{dataset}/first_head_client{currentClient}.pth')
                                print('Saving the best model for Head 1')
                            else:
                                no_improvement_count += 1
                                if no_improvement_count >= patience:
                                    print(f"Validation loss has not improved for {patience} epochs. Stopping early.")
                                    break
                        print('Training complete for the first head!')
                        
                    else:
                        
                
                        
                        # Function to define and load student_model_clone
                        def load_student_model_clone(client_number):
                            base_model_clone = models.resnet18(weights=None)
                            model_clone = CustomResNet(base_model_clone)
                            model_clone.to(device)
                            checkpoint = torch.load(f'./{dataset}/first_head_client{client_number}.pth')
                            model_clone.load_state_dict(checkpoint['model_state_dict'])
                            return model_clone

                        # Load all remote clients
                        remote_clients = [load_student_model_clone(client_number) for client_number in remoteClients]

                        checkpoint = torch.load(f'./{dataset}/first_head_client{currentClient}.pth')
                        student_model.load_state_dict(checkpoint['model_state_dict'])
                        #print the structure of the model
                        #print(student_model)
                        #sys.exit()

                        

                        # Define a new optimizer for training the second head
                        optimizer = optim.SGD(filter(lambda p: p.requires_grad, student_model.parameters()), lr=0.001, momentum=0.9, weight_decay=5e-4)
                        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

                        # Train the second head
                        for epoch in range(num_epochs):
                            student_model.train()
                            running_train_loss = 0.0
                            correct_train = 0
                            total_train = 0
                            with tqdm(trainloader_head2, unit="batch") as tepoch:
                                for inputs, labels in tepoch:
                                    inputs, labels = inputs.to(device), labels.to(device)
                                    optimizer.zero_grad()

                                    
                                    #Compute forward pass for remote clients
                                    with torch.no_grad():
                                        remote_output1,_ = remote_clients[0](inputs)
                                        remote_output2,_ = remote_clients[1](inputs)
                                        remote_output1 = remote_output1.to(device)
                                        remote_output2 = remote_output2.to(device)
                                        
                                        if dosum != True:
                                            """for Average CE uncomment this"""
                                            remote_output = (remote_output1 + remote_output2) / 2
                                            remote_output = remote_output.to(device)

                                    _, outputs2 = student_model(inputs)
                                    loss2 = criterion(outputs2, labels)

                                    
                                    

                                    if mode == 'CE':
                                        if dosum==True:
                                            # Compute distillation loss
                                            distillation_loss1 = nn.CrossEntropyLoss(reduction='sum')(outputs2/temperature, nn.functional.softmax(remote_output1/temperature, dim=1))
                                            distillation_loss2 = nn.CrossEntropyLoss(reduction='sum')(outputs2/temperature, nn.functional.softmax(remote_output2/temperature, dim=1))
                                            distillation_loss = (distillation_loss1 + distillation_loss2)

                                    
                                        else:
                                            distillation_loss = nn.CrossEntropyLoss(reduction='sum')(outputs2 / temperature, nn.functional.softmax(remote_output / temperature, dim=1))
                                            
                                    elif mode == 'KL':
                                        if dosum==True:
                                            #compute the sum of the KL divergence
                                            distillation_loss1 = nn.KLDivLoss(reduction='sum',log_target=False)(nn.functional.softmax(outputs2 / temperature, dim=1).log(), nn.functional.softmax(remote_output1 / temperature, dim=1))
                                            distillation_loss2 = nn.KLDivLoss(reduction='sum',log_target=False)(nn.functional.softmax(outputs2 / temperature, dim=1).log(), nn.functional.softmax(remote_output2 / temperature, dim=1))
                                            distillation_loss = (distillation_loss1 + distillation_loss2)
                                        else:
                                            distillation_loss = nn.KLDivLoss(reduction='sum',log_target=False)(nn.functional.softmax(outputs2 / temperature, dim=1).log(), nn.functional.softmax(remote_output / temperature, dim=1))                                      
                                    elif mode == 'JS':
                                        if dosum==True:
                                            #compute the sum of the JS divergence
                                            distillation_loss1 = jensen_shannon_divergence(nn.functional.softmax(outputs2 / temperature, dim=1), nn.functional.softmax(remote_output1 / temperature, dim=1))
                                            distillation_loss2 = jensen_shannon_divergence(nn.functional.softmax(outputs2 / temperature, dim=1), nn.functional.softmax(remote_output2 / temperature, dim=1))
                                            distillation_loss = (distillation_loss1 + distillation_loss2)
                                        else:
                                            distillation_loss = jensen_shannon_divergence(nn.functional.softmax(outputs2 / temperature, dim=1), nn.functional.softmax(remote_output / temperature, dim=1))
                                    elif mode == 'TD':
                                        if dosum==True:
                                            #compute the sum of the Triangular distance
                                            distillation_loss1 = triangular_distance(nn.functional.softmax(outputs2 / temperature, dim=1), nn.functional.softmax(remote_output1 / temperature, dim=1))
                                            distillation_loss2 = triangular_distance(nn.functional.softmax(outputs2 / temperature, dim=1), nn.functional.softmax(remote_output2 / temperature, dim=1))
                                            distillation_loss = (distillation_loss1 + distillation_loss2)
                                        else:
                                            distillation_loss = triangular_distance(nn.functional.softmax(outputs2 / temperature, dim=1), nn.functional.softmax(remote_output / temperature, dim=1))
                                    elif mode == 'SED':
                                        if dosum==True:
                                            #compute the sum of the SED
                                            distillation_loss1 = compute_sed(nn.functional.softmax(outputs2 / temperature, dim=1), nn.functional.softmax(remote_output1 / temperature, dim=1))
                                            distillation_loss2 = compute_sed(nn.functional.softmax(outputs2 / temperature, dim=1), nn.functional.softmax(remote_output2 / temperature, dim=1))
                                            distillation_loss = (distillation_loss1 + distillation_loss2)
                                        else:
                                            distillation_loss = compute_sed(nn.functional.softmax(outputs2 / temperature, dim=1), nn.functional.softmax(remote_output / temperature, dim=1))
                                    
                                    else:
                                        raise ValueError(f"Unknown mode: {mode}")
                                    loss2 = alpha * loss2 + (1 - alpha) * distillation_loss



                                    loss2.backward()
                                    optimizer.step()
                                    running_train_loss += loss2.item()
                                    _, predicted2 = torch.max(outputs2, 1)
                                    total_train += labels.size(0)
                                    correct_train += (predicted2 == labels).sum().item()
                                    tepoch.set_postfix(loss=running_train_loss / total_train, accuracy=100. * correct_train / total_train)


                            scheduler.step()
                            student_model.eval()
                            running_val_loss = 0.0
                            correct_val = 0
                            total_val = 0
                            with torch.no_grad():
                                for inputs, labels in valloader_head2:
                                    inputs, labels = inputs.to(device), labels.to(device)
                                    _, outputs2 = student_model(inputs)
                                    loss2 = criterion(outputs2, labels)
                                    running_val_loss += loss2.item()
                                    _, predicted2 = torch.max(outputs2, 1)
                                    total_val += labels.size(0)
                                    correct_val += (predicted2 == labels).sum().item()
                            avg_val_loss = running_val_loss / len(valloader_head2)
                            val_accuracy = 100 * correct_val / total_val
                            print(f"Epoch [{epoch + 1}/{num_epochs}] - Head 2, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
                            if avg_val_loss < best_val_loss:
                                best_val_loss = avg_val_loss
                                no_improvement_count = 0
                                torch.save({'model_state_dict': student_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'./{dataset}/clients_{distillation_case}/second_head_client{currentClient}_alpha{alpha}_temp{temperature}_{mode}.pth')
                                print('Saving the best model for Head 2')
                            else:
                                no_improvement_count += 1
                                if no_improvement_count >= patience:
                                    print(f"Validation loss has not improved for {patience} epochs. Stopping early.")
                                    break
                        print('Training complete for the second head!')

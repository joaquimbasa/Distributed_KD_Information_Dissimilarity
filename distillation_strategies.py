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
import csv
import scipy.spatial.distance
from scipy.spatial import distance
from scipy.stats import entropy

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


class CustomResNet(nn.Module):
    def __init__(self, base_model):
        super(CustomResNet, self).__init__()
        self.base = nn.Sequential(*list(base_model.children())[:-1])  # Exclude the last FC layer
        self.fc1 = nn.Linear(base_model.fc.in_features, 10)  # First head
        self.fc2 = nn.Linear(base_model.fc.in_features, 10)  # Second head

    def forward(self, x):
        # Forward pass through the base model
        x = self.base(x)
        x = torch.flatten(x, 1)  # Flatten the output to feed into the FC layers
        
        # Compute outputs for both heads
        out1 = self.fc1(x)
        out2 = self.fc2(x)

        return out1, out2



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


def msed(tensors):
    n = len(tensors)
    
    # Step 1: Compute the average tensor (mean over all tensors)
    avg_tensor = torch.mean(torch.stack(tensors), dim=0)
    
    # Step 2: Compute the entropy of the average tensor
    entropy_avg = compute_entropies(avg_tensor)
    
    # Step 3: Compute the entropies of individual tensors and their average
    individual_entropies = torch.stack([compute_entropies(tensor) for tensor in tensors])
    avg_individual_entropy = torch.mean(individual_entropies, dim=0)
    
    # Step 4: Apply the MSED formula
    numerator = entropy_avg - avg_individual_entropy
    msed_value = torch.exp(numerator) - 1
    
    # Step 5: Scale by 1 / (n - 1)
    msed_value = msed_value / (n - 1)
    
    return torch.sum(msed_value)

#Define a cross entropy loss function
def cross_entropy_loss(p, q):
    q = torch.where(q == 0, torch.ones_like(q), q)
    return torch.sum(p * torch.log(q))


# implement a function to compute KL divergence and use - instead of / to avoid nan values
"""def kl_divergence(p, q):
    new_p = torch.where(p == 0, torch.ones_like(p), p)
    plogp= torch.sum(new_p * torch.log(new_p))
    return plogp - cross_entropy_loss(p, q)"""

#all new functions
def shannon_entropy(tensor, dim=1):
  '''
  Compute shannon entropy from scratch
  '''
  #tensor = torch.where(tensor == 0, torch.ones_like(tensor), tensor)
  entropy = -torch.sum(tensor * torch.log(tensor), dim=dim)
  return entropy


def compute_kl_torch(logit_target, logit_input, reduction='sum'):
   # p=target
   # q=input
    target_prob=nn.functional.softmax(logit_target, dim=1)
    input=nn.functional.log_softmax(logit_input, dim=1)
    return nn.KLDivLoss(reduction=reduction)(input,target_prob)

"""def jensen_shannon_divergence(p,q,dim=1):
  tensor_mean=(p+q)/2.0
  entropy_mean=shannon_entropy(tensor_mean, dim=dim)
  mean_of_entropy= (shannon_entropy(p,dim=dim)+shannon_entropy(q,dim=dim))/2.0
  jsd= entropy_mean - mean_of_entropy
  #print(jsd)
  
  return torch.sum(jsd)"""

def sed(tensor1, tensor2):
    tensor_mean = torch.mul(tensor1 + tensor2, 0.5)
    entropy_tens_mean = compute_entropies(tensor_mean)
    mean_entropies = torch.mul(compute_entropies(tensor1) + compute_entropies(tensor2), 0.5)
    sed = torch.sum(torch.exp(entropy_tens_mean - mean_entropies) - 1)
    return sed

# Load the value of temperature from JSON

alphas = [0.5]
num_clients = 8
destinationClient = 8

#456
currentClients = [0,1, 2, 3, 4, 5, 6, 7]

topology = "full"
#remoteClients = [0, 1, 2, 3,4, 5, 6, 7]

dataset = "cifar10"
dosum_values = [False]
# iteration = 3

for iteration in range(1, 11):
# Assuming the necessary methods (like distillation loss computation) and classes (like CustomResNet) are already defined.
    for currentClient in currentClients:
        print(f'******Client******: {currentClient}')
        if currentClient == 0:
            remoteClients = [1,2,5,4,7,3,6,currentClient]
        elif currentClient == 1:
            remoteClients = [0,4,7,3,6,2,5,currentClient]
        elif currentClient == 2:
            remoteClients = [0,5,3,6,4,1,7,currentClient]
        elif currentClient == 3:
            remoteClients = [7,6,2,1,5,0,4,currentClient]
        elif currentClient == 4:
            remoteClients = [1, 5, 6, 0,2,3,7,currentClient]
        elif currentClient == 5:
            remoteClients = [2,4,0,7,3,6,1,currentClient]
        elif currentClient == 6:
            remoteClients = [3,4,7,2,1,5,0,currentClient]
        elif currentClient == 7:
            remoteClients = [3,1,6,5,0,2,4,currentClient]

        # Load configuration from alpha3.json
        with open('alpha3.json') as f:
            data = json.load(f)
            temperatures = data['temperature']
            modes = data['mode']
        
        

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
                            trainset_head1 = torchvision.datasets.ImageFolder(root=f'.data/niid_{dataset}_S100_C{num_clients}/train/client_{currentClient}', transform=transform_train)
                            valset_head1 = torchvision.datasets.ImageFolder(root=f'.data/niid_{dataset}_S100_C{num_clients}/val/client_{currentClient}', transform=transform_val)
                            trainset_head2 = torchvision.datasets.ImageFolder(root=f'.data/niid_{dataset}_S100_C{num_clients}/train/client_{currentClient}', transform=transform_train)
                            valset_head2 = torchvision.datasets.ImageFolder(root=f'.data/niid_{dataset}_S100_C{num_clients}/val/client_{currentClient}', transform=transform_val)

                            # Define batch size and workers
                            batch_size = 128
                            worker = 6

                            # Create data loaders
                            trainloader_head1 = DataLoader(trainset_head1, batch_size=batch_size, shuffle=True, num_workers=worker)
                            valloader_head1 = DataLoader(valset_head1, batch_size=batch_size, shuffle=False, num_workers=worker)
                            trainloader_head2 = DataLoader(trainset_head2, batch_size=batch_size, shuffle=True, num_workers=worker)
                            valloader_head2 = DataLoader(valset_head2, batch_size=batch_size, shuffle=False, num_workers=worker)

                            # Define student model

                            if currentClient == 0 or currentClient == 4 or currentClient == 7:
                                base_model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
                                student_model = CustomResNet(base_model)
                                student_model.to(device)
                            if currentClient == 1 or currentClient == 5 or currentClient == 6:
                                base_model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
                                student_model = CustomResNet(base_model)
                                student_model.to(device)
                            if currentClient == 2 or currentClient == 3:
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
                            if not os.path.exists(f'./{dataset}/niid_S100_C{num_clients}/first_head_client{currentClient}.pth'):
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
                                        torch.save({'model_state_dict': student_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'./{dataset}/niid_S100_C{num_clients}/first_head_client{currentClient}.pth')
                                        print('Saving the best model for Head 1')
                                    else:
                                        no_improvement_count += 1
                                        if no_improvement_count >= patience:
                                            print(f"Validation loss has not improved for {patience} epochs. Stopping early.")
                                            break
                            
                            else:

                                
                                # Function to initialize fc2 weights from fc1
                                def initialize_fc2_from_fc1(model):
                                    with torch.no_grad():
                                        fc1_weights = model.fc1.weight.data
                                        sorted_weights, _ = torch.sort(fc1_weights, dim=1)
                                        model.fc2.weight.data.copy_(sorted_weights)
                                        model.fc2.bias.data.zero_()  # Optionally zero out the bias

                                def load_second_head_checkpoint(student_model):
                                    # Check if a checkpoint for the second head exists
                                    second_head_checkpoint_path = f'./{dataset}/niid_S100_C{destinationClient}/clients_{distillation_case}_{topology}/second_head_client{currentClient}_alpha{alpha}_temp{temperature}_{mode}_iter{iteration-1}.pth'
                                    
                                    if os.path.exists(second_head_checkpoint_path) and iteration >  1:
                                        print("Loading checkpoint for the second head...")
                                        checkpoint = torch.load(second_head_checkpoint_path)
                                        student_model.load_state_dict(checkpoint['model_state_dict'])
                                    else:
                                        print("No existing second head checkpoint found, initializing second head.")
                                        checkpoint = torch.load(f'./{dataset}/niid_S100_C{num_clients}/first_head_client{currentClient}.pth')
                                        student_model.load_state_dict(checkpoint['model_state_dict'])
                                        initialize_fc2_from_fc1(student_model)
                                    return student_model
                                
                                    

                                    
                                    
                                        
                                def load_student_model_clone(client_number, dataset, num_clients, device):
                                    if client_number in [0, 4, 7]:
                                        base_model_clone = models.resnet18(weights=None)
                                    elif client_number in [1, 5, 6]:
                                        base_model_clone = models.resnet18(weights=None)
                                    elif client_number in [2, 3]:
                                        base_model_clone = models.resnet18(weights=None)
                                    else:
                                        raise ValueError("Invalid client number for model loading")

                                    model_clone = CustomResNet(base_model_clone)
                                    model_clone.to(device)
                                    if iteration == 1:
                                        checkpoint = torch.load(f'./{dataset}/niid_S100_C{num_clients}/first_head_client{client_number}.pth')
                                        model_clone.load_state_dict(checkpoint['model_state_dict'])
                                    else:
                                        checkpoint = torch.load(f'./{dataset}/niid_S100_C{destinationClient}/clients_{distillation_case}_{topology}/second_head_client{client_number}_alpha{alpha}_temp{temperature}_{mode}_iter{iteration-1}.pth')
                                        model_clone.load_state_dict(checkpoint['model_state_dict'])
                                    return model_clone

                                # Load the model for training the second head
                                checkpoint = torch.load(f'./{dataset}/niid_S100_C{num_clients}/first_head_client{currentClient}.pth')
                                student_model.load_state_dict(checkpoint['model_state_dict'])

                                # Initialize second head
                                # Load the model for training the second head
                                

                                # Initialize or load the second head
                                student_model = load_second_head_checkpoint(student_model)
                                

                                

                                # load remote clients
                                
                                remote_clients = [load_student_model_clone(client_number, dataset, num_clients, device) for client_number in remoteClients]
                                
                                
                                # Define optimizer for second head training (fc2)
                                optimizer = optim.SGD(student_model.fc2.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
                                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

                                # Train the second head
                                start_time = time.time()  # Start timing the training
                                for epoch in range(num_epochs):
                                    
                                    student_model.train()
                                    running_train_loss = 0.0
                                    correct_train = 0
                                    total_train = 0
                                    with tqdm(trainloader_head2, unit="batch") as tepoch:
                                        for inputs, labels in tepoch:
                                            inputs, labels = inputs.to(device), labels.to(device)
                                            optimizer.zero_grad()

                                            # Compute forward pass for remote clients
                                            remote_outputs = []
                                            with torch.no_grad():
                                                for remote_client in remote_clients:
                                                    if iteration == 1:
                                                        remote_output, _ = remote_client(inputs)
                                                        remote_outputs.append(remote_output.to(device))
                                                    else:
                                                        #print("Iteration 2 remote forward pass")
                                                        _, remote_output = remote_client(inputs)
                                                        remote_outputs.append(remote_output.to(device))

                                            # Aggregate remote outputs based on 'dosum' flag
                                            if dosum == False:
                                                remote_output = torch.stack(remote_outputs).mean(dim=0)
                                            

                                            _, outputs2 = student_model(inputs)
                                            loss2 = criterion(outputs2, labels)

                                            # Compute distillation loss (same as original)
                                            if mode == 'CE':
                                                if dosum:
                                                    distillation_loss = sum(
                                                        nn.CrossEntropyLoss(reduction='sum')(outputs2 / temperature, nn.functional.softmax(ro / temperature, dim=1))
                                                        for ro in remote_outputs
                                                    )
                                                else:
                                                    distillation_loss = nn.CrossEntropyLoss(reduction='sum')(outputs2 / temperature, nn.functional.softmax(remote_output / temperature, dim=1))
                                            elif mode == 'log_ce':
                                                if dosum:
                                                    distillation_loss = sum(
                                                        nn.CrossEntropyLoss(reduction='sum')(nn.functional.log_softmax(outputs2 / temperature, dim=1), nn.functional.softmax(ro / temperature, dim=1))
                                                        for ro in remote_outputs
                                                    )
                                                else:
                                                    distillation_loss = nn.CrossEntropyLoss(reduction='sum')(nn.functional.log_softmax(outputs2 / temperature, dim=1), nn.functional.softmax(remote_output / temperature, dim=1))
                                            elif mode == 'KL'  :
                                                if dosum:
                                                    distillation_loss = sum(
                                                        compute_kl_torch(ro / temperature, outputs2 / temperature, reduction='sum')
                                                        for ro in remote_outputs
                                                    )
                                                else:
                                                    distillation_loss = compute_kl_torch(remote_output / temperature, outputs2 / temperature, reduction='sum')
                                            elif mode == 'JS' :
                                                if dosum:
                                                    distillation_loss = sum(
                                                        jensen_shannon_divergence(nn.functional.softmax(outputs2 / temperature, dim=1), nn.functional.softmax(ro / temperature, dim=1))
                                                        for ro in remote_outputs
                                                    )
                                                else:
                                                    distillation_loss = jensen_shannon_divergence(nn.functional.softmax(outputs2 / temperature, dim=1), nn.functional.softmax(remote_output / temperature, dim=1))
                                            elif mode == 'TD' :
                                                if dosum:
                                                    
                                                    distillation_loss = sum(
                                                        triangular_distance(nn.functional.softmax(outputs2 / temperature, dim=1), nn.functional.softmax(ro / temperature, dim=1))
                                                        for ro in remote_outputs
                                                    )
                                                else:
                                                    distillation_loss = triangular_distance(nn.functional.softmax(outputs2 / temperature, dim=1), nn.functional.softmax(remote_output / temperature, dim=1))
                                            elif mode == 'SED_exp' :
                                                if dosum:
                                                    distillation_loss = sum(
                                                        sed(nn.functional.softmax(outputs2 / temperature, dim=1), nn.functional.softmax(ro / temperature, dim=1))
                                                        for ro in remote_outputs
                                                    )
                                                else:
                                                    distillation_loss = sed(nn.functional.softmax(outputs2 / temperature, dim=1), nn.functional.softmax(remote_output / temperature, dim=1))
                                            elif mode == 'SED'  :
                                                if dosum:
                                                    distillation_loss = sum(
                                                        compute_sed(nn.functional.softmax(outputs2 / temperature, dim=1), nn.functional.softmax(ro / temperature, dim=1))
                                                        for ro in remote_outputs
                                                    )
                                                else:
                                                    distillation_loss = compute_sed(nn.functional.softmax(outputs2 / temperature, dim=1), nn.functional.softmax(remote_output / temperature, dim=1))
                                            elif mode == 'MSED_exp' :
                                                if dosum:
                                                    msed_tensors = [nn.functional.softmax(outputs2 / temperature, dim=1)] + [nn.functional.softmax(ro / temperature, dim=1) for ro in remote_outputs]
                                                    distillation_loss = msed(msed_tensors)
                                                else:
                                                    msed_tensors = [nn.functional.softmax(outputs2 / temperature, dim=1)] + [nn.functional.softmax(ro / temperature, dim=1) for ro in remote_outputs]
                                                    distillation_loss = msed(msed_tensors)
                                            elif mode == 'MSED' :
                                                if dosum:
                                                    msed_tensors = [nn.functional.softmax(outputs2 / temperature, dim=1)] + [nn.functional.softmax(ro / temperature, dim=1) for ro in remote_outputs]
                                                    distillation_loss = compute_msed(msed_tensors)
                                                else:
                                                    msed_tensors = [nn.functional.softmax(outputs2 / temperature, dim=1)] + [nn.functional.softmax(ro / temperature, dim=1) for ro in remote_outputs]
                                                    distillation_loss = compute_msed(msed_tensors)
                                            else:
                                                raise ValueError(f"Unknown mode: {mode}")
                                            # Total loss
                                            
                                            loss2 = alpha * loss2 + (1 - alpha) * distillation_loss
                                            loss2.backward()
                                            optimizer.step()
                                            running_train_loss += loss2.item()

                                            # Accuracy computation
                                            _, predicted2 = torch.max(outputs2, 1)
                                            total_train += labels.size(0)
                                            correct_train += (predicted2 == labels).sum().item()

                                            # Update progress bar
                                            tepoch.set_postfix(loss=running_train_loss / total_train, accuracy=100. * correct_train / total_train)
                                    
                                    # Validation phase
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
                                        
                                        torch.save({'model_state_dict': student_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'./{dataset}/niid_S100_C{destinationClient}/clients_{distillation_case}_{topology}/second_head_client{currentClient}_alpha{alpha}_temp{temperature}_{mode}_iter{iteration}.pth')
                                        
                                        
                                        print('Saving the best model for Head 2')
                                    else:
                                        no_improvement_count += 1
                                        if no_improvement_count >= patience:
                                            print(f"Validation loss has not improved for {patience} epochs. Stopping early.")
                                            break
                                    # End timing and log the training details
                                end_time = time.time()
                                training_time = end_time - start_time  
                                #log_training_details(mode,f"Training_Client_{currentClient}", training_time)
            
                                print('Training complete for the second head!')

                                #print(f"Time taken for training second head: {elapsed_time // 60:.0f} minutes and {elapsed_time % 60:.0f} seconds")


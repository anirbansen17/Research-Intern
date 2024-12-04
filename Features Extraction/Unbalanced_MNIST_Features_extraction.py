import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np
from tqdm import tqdm

# Define the function to create an imbalanced dataset
def get_imbalanced_data(dataset, num_sample_per_class, shuffle=False, random_seed=0):
    """
    Return a list of imbalanced indices from a dataset.
    Input: A dataset (e.g., MNIST), num_sample_per_class: list of integers
    Output: imbalanced_list
    """
    np.random.seed(random_seed)
    length = len(dataset)
    num_sample_per_class = list(num_sample_per_class)
    selected_list = []
    indices = list(range(length))

    if shuffle:
        np.random.shuffle(indices)

    for index in indices:
        _, label = dataset[index]
        if num_sample_per_class[label] > 0:
            selected_list.append(index)
            num_sample_per_class[label] -= 1

    return selected_list


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), 
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)


num_sample_per_class = [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]


imbalanced_indices = get_imbalanced_data(train_dataset, num_sample_per_class, shuffle=True, random_seed=0)

imbalanced_train_dataset = torch.utils.data.Subset(train_dataset, imbalanced_indices)
imbalanced_train_loader = torch.utils.data.DataLoader(imbalanced_train_dataset, batch_size=64, shuffle=True)


model = resnet50(pretrained=True)


model = torch.nn.Sequential(*list(model.children())[:-1])
model = model.to(device)
model.eval()


def extract_features(loader):
    features = []
    labels = []
    with torch.no_grad():
        for images, lbls in tqdm(loader, desc="Extracting features"):
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.view(outputs.size(0), -1)  
            features.append(outputs.cpu())
            labels.append(lbls.cpu())
    return torch.cat(features), torch.cat(labels)


print("Extracting features from the imbalanced training set...")
imbalanced_train_features, imbalanced_train_labels = extract_features(imbalanced_train_loader)
print("Extracting features from the test set...")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
test_features, test_labels = extract_features(test_loader)


torch.save(imbalanced_train_features, 'imbalanced_train_features.pt')
torch.save(imbalanced_train_labels, 'imbalanced_train_labels.pt')
torch.save(test_features, 'test_features.pt')
torch.save(test_labels, 'test_labels.pt')

print("Feature extraction complete and saved to files.")

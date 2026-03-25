import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
import mlflow
import mlflow.pytorch
import time
import yaml
from thop import profile
from models.resnet50 import ResNet50FineTuned
from optimizers.adamw import adamw
from schedulers.onecyclelr import onecyclelr

mlflow.set_tracking_uri("sqlite:////ceph/home/student.aau.dk/rk33gs/MLOps/mlflow.db")

mlflow.set_experiment("resnet50-emotion-classifier")

# Enable system metrics monitoring
#mlflow.config.enable_system_metrics_logging()
#mlflow.config.set_system_metrics_sampling_interval()


with open("config/train_config.yaml") as f:
    config = yaml.safe_load(f)

dataset_config = config['dataset']

model_config = config['model']

train_config = config['train']

optimizer_config = config['optimizer']

scheduler_config = config['scheduler']

evaluate_config = config['evaluate']

dataset_path = dataset_config['dataset_path']

classes_to_idx = config['classes']


image_path_list = []
image_label_list = []


for class_name, class_idx in classes_to_idx.items():
    folder_path = os.path.join(dataset_path, class_name)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        image_path_list.append(file_path)
        image_label_list.append(class_idx)



train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(image_path_list, image_label_list, test_size=dataset_config['test_size'], random_state=dataset_config['random_state'])
train_paths, val_paths, train_labels, val_labels = train_test_split(train_val_paths, train_val_labels, test_size=dataset_config['test_size'], random_state=dataset_config['random_state'])


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=train_config['RandomResizedCrop']['size'], 
                                 scale=tuple(train_config['RandomResizedCrop']['scale'])),
    transforms.RandomHorizontalFlip(p=train_config['RandomHorizontalFlip']['p']),
    transforms.RandomRotation(degrees=train_config['RandomRotation']['degrees']),
    transforms.ColorJitter(brightness=train_config['ColorJitter']['brightness'], 
                           contrast=train_config['ColorJitter']['contrast']),
    transforms.ToTensor(),
    transforms.Normalize(mean=train_config['Normalize']['mean'], 
                         std=train_config['Normalize']['std'])
    ])

val_test_transform = transforms.Compose([
    transforms.Resize(size=evaluate_config['Resize']['size']),
    transforms.CenterCrop(size=evaluate_config['CenterCrop']['size']),
    transforms.ToTensor(),
    transforms.Normalize(mean=evaluate_config['Normalize']['mean'], 
                         std=evaluate_config['Normalize']['std'])
    ])


class CustomDataset(Dataset):
    def __init__(self, img_paths, img_labels, transform=None):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        label = torch.tensor(self.img_labels[idx], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label



train_set = CustomDataset(train_paths,
                          train_labels,
                          transform=train_transform)

val_set = CustomDataset(val_paths,
                        val_labels,
                        transform=val_test_transform)


class_counts = Counter(train_labels)
num_classes = len(classes_to_idx)
total_samples = len(train_labels)
weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]



train_dataloader = DataLoader(train_set, batch_size=train_config['batch_size'], 
                              shuffle=train_config['shuffle'], 
                              num_workers=train_config['num_workers'], 
                              pin_memory=train_config['pin_memory'])

val_dataloader = DataLoader(val_set, batch_size=train_config['batch_size'], 
                            shuffle=evaluate_config['shuffle'], 
                            num_workers=train_config['num_workers'], 
                            pin_memory=evaluate_config['pin_memory'])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ResNet50FineTuned(model_config)

model = model.to(device)

class_weights = torch.tensor(weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = adamw(model, model_config, optimizer_config)

scheduler = onecyclelr(optimizer, scheduler_config, train_dataloader)


epochs = train_config['epochs']

train_loss_list = []
val_loss_list = []

with mlflow.start_run(run_name="training"):

    mlflow.log_param("test_param", 123)
    mlflow.log_metric("test_metric", 0.5)

    # Logging model parameters and flops by running a single tensor, consistent with the shape of the images,
    # through the model.
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(model, inputs=(dummy_input,))
    mlflow.log_metric("model_params", params)
    mlflow.log_metric("model_flops", flops)

    start_time = time.time()

    mlflow.log_param("model", model_config['name'])
    mlflow.log_param("batch_size", train_config['batch_size'])
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("optimizer", optimizer_config['name'])
    mlflow.log_param("learning_rate", optimizer_config['lr'])
    
    best_acc = 0.0
    for epoch in range(epochs):
        print(f"epoch {epoch+1}/{epochs}")
        running_train_loss = 0.0
        running_train_corrects = 0.0

        model.train()
        for X_train, y_train in train_dataloader:
            X_train = X_train.to(device)
            y_train = y_train.to(device)

            optimizer.zero_grad()

            output_train = model(X_train)
            train_loss = criterion(output_train, y_train)
            train_loss.backward()
            optimizer.step()
            scheduler.step()

            _, preds = torch.max(output_train, 1)
            running_train_corrects += torch.sum(preds == y_train.data).item()
            running_train_loss += train_loss.item()

        train_epoch_acc = running_train_corrects / len(train_set)
        train_epoch_loss = running_train_loss / len(train_dataloader)
        train_loss_list.append(train_epoch_loss)
        print(f"Training loss: {train_epoch_loss:.4f} Training accuracy: {train_epoch_acc:.4f}")

        mlflow.log_metric("train_epoch_loss", train_epoch_loss, step=epoch)
        mlflow.log_metric("train_epoch_accuracy", train_epoch_acc, step=epoch)


        model.eval()
        running_val_loss = 0.0
        running_val_corrects = 0.0
        with torch.no_grad():
            for X_val, y_val in val_dataloader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                output_val = model(X_val)
                val_loss = criterion(output_val, y_val)
                running_val_loss += val_loss.item()

                _, preds = torch.max(output_val, 1)
                running_val_corrects += torch.sum(preds == y_val.data).item()

        val_epoch_acc = running_val_corrects / len(val_set)
        val_epoch_loss = running_val_loss / len(val_dataloader)
        val_loss_list.append(val_epoch_loss)

        mlflow.log_metric("val_epoch_loss", val_epoch_loss, step=epoch)
        mlflow.log_metric("val_epoch_accuracy", val_epoch_acc, step=epoch)

        print(f"Validation loss: {val_epoch_loss:.4f} Validation accuracy: {val_epoch_acc:.4f}")

        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_state = model.state_dict()

    print(f"Best evaluation accuracy", best_acc)

    print(train_loss_list, val_loss_list)

    mlflow.log_metric("best_val_accuracy", best_acc)

    model.load_state_dict(best_model_state)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_val, y_val in val_dataloader:
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            output = model(X_val)
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_val.cpu().numpy())

    report = classification_report(
        all_labels,
        all_preds,
        target_names=classes_to_idx.keys(),
        output_dict=True)

    accuracy = report["accuracy"]

    mlflow.log_metric("final_val_accuracy", accuracy)

    print(classification_report(
        all_labels,
        all_preds,
        target_names=classes_to_idx.keys()))

    plt_epochs = range(1, epochs + 1)
    plt.plot(plt_epochs, train_loss_list, label='Train Loss')
    plt.plot(plt_epochs, val_loss_list, label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.savefig('loss_plot.png', dpi=300, bbox_inches='tight')

    mlflow.log_figure(plt.gcf(), "loss_plot.png")

    plt.close()

    end_time = time.time()
    training_duration = (end_time - start_time) / 60
    mlflow.log_metric("training_duration_minutes", training_duration)
    print(f"Training completed in {training_duration:.2f} minutes")

    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model") # Path name to the folder where the model artifact will be stored.
        #registered_model_name="resnet50-emotion-classifier")

    # Transition the newly registered model to Staging automatically
    # from mlflow.tracking import MlflowClient

    # client = MlflowClient()
    # latest_version_info = client.get_latest_versions("resnet50-emotion-classifier", stages=["None"])[0]
    # client.transition_model_version_stage(
    #     name="resnet50-emotion-classifier",
    #     version=latest_version_info.version,
    #     stage="Staging",
    #     archive_existing_versions=False  # optional
    # )
    # print(f"Model version {latest_version_info.version} moved to Staging.")
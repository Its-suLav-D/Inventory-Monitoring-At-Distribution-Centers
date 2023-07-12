# Automated Inventory Monitoring System for Distribution Centers

This project focuses on developing a machine learning model that accurately counts the number of objects within each bin at a distribution center. Leveraging AWS SageMaker and robust machine learning engineering practices, the solution aims to improve inventory management and ensure the correct number of items in each delivery consignment.

## Project Set Up and Installation

The project was primarily developed using AWS SageMaker. The following steps are needed to replicate the project:

1. Set up an AWS account and configure AWS SageMaker.
2. Create a new notebook instance in SageMaker and upload the project notebook.
3. To store and access the dataset, create an S3 bucket and upload the Amazon Bin Image Dataset to the bucket.

## Dataset

### Overview

The dataset used in this project is the Amazon Bin Image Dataset, which contains 500,000 images of bins with varying numbers of objects. The metadata files associated with each image provide details such as object count, object type, and image dimensions.

### Access

The dataset was accessed through an S3 bucket in AWS. This was achieved by creating a bucket, uploading the dataset to the bucket, and accessing the data directly from the SageMaker notebook using the Boto3 AWS SDK.

## Model Training

The model of choice for this task is the ResNet50 Convolutional Neural Network (CNN). This model is known for its ability to effectively train deep networks through its unique "skip connections", making it suitable for this multiclass classification task. The model was trained using the AWS SageMaker platform.

The chosen loss function for this task was Sparse Categorical Crossentropy. This loss function is computationally efficient and suitable for tasks involving mutually exclusive classes represented as integers.

## Machine Learning Pipeline

The pipeline for this project included the following steps:

1. **Data Preprocessing**: The images were resized to 224x224 pixels, converted into tensors, and normalized.
2. **Model Definition**: A custom ResNet50 model was defined, where the last fully connected layer was modified to match the number of output classes.
3. **Model Training**: The model was trained on the preprocessed dataset using AWS SageMaker.
4. **Model Evaluation**: The trained model was evaluated on a validation set using the metrics of accuracy and sparse categorical accuracy.
5. **Model Deployment**: The model was deployed using Amazon SageMaker.

## Results and Next Steps

The deployed model is successfully making predictions. The next steps include conducting a more comprehensive testing of the model's prediction capabilities and setting up auto-scaling rules to handle varied traffic to the endpoint. Additionally, monitoring the endpoint's performance to ensure continued performance and functionality is recommended.

## Project Links

[Github](https://github.com/Its-suLav-D/Inventory-Monitoring-At-Distribution-Centers/tree/master/starter)

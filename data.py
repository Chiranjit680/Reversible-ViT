import torch
import torchvision
import torchvision.transforms as transforms
import os
import shutil
import timm

from torch.utils.data.distributed import DistributedSampler

def get_data_loader(args):
    
    trainset, testset = get_dataset(args)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=2)
    
    num_classes = len(trainset.classes)  
    images, labels = next(iter(trainloader))
    print(f"Train samples: {len(trainset)}, Test samples: {len(testset)}, Num Classes : {num_classes}, Batch shape: {images.shape}")

    return trainloader, testloader

def get_data_loader_ddp(args, rank, world_size, pin_memory, num_workers):

    trainset, testset = get_dataset(args)

    train_sampler = DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=train_sampler)

    test_sampler = DistributedSampler(testset)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=test_sampler)
    
    num_classes = len(trainset.classes)  
    images, labels = next(iter(trainloader))
    print(f"Train samples: {len(trainset)}, Test samples: {len(testset)}, Num Classes : {num_classes}, Batch shape: {images.shape}")

    return trainloader, testloader

def get_dataset(args):

    print("==> Preparing data..")

    if args.dataset == "CIFAR10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Will downloaded and save the dataset if needed
        trainset = torchvision.datasets.CIFAR10(
            root="../data/cifar-10", train=True, download=True, transform=transform_train
        )

        testset = torchvision.datasets.CIFAR10(
            root="../data/cifar-10", train=False, download=True, transform=transform_test
        )

        return trainset, testset    
    
    elif args.dataset == "CIFAR100":

        if args.deit_scheme:
            
            transform_train = timm.data.create_transform(
                input_size=32,
                is_training=True,
                color_jitter=0.4,
                auto_augment="rand-m7-mstd0.5-inc1",
                interpolation="bicubic",
                re_prob=0.25,
                re_mode="pixel",
                re_count=1,
                mean=[0.5071, 0.4867, 0.4408], 
                std=[0.2675, 0.2565, 0.2761]
            )
            
            transform_train.transforms[0] = transforms.RandomCrop(32, padding=4)

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            ])

        else: 

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            ])

        # Will downloaded and save the dataset if needed
        trainset = torchvision.datasets.CIFAR100(
            root="../data/cifar-100", train=True, download=True, transform=transform_train
        )

        testset = torchvision.datasets.CIFAR100(
            root="../data/cifar-100", train=False, download=True, transform=transform_test
        )

        return trainset, testset
    
    elif args.dataset == "TinyImageNet":

        if os.path.isdir("../data/tiny-imagenet-200"):
            print("data/tiny-imagenet-200 directory already exists!")
            print("Using Existing Image Folder Directory")
        else:
            print("Downloading Tiny ImageNet")
            
            if not os.path.isdir("../data"):
                os.makedirs("../data")
            os.system("wget http://cs231n.stanford.edu/tiny-imagenet-200.zip")
            os.system("unzip -q tiny-imagenet-200.zip -d ../data/")
            os.system(f"rm -rf tiny-imagenet-200.zip")

            with open("../data/tiny-imagenet-200/val/val_annotations.txt", "r") as f:
                data = f.readlines()

            for line in data:
                image_name, class_id, *_ = line.split("\t")

                if not os.path.exists(f"../data/tiny-imagenet-200/val/{class_id}"):
                    os.makedirs(f"../data/tiny-imagenet-200/val/{class_id}")

                shutil.move(f"../data/tiny-imagenet-200/val/images/{image_name}", f"../data/tiny-imagenet-200/val/{class_id}/{image_name}")

            shutil.rmtree("../data/tiny-imagenet-200/val/images/")
            os.system("rm -rf ../data/tiny-imagenet-200/val/val_annotations.txt")

            for dir in os.listdir("../data/tiny-imagenet-200/train/"):
                for img in os.listdir(f"../data/tiny-imagenet-200/train/{dir}/images/"):
                    shutil.move(f"../data/tiny-imagenet-200/train/{dir}/images/{img}", f"../data/tiny-imagenet-200/train/{dir}/{img}")

                shutil.rmtree(f"../data/tiny-imagenet-200/train/{dir}/images/")
                os.system(f"rm -rf ../data/tiny-imagenet-200/train/{dir}/*.txt")

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Define dataset paths
        train_dir = "../data/tiny-imagenet-200/train"
        val_dir = "../data/tiny-imagenet-200/val"

        # Load dataset using ImageFolder
        trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)

        testset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform_test)

        return trainset, testset
        
    
    elif args.dataset == "imagenet-100":

        if os.path.isdir("../data/imagenet-100"):
            print("Loading data from ../data/imagenet-100")
        else:
            print("../data/imagenet-100 not found")
            exit()

        # transform_train = transforms.Compose([
        #     transforms.RandomResizedCrop(256),
        #     transforms.CenterCrop(224),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])

        # transform_test = transforms.Compose([
        #     transforms.RandomResizedCrop(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])

        transform_train = timm.data.create_transform(
            input_size=224,
            is_training=True,
            color_jitter=0.4,
            auto_augment="rand-m7-mstd0.5-inc1",
            interpolation="bicubic",
            re_prob=0.25,
            re_mode="pixel",
            re_count=1
        )

        transform_test = transforms.Compose([
            transforms.Resize(int(224 / 0.875), interpolation=3), # eval crop ratio
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Define dataset paths
        train_dir = "../data/imagenet-100/train"
        val_dir = "../data/imagenet-100/val"

        # Load dataset using ImageFolder
        trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)

        testset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform_test)

        return trainset, testset
    
    elif args.dataset == "imagenet":

        if os.path.isdir("../data/imagenet"):
            print("Loading data from ../data/imagenet")
        else:
            print("../data/imagenet not found")
            exit()

        transform_train = timm.data.create_transform(
            input_size=224,
            is_training=True,
            color_jitter=0.4,
            auto_augment="rand-m7-mstd0.5-inc1",
            interpolation="bicubic",
            re_prob=0.25,
            re_mode="pixel",
            re_count=1
        )

        transform_test = transforms.Compose([
            transforms.Resize(int(224 / 0.875), interpolation=3), # eval crop ratio
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Define dataset paths
        train_dir = "../data/imagenet/train"
        val_dir = "../data/imagenet/val"

        # Load dataset using ImageFolder
        trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)

        testset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform_test)

        return trainset, testset


    else:
        print(f"{args.dataset} not known")
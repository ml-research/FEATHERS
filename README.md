# HANF (Hyperparameter and Neural Architecture Search in Federated Learning)

## Experiments for Presentation
| Dataset       | Number of Clients | Reult   |
|---------------|-------------------|---------|
| FashionMNIST  | 2                 | 5/5     |
| FashionMNIST  | 5                 | 4/5     |
| FashionMNIST  | 10                | -       |
| CIFAR10       | 2                 | 5/5     |
| CIFAR10       | 5                 | 3/5     |
| CIFAR10       | 10                | -       |

## Experiments for Paper
### iid
| Dataset       | Number of Clients | Reult   |
|---------------|-------------------|---------|
| FashionMNIST  | 2                 | 5/5     |
| FashionMNIST  | 5                 | 5/5     |
| FashionMNIST  | 10                | 0/5     |
| CIFAR10       | 2                 | 0/5     |
| CIFAR10       | 5                 | 3/5     |
| CIFAR10       | 10                | 0/5     |

### non-iid (label skew)
| Dataset       | Number of Clients | Reult   |
|---------------|-------------------|---------|
| FashionMNIST  | 2                 | 0/5     |
| FashionMNIST  | 5                 | 0/5     |
| FashionMNIST  | 10                | 0/5     |
| CIFAR10       | 2                 | 0/5     |
| CIFAR10       | 5                 | 0/5     |
| CIFAR10       | 10                | 0/5     |
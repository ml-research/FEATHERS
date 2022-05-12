# HANF (Hyperparameter and Neural Architecture Search in Federated Learning)

## Experiments for Paper (Search)
### iid
| Dataset       | Number of Clients | Reult   |
|---------------|-------------------|---------|
| FashionMNIST  | 2                 | 5/5     |
| FashionMNIST  | 5                 | 5/5     |
| FashionMNIST  | 10                | 0/5     |
| CIFAR10       | 2                 | 5/5     |
| CIFAR10       | 5                 | 5/5     |
| CIFAR10       | 10                | 0/5     |

### non-iid (label skew)
| Dataset       | Number of Clients | Reult   |
|---------------|-------------------|---------|
| FashionMNIST  | 2                 | 2/5     |
| FashionMNIST  | 5                 | 2/5     |
| FashionMNIST  | 10                | 0/5     |
| CIFAR10       | 2                 | 0/5     |
| CIFAR10       | 5                 | 1/5     |
| CIFAR10       | 10                | 0/5     |

## Experiments for Paper (Validation)
### iid
| Dataset       | Number of Clients | Reult   |
|---------------|-------------------|---------|
| FashionMNIST  | 2                 | 0/5     |
| FashionMNIST  | 5                 | 0/5     |
| FashionMNIST  | 10                | 0/5     |
| CIFAR10       | 2                 | 1/5     |
| CIFAR10       | 5                 | 2/5     |
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
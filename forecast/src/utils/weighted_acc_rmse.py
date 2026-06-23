import numpy as np
import torch

# torch version for rmse comp
def lat_np(j: np.ndarray, num_lat: int) -> np.ndarray:
    return 90 - j * 180/float(num_lat-1)

def latitude_weighting_factor(j: np.ndarray, num_lat: int, s: np.ndarray) -> np.ndarray:
    return num_lat * np.cos(3.1416/180. * lat_np(j, num_lat))/s

def weighted_acc_channels(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc
    num_lat = pred.shape[2]
    lat_t = np.arange(start=0, stop=num_lat)
    s = np.sum(np.cos(3.1416/180. * lat_np(lat_t, num_lat)))
    weight = np.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    result = np.sum(weight * pred * target, axis=(-1,-2), keepdims=True) / np.sqrt(np.sum(weight * pred * pred, axis=(-1,-2), keepdims=True) * np.sum(weight * target *
    target, axis=(-1,-2), keepdims=True))
    return result

def weighted_acc(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    result = weighted_acc_channels(pred, target)
    return np.mean(result, axis=0, keepdims=True)

def weighted_rmse_channels(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    lat_t = np.arange(start=0, stop=num_lat)

    s = np.sum(np.cos(3.1416/180. * lat_np(lat_t, num_lat)))
    weight = np.reshape(latitude_weighting_factor(lat_t, num_lat, s), (1, 1, -1, 1))
    # weight = 1
    result = np.sqrt(np.mean(weight * (pred - target)**2., axis=(-1,-2), keepdims=True))
    return result

def weighted_rmse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    result = weighted_rmse_channels(pred, target)
    return np.mean(result, axis=0)

def weighted_mae_channels(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    lat_t = np.arange(start=0, stop=num_lat)

    s = np.sum(np.cos(3.1416/180. * lat_np(lat_t, num_lat)))
    weight = np.reshape(latitude_weighting_factor(lat_t, num_lat, s), (1, 1, -1, 1))
    # weight = 1
    result = np.mean(weight * np.abs(pred - target), axis=(-1,-2), keepdims=True)
    return result

def weighted_mae(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    result = weighted_mae_channels(pred, target)
    return np.mean(result, axis=0)

# torch version for rmse comp
def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    return 90 - j * 180/float(num_lat-1)

def latitude_weighting_factor_torch(j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return num_lat * torch.cos(3.1416/180. * lat(j, num_lat))/s

def weighted_acc_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sum(weight * pred * target, dim=(-1,-2)) / torch.sqrt(torch.sum(weight * pred * pred, dim=(-1,-2)) * torch.sum(weight * target *
    target, dim=(-1,-2)))
    return result

def weighted_acc_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = weighted_acc_torch_channels(pred, target)
    return torch.mean(result, dim=0)

def weighted_rmse_torch_channels(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sum(weight * mask * (pred - target)**2., dim=(-1,-2))
    result /= torch.sum(mask, dim=(-1,-2))
    result = torch.sqrt(result)
    return result

def weighted_rmse_torch(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    result = weighted_rmse_torch_channels(pred, target, mask)
    return torch.mean(result, dim=0)

def weighted_bias_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.mean(weight * (pred - target), dim=(-1,-2))
    return result

def weighted_bias_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = weighted_bias_torch_channels(pred, target)
    return torch.mean(result, dim=0)

def weighted_mae_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.mean(weight * torch.abs(pred - target), dim=(-1,-2))
    return result

def weighted_mae_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = weighted_mae_torch_channels(pred, target)
    return torch.mean(result, dim=0)

def weighted_latitude_weighting_factor_torch(j: torch.Tensor, real_num_lat:int, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return real_num_lat * torch.cos(3.1416/180. * lat(j, num_lat)) / s

def type_weighted_activity_torch_channels(pred: torch.Tensor, metric_type="all") -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    northern_index = int(110. / 180. * num_lat + 0.5)
    souther_index = int(70. / 180. * num_lat + 0.5)

    if metric_type == "all":
        s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
        weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t, num_lat, num_lat, s), (1, 1, -1, 1))
        result = torch.sqrt(torch.mean(weight * (pred - torch.mean(weight * pred, dim=(-1, -2), keepdim=True)) ** 2, dim=(-1, -2)))
        return result

    elif metric_type == "northern":
        northern_s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat))[northern_index:])
        northern_weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t[northern_index:], souther_index, num_lat, northern_s), (1, 1, -1, 1))
        northern_result = torch.sqrt(torch.mean(northern_weight * (pred[:, :, northern_index:] - torch.mean(northern_weight * pred[:, :, northern_index:], dim=(-1, -2), keepdim=True)) ** 2, dim=(-1, -2)))
        return northern_result
    
    elif metric_type == "southern":
        southern_s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat))[:souther_index])
        southern_weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t[:souther_index], souther_index, num_lat, southern_s), (1, 1, -1, 1))
        southern_result = torch.sqrt(torch.mean(southern_weight * (pred[:, :, :souther_index] - torch.mean(southern_weight * pred[:, :, :souther_index], dim=(-1, -2), keepdim=True)) ** 2, dim=(-1, -2)))
        return southern_result
    
def activity_torch(pred, clim_time_mean_daily, data_std=None):
    """
    Activity metric.
    Parameters
    ----------
    pred: tensor, required, the predicted;
    gt: tensor, required, the ground-truth
    data_std: when gt and pred is de-normlized, it keeps as None.
    Returns
    -------
    """
    if data_std is None:
        result = torch.mean(type_weighted_activity_torch_channels(pred - clim_time_mean_daily, metric_type="all"), dim=0)
    else:
        result = torch.mean(type_weighted_activity_torch_channels(pred - clim_time_mean_daily, metric_type="all") * data_std, dim=0)

    return result
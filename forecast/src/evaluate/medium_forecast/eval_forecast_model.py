import sys
sys.path.append(".")
import os


import numpy as np
import torch
from src.models.forecast.oceanforecast_module import ForecastLitModule
from src.evaluate.inference_forecast import autoregressive_inference
import argparse

import h5py

import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(name)s - %(levelname)s - %(message)s')


IN_VARIABLES = [
    "z_500", "z_850", "msl", "t2m", "u10", "v10",
    # "ssh", "ice_area_fraction", "ice_thickness", "mld",
    "thetao_0.5",    ]

OUT_VARIABLES = [
    # "ssh", "ice_area_fraction", "ice_thickness", "mld",
    "thetao_0.5",
    ]

def get_normalize(data_dir, variables):
    normalize_mean = dict(np.load(os.path.join(data_dir, "normalize_mean.npz")))
    mean = []
    for var in variables:
        mean.append(normalize_mean[var])
    normalize_mean = np.concatenate(mean).astype(np.float32)
    normalize_std = dict(np.load(os.path.join(data_dir, "normalize_std.npz")))
    normalize_std = np.concatenate([normalize_std[var] for var in variables]).astype(np.float32)

    return normalize_mean.reshape(1, normalize_mean.shape[0], 1, 1), normalize_std.reshape(1, normalize_std.shape[0], 1, 1)

def open_h5(file_list, shard_idx):
    _file = h5py.File(file_list[shard_idx], 'r')
    return _file

def forecast_model_inference(data_dir,
                            resolution,
                            years,
                            pretrain_ckpt,
                            output_dir,
                            forecast_days,
                            dt,
                            decorrelation_days,
                            model_name,
                            device):
    
    forecast_model = ForecastLitModule.load_from_checkpoint(f"{pretrain_ckpt}/{model_name}.ckpt",
                                                            mean_path=f"{data_dir}/normalize_mean.npz",
                                                            std_path=f"{data_dir}/normalize_std.npz",
                                                            ckpt_path=None,
                                                            pressure_weight=False)
    
    forecast_net = forecast_model.net.to(device).eval()

    mult = forecast_model.mult
    in_mean, in_std = get_normalize(data_dir, IN_VARIABLES)
    out_mean, out_std = get_normalize(data_dir, OUT_VARIABLES)
    gt_listers = [os.path.join(data_dir, "test", f) 
                    for f in os.listdir(os.path.join(data_dir, "test")) 
                    if os.path.isfile(os.path.join(data_dir, "test", f))]
    # 使用sorted函数和自定义的排序键对文件列表进行排序
    gt_list = [f for f in gt_listers if "times" not in f]
    h5gt = [open_h5(gt_list, idx) for idx in range(len(gt_list))]
    
    
    gt_listers1 = [os.path.join(f"/hpcfs/fhome/yangjh16/CoastNet/odata/", "test", f) 
                    for f in os.listdir(os.path.join(f"/hpcfs/fhome/yangjh16/CoastNet/odata/", "test")) 
                    if os.path.isfile(os.path.join(f"/hpcfs/fhome/yangjh16/CoastNet/odata/", "test", f))]
    # 使用sorted函数和自定义的排序键对文件列表进行排序
    gt_list1 = [f for f in gt_listers1 if "times" not in f]
    h5gt1 = [open_h5(gt_list1, idx) for idx in range(len(gt_list1))]



    clim_list = [os.path.join(data_dir, "climatology", f) 
                    for f in os.listdir(os.path.join(data_dir, "climatology")) 
                    if os.path.isfile(os.path.join(data_dir, "climatology", f))]
    h5clim = [open_h5(clim_list, idx) for idx in range(len(clim_list))]

    time_ = np.load(os.path.join(data_dir, "test", "times.npz"))
    
    time_list = []
    for i, key in enumerate(time_.keys()):
        array = time_[key]
        time_list.append(array)
    
    final_time = np.concatenate(time_list, axis=0)

    # lon = np.load(os.path.join(data_dir, "lon.npy"))
    # lat = np.load(os.path.join(data_dir, "lat.npy"))

    # 取初始场
    n_samples = final_time.shape[0] - forecast_days # eval_dataset.shape[0]
    stop = n_samples
    stop = 1
    ics = list(np.arange(0, stop, decorrelation_days))
    ics.sort()
    
    val_forecsat, val_real, val_rmse, val_acc, val_activity = [], [], [], [], []

    for i, ic in enumerate(ics):
        seq_real, seq_forecsat, seq_rmse, seq_acc, seq_activity = autoregressive_inference(ic,
                                                                                            resolution, 
                                                                                            years,
                                                                                            in_mean,
                                                                                            in_std,
                                                                                            out_mean,
                                                                                            out_std,
                                                                                            forecast_net,
                                                                                            dt,
                                                                                            forecast_days,
                                                                                            h5gt,
                                                                                            h5gt1,
                                                                                            h5clim,
                                                                                            mult,
                                                                                            IN_VARIABLES,
                                                                                            OUT_VARIABLES,
                                                                                            device)

        if i == 0:
            # val_real = seq_real
            # val_forecsat = seq_forecsat
            val_rmse = seq_rmse
            val_acc = seq_acc
            val_activity = seq_activity
        else:
            if seq_real is not None:
                # val_real = np.concatenate((val_real, seq_real), axis=0)
                # val_forecsat = np.concatenate((val_forecsat, seq_forecsat), axis=0)
                val_rmse = np.concatenate((val_rmse, seq_rmse), axis=0)
                val_acc = np.concatenate((val_acc, seq_acc), axis=0)
                val_activity = np.concatenate((val_activity, seq_activity), axis=0)
    
    for i in range(val_rmse.shape[-1]):
        logging.info(f"RMSE of {OUT_VARIABLES[i]} is: {np.mean(val_rmse, axis=0)[:, i]}")
        logging.info(f"ACC of {OUT_VARIABLES[i]} is: {np.mean(val_acc, axis=0)[:, i]}")
        logging.info(f"Activity of {OUT_VARIABLES[i]} is: {np.mean(val_activity, axis=0)[:, i]}")
    
    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/rmse_{model_name}_forecast.npy", val_rmse)
    np.save(f"{output_dir}/acc_{model_name}_forecast.npy", val_acc)
    np.save(f"{output_dir}/activity_{model_name}_forecast.npy", val_activity)
    
    # xr_forecsat_sfc = []
    # xr_forecsat_pl = []
    # for idx, var in enumerate(RAW_VARIABLES):
    #     if var in ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure"]:
    #         xr_tmp = xr.DataArray(
    #             (val_forecsat[:,:,VARIABLES.index(var)] * std[:,VARIABLES.index(var):VARIABLES.index(var)+1] + mean[:,VARIABLES.index(var):VARIABLES.index(var)+1]).astype(np.float32),
    #             dims=["time", "lead_time", "lat", "lon"],
    #             coords={
    #                 'time': final_time[ics],
    #                 "lead_time": np.asarray(np.arange(0, forecast_hours + 1, dt)).astype(np.float32),
    #                 'lat': lat.astype(np.float32),
    #                 'lon': lon.astype(np.float32)
    #             },
    #             name=NAME_TO_VAR[var]
    #         )
    #         xr_forecsat_sfc.append(xr_tmp)
    #     else:
    #         xr_tmp = xr.DataArray(
    #             (val_forecsat[:,:,4+(idx-4)*len(DEFAULT_PRESSURE_LEVELS):4+(idx-3)*len(DEFAULT_PRESSURE_LEVELS)] * np.expand_dims(std[:,4+(idx-4)*len(DEFAULT_PRESSURE_LEVELS):4+(idx-3)*len(DEFAULT_PRESSURE_LEVELS)], axis=0) + np.expand_dims(mean[:,4+(idx-4)*len(DEFAULT_PRESSURE_LEVELS):4+(idx-3)*len(DEFAULT_PRESSURE_LEVELS)], axis=0)).astype(np.float32),
    #             dims=["time", "lead_time", "lev", "lat", "lon"],
    #             coords={
    #                 'time': final_time[ics],
    #                 "lead_time": np.asarray(np.arange(0, forecast_hours + 1, dt)).astype(np.float32),
    #                 "lev": np.asarray(DEFAULT_PRESSURE_LEVELS).astype(np.float32),
    #                 'lat': lat.astype(np.float32),
    #                 'lon': lon.astype(np.float32)
    #             },
    #             name=NAME_TO_VAR[var]
    #         )
    #         xr_forecsat_pl.append(xr_tmp)
            
    # xr_forecsat_sfc = xr.merge(xr_forecsat_sfc)
    # print(xr_forecsat_sfc)
    # xr_forecsat_pl = xr.merge(xr_forecsat_pl)
    # print(xr_forecsat_pl)
    # xr_forecsat_pl.to_netcdf(os.path.join(output_dir, f"{model_name}_forecast_pl_1.40625deg.nc"))
    # xr_forecsat_sfc.to_netcdf(os.path.join(output_dir, f"{model_name}_forecast_sfc_1.40625deg.nc"))
    
def prepare_parser():
    parser = argparse.ArgumentParser(description='Inference for prediction and assimilation loop!')

    parser.add_argument(
        '--data_dir',
        type=str,
        help='path of the validation data',
        default="../../data/coams"
    )
    
    parser.add_argument(
        '--resolution',
        type=float,
        help='resolution of the data',
        default=1.5
    )

    parser.add_argument(
        '--years',
        type=list,
        help='start index of the dataset',
        default=[2020]
    )

    parser.add_argument(
        '--pretrain_dir',
        type=str,
        help='path for pretrain prediction models',
        default='../ckpt_zr'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        help='path for output',
        default='../../data/results_zr'
    )

    parser.add_argument(
        '--forecast_days',
        type=int,
        help='length of the forecasting exp [d]',
        default=30
    )
    
    parser.add_argument(
        '--dt',
        help='time interval',
        default=1,
    )

    parser.add_argument(
        '--decorrelation_days',
        type=int,
        help='decoorelation between each initial time [d]',
        default=30
    )

    parser.add_argument(
        '--model_name',
        type=str,
        help='method used to do forecasting',
        default='sformer_pretrain'
    )

    return parser


if __name__ == '__main__':
    parser = prepare_parser()
    args = parser.parse_args()
    data_dir = args.data_dir
    resolution = args.resolution
    years = args.years
    pretrain_ckpt = args.pretrain_dir
    output_dir = args.output_dir
    forecast_days = args.forecast_days
    dt = args.dt
    decorrelation_days = args.decorrelation_days
    model_name = args.model_name
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    forecast_model_inference(data_dir,
                            resolution,
                            years,
                            pretrain_ckpt,
                            output_dir,
                            forecast_days,
                            dt,
                            decorrelation_days,
                            model_name,
                            device)
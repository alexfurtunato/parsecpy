import os
import numpy as np
import xarray as xr

def test_enviroment():
    try:
        p_str = str(type(get_ipython()))
        'zmqshell' in str(type(p_str))
    except:
        print('Error')


def data_detach(data):
    """
    Detach the independent and dependent variables from DataArray.

    :param data: A xarray DataArray with data to detach
    :return: Tuple with the variables x and y.
    """

    x = []
    y = []
    data_serie = data.to_series()
    for i in data_serie.iteritems():
        x.append(i[0])
        y.append(i[1])
    xnp = np.array(x)
    ynp = np.array(y)
    return {'x': xnp, 'y': ynp, 'dims': data.dims}


def data_attach(data, dims):
    """
    Build a xarray DataArray from tuple with independent
    and dependent variables.

    :param data: A tuple of two lists: input values and output values
    :param dims: Tuple of strings with dimensions
    :return: DataArray of data.
    """

    xnp = np.array(data[0])
    ynp = np.array(data[1])
    coords = []
    shape = []
    for i,d in enumerate(dims):
        x = sorted(np.unique(xnp[:, i]), key=int)
        coords.append((d, x))
        shape.append(len(x))
    data_da = xr.DataArray(ynp.reshape(tuple(shape)), coords=coords)
    return data_da
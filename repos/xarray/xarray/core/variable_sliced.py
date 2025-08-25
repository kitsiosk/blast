import copy
import functools
import itertools
import numbers
import warnings
from collections import defaultdict
from datetime import timedelta
from distutils.version import LooseVersion
from typing import (
    Any,
    Dict,
    Hashable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd

import xarray as xr  # only for Dataset and DataArray

from . import arithmetic, common, dtypes, duck_array_ops, indexing, nputils, ops, utils
from .indexing import (
    BasicIndexer,
    OuterIndexer,
    PandasIndexAdapter,
    VectorizedIndexer,
    as_indexable,
)
from .npcompat import IS_NEP18_ACTIVE
from .options import _get_keep_attrs
from .pycompat import (
    cupy_array_type,
    dask_array_type,
    integer_types,
    is_duck_dask_array,
)
from .utils import (
    OrderedSet,
    _default,
    decode_numpy_dict_values,
    drop_dims_from_indexers,
    either_dict_or_kwargs,
    ensure_us_time_resolution,
    infix_dims,
    is_duck_array,
    maybe_coerce_to_str,
)

NON_NUMPY_SUPPORTED_ARRAY_TYPES = (
    (
        indexing.ExplicitlyIndexed,
        pd.Index,
    )
    + dask_array_type
    + cupy_array_type
)
# https://github.com/python/mypy/issues/224
BASIC_INDEXING_TYPES = integer_types + (slice,)  # type: ignore

VariableType = TypeVar("VariableType", bound="Variable")

    # inherits from ValueError for backward compatibility
    # TODO: move this to an xarray.exceptions module?

def as_compatible_data(data, fastpath=False):
    """Prepare and wrap data to put in a Variable.

    - If data does not have the necessary attributes, convert it to ndarray.
    - If data has dtype=datetime64, ensure that it has ns precision. If it's a
      pandas.Timestamp, convert it to datetime64.
    - If data is already a pandas or xarray object (other than an Index), just
      use the values.

    Finally, wrap it up with an adapter if necessary.
    """
    if fastpath and getattr(data, "ndim", 0) > 0:
        # can't use fastpath (yet) for scalars
        return _maybe_wrap_data(data)

    if isinstance(data, Variable):
        return data.data

    if isinstance(data, NON_NUMPY_SUPPORTED_ARRAY_TYPES):
        return _maybe_wrap_data(data)

    if isinstance(data, tuple):
        data = utils.to_0d_object_array(data)

    if isinstance(data, pd.Timestamp):
        # TODO: convert, handle datetime objects, too
        data = np.datetime64(data.value, "ns")

    if isinstance(data, timedelta):
        data = np.timedelta64(getattr(data, "value", data), "ns")

    # we don't want nested self-described arrays
    data = getattr(data, "values", data)

    if isinstance(data, np.ma.MaskedArray):
        mask = np.ma.getmaskarray(data)
        if mask.any():
            dtype, fill_value = dtypes.maybe_promote(data.dtype)
            data = np.asarray(data, dtype=dtype)
            data[mask] = fill_value
        else:
            data = np.asarray(data)

    if not isinstance(data, np.ndarray):
        if hasattr(data, "__array_function__"):
            if IS_NEP18_ACTIVE:
                return data
            else:
                raise TypeError(
                    "Got an NumPy-like array type providing the "
                    "__array_function__ protocol but NEP18 is not enabled. "
                    "Check that numpy >= v1.16 and that the environment "
                    'variable "NUMPY_EXPERIMENTAL_ARRAY_FUNCTION" is set to '
                    '"1"'
                )

    # validate whether the data is valid data types.
    data = np.asarray(data)

    if isinstance(data, np.ndarray):
        if data.dtype.kind == "O":
            data = _possibly_convert_objects(data)
        elif data.dtype.kind == "M":
            data = _possibly_convert_objects(data)
        elif data.dtype.kind == "m":
            data = _possibly_convert_objects(data)

    return _maybe_wrap_data(data)

# for backwards compatibility
Coordinate = utils.alias(IndexVariable, "Coordinate")
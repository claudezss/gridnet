import tempfile
from pathlib import Path

import numpy as np
import pandapower as pp
import pandas as pd
from pandapower.control import ConstControl
from pandapower.networks import create_cigre_network_mv
from pandapower.timeseries import OutputWriter, run_timeseries
from pandapower.timeseries.data_sources.frame_data import DFData

from gridnet import ROOT


def run(
    net: pp.pandapowerNet = create_cigre_network_mv(with_der="pv_wind"),
    n_ts: int = 8760,
    output_dir: Path = ROOT.parent / "data",
) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)

    timestamp = pd.date_range("2024-01-01", periods=n_ts, freq="h")

    df = (
        pd.DataFrame(
            np.random.uniform(0, 2, size=(n_ts, len(net.load.index))),
            index=list(range(n_ts)),
            columns=net.load.index,
        )
        * net.load.p_mw.values
    )
    ds = DFData(df)

    _ = ConstControl(
        net,
        element="load",
        element_index=net.load.index,
        variable="p_mw",
        data_source=ds,
        profile_name=net.load.index,
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        ow = OutputWriter(
            net, output_path=tmpdir, output_file_type=".csv", csv_separator=","
        )
        ow.log_variable("res_bus", "vm_pu")
        ow.log_variable("res_bus", "va_degree")
        ow.log_variable("res_load", "p_mw")
        ow.log_variable("res_load", "q_mvar")

        run_timeseries(net)

        vm_pu = pd.read_csv(tmpdir / "res_bus" / "vm_pu.csv", index_col=0)
        vm_pu.columns = net.bus.name.values
        vm_pu["timestamp"] = timestamp

        va_degree = pd.read_csv(tmpdir / "res_bus" / "va_degree.csv", index_col=0)
        va_degree.columns = net.bus.name.values
        va_degree["timestamp"] = timestamp

        p_mw = pd.read_csv(tmpdir / "res_load" / "p_mw.csv", index_col=0)
        p_mw.columns = net.load.name.values
        p_mw["timestamp"] = timestamp

        q_mvar = pd.read_csv(tmpdir / "res_load" / "q_mvar.csv", index_col=0)
        q_mvar.columns = net.load.name.values
        q_mvar["timestamp"] = timestamp

        vm_pu.to_parquet(output_dir / "vm_pu.parquet", index=False)
        va_degree.to_parquet(output_dir / "va_degree.parquet", index=False)
        p_mw.to_parquet(output_dir / "p_mw.parquet", index=False)
        q_mvar.to_parquet(output_dir / "q_mvar.parquet", index=False)


run()

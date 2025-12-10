import pyarrow as pa

Base = [
    ("network_name", pa.string()),
    ("feeder_name", pa.string()),
]

ShuntInformationSchema = pa.schema(
    [
        *Base,
        ("from_node", pa.string()),
        ("asset_id", pa.string()),
        ("asset_type", pa.string()),
        ("phase", pa.string()),
        ("p_a_kw", pa.float32()),
        ("q_a_kvar", pa.float32()),
        ("p_b_kw", pa.float32()),
        ("q_b_kvar", pa.float32()),
        ("p_c_kw", pa.float32()),
        ("q_c_kvar", pa.float32()),
    ]
)


ShuntPFResultSchema = pa.schema(
    [
        *Base,
        ("asset_id", pa.string()),
        ("timestamp", pa.TimestampType("ms")),
        ("p_a_kw", pa.float32()),
        ("q_a_kvar", pa.float32()),
        ("p_b_kw", pa.float32()),
        ("q_b_kvar", pa.float32()),
        ("p_c_kw", pa.float32()),
        ("q_c_kvar", pa.float32()),
    ]
)

NodeInformationSchema = pa.schema(
    [
        *Base,
        ("node_id", pa.string()),
        ("phase", pa.string()),
        ("base_v_kv", pa.float32()),
    ]
)

NodePFResultSchema = pa.schema(
    [
        *Base,
        ("node_id", pa.string()),
        ("timestamp", pa.TimestampType("ms")),
        ("v_mag_a_pu", pa.float32()),
        ("v_mag_b_pu", pa.float32()),
        ("v_mag_c_pu", pa.float32()),
        ("v_ang_a_rad", pa.float32()),
        ("v_ang_b_rad", pa.float32()),
        ("v_ang_c_rad", pa.float32()),
    ]
)

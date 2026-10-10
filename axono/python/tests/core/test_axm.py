"""axono.format (.axm 权重容器) 测试。"""

import os

import numpy as np
import pytest

from axono.format import AxmReader, load_axm, read_axm_index, save_axm


def test_roundtrip(tmp_path):
    p = str(tmp_path / "w.axm")
    sd = {
        "layer.weight": np.random.RandomState(0).randn(64, 32).astype(np.float32),
        "layer.bias": np.arange(64, dtype=np.float32),
        "embed": np.random.RandomState(1).randn(100, 16).astype(np.float16),
    }
    save_axm(sd, p)
    out = load_axm(p)
    assert set(out) == set(sd)
    for k in sd:
        np.testing.assert_array_equal(out[k], sd[k])


def test_dtype_convert_fp16(tmp_path):
    p = str(tmp_path / "w16.axm")
    sd = {"w": np.random.RandomState(2).randn(32, 16).astype(np.float32)}
    save_axm(sd, p, dtype="float16")
    out = load_axm(p)
    assert out["w"].dtype == np.float16
    np.testing.assert_allclose(
        out["w"].astype(np.float32), sd["w"], rtol=1e-3, atol=1e-3
    )
    # fp16 体积约减半
    p32 = str(tmp_path / "w32.axm")
    save_axm(sd, p32)
    assert os.path.getsize(p) < os.path.getsize(p32) * 0.75


def test_reader_streaming(tmp_path):
    p = str(tmp_path / "w.axm")
    sd = {"a": np.ones((8, 8), np.float32), "b": np.zeros((4,), np.float32)}
    save_axm(sd, p, meta={"source": "test"})
    r = AxmReader(p)
    assert sorted(r.names()) == ["a", "b"]
    assert r.meta()["source"] == "test"
    np.testing.assert_array_equal(r.get("b"), sd["b"])


def test_index_only(tmp_path):
    p = str(tmp_path / "w.axm")
    save_axm({"x": np.ones((3, 3), np.float32)}, p)
    idx, data_start = read_axm_index(p)
    assert idx["tensors"][0]["name"] == "x"
    assert data_start > 16


def test_bad_magic(tmp_path):
    p = str(tmp_path / "bad.bin")
    with open(p, "wb") as f:
        f.write(b"XXXX")
    with pytest.raises(ValueError):
        load_axm(p)


def test_large_roundtrip(tmp_path):
    """~100MB 张量 roundtrip (对齐 + 大块路径)。"""
    p = str(tmp_path / "big.axm")
    big = np.random.RandomState(3).randn(2048, 12288).astype(np.float32)
    save_axm({"big": big}, p)
    out = load_axm(p)["big"]
    np.testing.assert_array_equal(out, big)

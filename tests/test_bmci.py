import numpy as np

from satrain_models.bmci_fast import BMCIc
from satrain_models.datamodule import SatRainDataModule


def test_bmci_retrieval_no_cutoff(tmp_path, satrain_test_data):
    """
    Test that a simple GMI retrieval yields reasonable correlations
    and can be saved.
    """
    data_module = SatRainDataModule(
        config={
            "base_sensor": "gmi",
            "geometry": "on_swath",
            "format": "tabular",
            "subset": "xl",
            "retrieval_input": [{
                "name": "gmi",
                "include_angles": False
            }]
        }
    )
    X, y = data_module.load_tabular_data("validation")

    sigma = np.ones(13)

    bmci = BMCIc(sigma, cutoff=3)
    bmci.fit(X, y)
    inds = np.random.permutation(bmci.y.size)[:1_000]
    y_ref = bmci.y[inds]
    y_ret = bmci.predict(bmci.X[inds])
    valid = np.isfinite(y_ref) * np.isfinite(y_ret)
    corr = np.corrcoef(y_ref[valid], y_ret[valid])[0, 1]
    assert 0.5 < corr

    bmci_no_cutoff = BMCIc(sigma, cutoff=None)
    bmci_no_cutoff.fit(X, y)
    y_ret_no_cutoff = bmci_no_cutoff.predict(bmci.X[inds])

    assert np.isclose(y_ret_no_cutoff[valid], y_ret[valid], rtol=1e-3).all()

    bmci.save(tmp_path / "bmci.nc")

    bmci_loaded = BMCIc.load(tmp_path / "bmci.nc")
    y_ret_2 = bmci_loaded.predict(bmci.X[inds])

    assert np.isclose(y_ret[valid], y_ret_2[valid], rtol=1e-3).all()

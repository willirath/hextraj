"""Download external datasets into external/.

Re-running is a no-op if the file is already present and the checksum matches.
Open the downloaded zarr.zip directly with xarray:

    import zarr, xarray as xr
    ds = xr.open_zarr(zarr.ZipStore("external/cape_verde_drift_trajectories_1993-2017.zarr.zip"))
"""
import pooch

pooch.retrieve(
    url="https://zenodo.org/records/6826071/files/cape_verde_drift_trajectories_1993-2017.zarr.zip?download=1",
    known_hash="md5:9f9c38bfc15c0d0797ad68c97951cf4d",
    fname="cape_verde_drift_trajectories_1993-2017.zarr.zip",
    path="external/",
)

print("Done. Open with:")
print("  import zarr, xarray as xr")
print('  store = zarr.ZipStore("external/cape_verde_drift_trajectories_1993-2017.zarr.zip")')
print('  ds = xr.open_zarr(store, group="PH_IMA1_stokes-True_consistent_run_subset.zarr", consolidated=False)')

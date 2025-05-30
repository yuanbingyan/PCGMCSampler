import astra
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))

vol_geom = astra.create_vol_geom(512, 512)

"""
_, point_source = astra.data2d.shepp_logan(
    vol_geom,
    modified=True
)
"""
size = vol_geom['GridRowCount']
initialization = np.zeros(
    (size, size),
    dtype = np.float32
)

initialization[125, 50] = 1.
point_source = scipy.ndimage.gaussian_filter(
    initialization, 
    sigma=10
)
# point_source = np.rot90(point_source)

plt.subplot(1, 3, 1)
plt.title(
    "Original Point Source",
    fontsize=30
    )
plt.imshow(point_source, cmap='gray')

proj_geom = astra.create_proj_geom(
            'parallel', 
            1., 
            512, 
            np.linspace(
                0, 
                np.pi, 
                180,
                endpoint=False)
            )

proj_id = astra.create_projector(
            'line',
            proj_geom,
            vol_geom
          )


sino_id, sinogram = astra.create_sino(
    point_source,
    proj_id
    )
# sinogram = np.rot90(sinogram)


plt.subplot(1, 3, 2)
plt.title(
    "Sinogram",
    fontsize=30
    )
plt.imshow(sinogram, cmap='gray', aspect='auto')

recon_id = astra.data2d.create(
                            '-vol',
                            vol_geom,
                            0
                        )

config = astra.astra_dict('FBP')
config['ProjectorId']= proj_id
config['ProjectionDataId'] = sino_id
config['ReconstructionDataId'] = recon_id

alg_id = astra.algorithm.create(config)

astra.algorithm.run(alg_id)

reconstruction = astra.data2d.get(recon_id)
# reconstruction = np.rot90(reconstruction)

plt.subplot(1, 3, 3)
plt.title(
    "Reconstruction",
    fontsize=30
    )
plt.imshow(reconstruction, cmap='gray')

plt.tight_layout()
plt.show()

astra.clear()
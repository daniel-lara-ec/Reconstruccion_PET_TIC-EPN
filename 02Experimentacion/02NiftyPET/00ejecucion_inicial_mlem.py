import logging

# Guardar los datos
import pickle

from niftypet import nipet
import time as tm

from os import path
import time as tm
import functools
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from tqdm.auto import trange

logging.basicConfig(level=logging.WARN)
mMRpars = nipet.get_mmrparams()

# Medidas para calcular las diferencias
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)


# ## Funciones adicionales


def div_nzer(x, y):
    return np.divide(x, y, out=np.zeros_like(y), where=y != 0)


# Ab_PET_mMR_test
folderin = "/mnt/test1dev/TIC/00Datos/amyloidPET_FBP_TP0"
folderout = "."  # realtive to `{folderin}/niftyout`
itr = 7  # number of iterations (will be multiplied by 14 for MLEM)
fwhm = 2.5  # mm (for resolution modelling)
totCnt = None  # bootstrap sample (e.g. `300e6`) counts

# datain
folderin = path.expanduser(folderin)

# automatically categorise the input data
datain = nipet.classify_input(folderin, mMRpars, recurse=-1)

# output path
opth = path.join(datain["corepath"], "niftyout")

datain


SIGMA2FWHMmm = (
    (8 * np.log(2)) ** 0.5 * np.array([mMRpars["Cnt"]["SO_VX" + i] for i in "ZYX"]) * 10
)


# ## Mapas de atenuación


mu_h = nipet.hdw_mumap(datain, [1, 2, 4], mMRpars, outpath=opth, use_stored=True)
mu_o = nipet.obj_mumap(datain, mMRpars, outpath=opth, store=True)


# ## Creamos el histograma


# La función mmrhist del módulo NiftyPET se utiliza para procesar datos en modo lista y obtener datos histogramados, también conocidos como sinogramas, además de otras estadísticas de adquisición. Esta función es una parte crucial del flujo de trabajo de procesamiento de datos PET/MR, que convierte grandes cantidades de datos en bruto en formatos más manejables para su análisis.
#
# El uso típico de mmrhist implica proporcionar los datos de entrada categorizados y los parámetros específicos del escáner (mMRpars). Una vez ejecutada, mmrhist genera sinogramas directos y oblicuos que pueden visualizarse para análisis posteriores. Estos sinogramas son esenciales para reconstruir imágenes PET de alta calidad y realizar análisis cuantitativos precisos.
#
# Por ejemplo, para visualizar un sinograma directo generado por mmrhist, se puede elegir un índice de sinograma (por debajo de 127 para sinogramas directos y 127 o más para sinogramas oblicuos) y utilizar herramientas de visualización como matshow de matplotlib para observar los datos​


mMRpars["Cnt"]["BTP"] = 0
m = nipet.mmrhist(datain, mMRpars, outpath=opth, store=True, use_stored=True)

# ## Pasos previos para la reconstrucción

A = nipet.frwd_prj(mu_h["im"] + mu_o["im"], mMRpars, attenuation=True)
N = nipet.mmrnorm.get_norm_sino(datain, mMRpars, m)
AN = A * N


# Calculate forward projection for the provided input image.
sim = nipet.back_prj(AN, mMRpars)
msk = (
    nipet.img.mmrimg.get_cylinder(
        mMRpars["Cnt"], rad=29.0, xo=0.0, yo=0.0, unival=1, gpu_dim=False
    )
    <= 0.9
)

# Get the estimated sinogram of random events using the delayed event measurement.  The delayed sinogram is in the histogram dictionary obtained from the processing of the list-mode data.
r = nipet.randoms(m, mMRpars)[0]
print("Randoms: %.3g%%" % (r.sum() / m["psino"].sum() * 100))

eim = nipet.mmrchain(
    datain, mMRpars, mu_h=mu_h, mu_o=mu_o, histo=m, itr=1, outpath=opth
)["im"]

"""
Obtain a scatter sinogram using the mu-maps (hardware and object mu-maps) an estimate of emission image, the prompt measured sinogram, an estimate of the randoms sinogram and a normalisation sinogram.
"""
s = nipet.vsm(datain, (mu_h["im"], mu_o["im"]), eim, mMRpars, m, r)
print("Scatter: %.3g%%" % (s.sum() / m["psino"].sum() * 100))

## Realizamos la reconstrucción

psf = functools.partial(gaussian_filter, sigma=fwhm / SIGMA2FWHMmm)
sim_inv = div_nzer(1, psf(sim))
sim_inv[msk] = 0
rs_AN = div_nzer(r + s, AN)
recon_mlem = [np.ones_like(sim)]


def normalizar_imagen(imagen):
    return (imagen - np.min(imagen)) / (
        np.max(imagen) - np.min(imagen) if np.max(imagen) - np.min(imagen) != 0 else 1
    )


def calcular_parametro_parada(
    reconstruccion_actual,
    reconstruccion_anterior,
    datos_iteraciones,
    modo="full",
    criterio="SSIM",
):

    dato_anterior = reconstruccion_anterior
    dato_actual = reconstruccion_actual

    # normalizamos los datos
    dato_anterior = normalizar_imagen(dato_anterior)
    dato_actual = normalizar_imagen(dato_actual)

    if modo == "full":
        dimensiones_imagen = dato_anterior.shape
        dimension_1 = dimensiones_imagen[0]
        dimension_2 = dimensiones_imagen[1]
        dimension_3 = dimensiones_imagen[2]

        # Obtenemos 3 muestras igualmente espaciadas en cada dimensión
        muestras_dimension_1 = [
            dimension_1 // 4,
            2 * dimension_1 // 4,
            3 * dimension_1 // 4,
        ]
        muestras_dimension_2 = [
            dimension_2 // 4,
            2 * dimension_2 // 4,
            3 * dimension_2 // 4,
        ]
        muestras_dimension_3 = [
            dimension_3 // 4,
            2 * dimension_3 // 4,
            3 * dimension_3 // 4,
        ]

        # Calculamos los criterios en cada dimensión
        # MSE
        criterio_MSE_dimension_1 = []

        for i in muestras_dimension_1:
            criterio_MSE_dimension_1.append(
                mean_squared_error(dato_anterior[i, :, :], dato_actual[i, :, :])
            )

        criterio_MSE_1 = np.mean(criterio_MSE_dimension_1)

        criterio_MSE_dimension_2 = []
        for i in muestras_dimension_2:
            criterio_MSE_dimension_2.append(
                mean_squared_error(dato_anterior[:, i, :], dato_actual[:, i, :])
            )

        criterio_MSE_2 = np.mean(criterio_MSE_dimension_2)

        criterio_MSE_dimension_3 = []
        for i in muestras_dimension_3:
            criterio_MSE_dimension_3.append(
                mean_squared_error(dato_anterior[:, :, i], dato_actual[:, :, i])
            )

        criterio_MSE_3 = np.mean(criterio_MSE_dimension_3)

        criterio_MSE = np.mean([criterio_MSE_1, criterio_MSE_2, criterio_MSE_3])

        print(criterio_MSE)

        # PSNR
        criterio_PSNR_dimension_1 = []
        for i in muestras_dimension_1:
            criterio_PSNR_dimension_1.append(
                peak_signal_noise_ratio(
                    dato_anterior[i, :, :], dato_actual[i, :, :], data_range=1
                )
            )

        criterio_PSNR_1 = np.mean(criterio_PSNR_dimension_1)

        criterio_PSNR_dimension_2 = []

        for i in muestras_dimension_2:
            criterio_PSNR_dimension_2.append(
                peak_signal_noise_ratio(
                    dato_anterior[:, i, :], dato_actual[:, i, :], data_range=1
                )
            )

        criterio_PSNR_2 = np.mean(criterio_PSNR_dimension_2)

        criterio_PSNR_dimension_3 = []

        for i in muestras_dimension_3:
            criterio_PSNR_dimension_3.append(
                peak_signal_noise_ratio(
                    dato_anterior[:, :, i], dato_actual[:, :, i], data_range=1
                )
            )

        criterio_PSNR_3 = np.mean(criterio_PSNR_dimension_3)

        criterio_PSNR = np.mean([criterio_PSNR_1, criterio_PSNR_2, criterio_PSNR_3])

        # SSIM

        criterio_SSIM_dimension_1 = []

        for i in muestras_dimension_1:
            criterio_SSIM_dimension_1.append(
                structural_similarity(
                    dato_anterior[i, :, :], dato_actual[i, :, :], data_range=1
                )
            )

        print(criterio_SSIM_dimension_1)

        criterio_SSIM_1 = np.mean(criterio_SSIM_dimension_1)

        criterio_SSIM_dimension_2 = []

        for i in muestras_dimension_2:
            criterio_SSIM_dimension_2.append(
                structural_similarity(
                    dato_anterior[:, i, :], dato_actual[:, i, :], data_range=1
                )
            )

        criterio_SSIM_2 = np.mean(criterio_SSIM_dimension_2)

        criterio_SSIM_dimension_3 = []

        for i in muestras_dimension_3:
            criterio_SSIM_dimension_3.append(
                structural_similarity(
                    dato_anterior[:, :, i], dato_actual[:, :, i], data_range=1
                )
            )

        criterio_SSIM_3 = np.mean(criterio_SSIM_dimension_3)

        criterio_SSIM = np.mean([criterio_SSIM_1, criterio_SSIM_2, criterio_SSIM_3])

    datos_criterios = {
        "MSE": criterio_MSE,
        "PSNR": criterio_PSNR,
        "SSIM": criterio_SSIM,
    }

    diferenica_MSE = np.abs(datos_iteraciones[-1]["MSE"] - criterio_MSE)
    diferenica_PSNR = np.abs(datos_iteraciones[-1]["PSNR"] - criterio_PSNR)
    diferenica_SSIM = np.abs(datos_iteraciones[-1]["SSIM"] - criterio_SSIM)

    valor_criterio_evaluacion = 0

    if criterio == "MSE":
        valor_criterio_evaluacion = diferenica_MSE
    elif criterio == "PSNR":
        valor_criterio_evaluacion = diferenica_PSNR
    elif criterio == "SSIM":
        valor_criterio_evaluacion = diferenica_SSIM

    return datos_criterios, valor_criterio_evaluacion


datos_iteraciones = [
    {
        "iteracion": -1,
        "tiempo_iteracion": 0,
        "MSE": 0,
        "PSNR": 0,
        "SSIM": 0,
    }
]

criterio_parada_mlem = 1
recon_mlem = []

reconstruccion_anterior = np.ones_like(sim)

for iteracion in trange(1, 1001):

    t0 = tm.time()

    fprj = nipet.frwd_prj(psf(reconstruccion_anterior), mMRpars) + rs_AN
    reconstruccion_actual = (
        reconstruccion_anterior
        * sim_inv
        * psf(nipet.back_prj(div_nzer(m["psino"], fprj), mMRpars))
    )

    # guardamos la reconstrucción cada 10 iteraciones
    if iteracion % 10 == 0:
        recon_mlem.append(reconstruccion_actual)

    tiempo_final = tm.time() - t0

    datos_criterios_mlem, criterio_parada_mlem = calcular_parametro_parada(
        reconstruccion_actual, reconstruccion_anterior, datos_iteraciones
    )

    reconstruccion_anterior = reconstruccion_actual

    datos_criterios_mlem["iteracion"] = iteracion
    datos_criterios_mlem["tiempo_iteracion"] = tiempo_final

    print(datos_criterios_mlem)

    datos_iteraciones.append(datos_criterios_mlem)

    iteracion += 1

# guardamos los datos obtenidos de las reconstrucciones: recon_mlem
with open(
    "datos_resultados/resultados_metrica_completa/recon_mlem_inicial.pickle", "wb"
) as handle:
    pickle.dump(recon_mlem, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Guardamos los datos de los criterios de parada: datos_criterios_parada
with open(
    "datos_resultados/resultados_metrica_completa/datos_iteraciones_inicial.pickle",
    "wb",
) as handle:
    pickle.dump(datos_iteraciones, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 🎭 Facial Expression Fairness: Auditoría y Mitigación de Sesgos en FER
![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-EE4C2C.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **Trabajo de Fin de Máster (TFM)** - Máster en Inteligencia Artificial, Universidad Politécnica de Madrid (UPM).
> **Autor:** Mario Gutiérrez López

- [🎭 Facial Expression Fairness: Auditoría y Mitigación de Sesgos en FER](#-facial-expression-fairness-auditoría-y-mitigación-de-sesgos-en-fer)
  - [Descripción del Proyecto](#descripción-del-proyecto)
  - [Memoria del Proyecto](#memoria-del-proyecto)
  - [Instalación](#instalación)
  - [Uso y ejecución](#uso-y-ejecución)




## Descripción del Proyecto
Este repositorio contiene el código fuente para la auditoría y mitigación de sesgos demográficos e iluminativos en sistemas de Reconocimiento de Expresiones Faciales (FER - *Facial Expression Recognition*). 

El objetivo principal es analizar cómo factores in-the-wild (como el *Face Skin Brightness* o el género) afectan a la distribución del espacio latente en redes neuronales profundas y proponer soluciones geométricas y de balanceo de clases.

## Memoria del Proyecto
El documento completo de la tesis se encuentra en fase de redacción. **[Puedes consultar el borrador en vivo aquí](https://es.overleaf.com/read/grfqztcmdkdf#494df4)**

## Instalación
Clona el repositorio e instala las dependencias (se recomienda usar un entorno virtual con Conda):

```bash
git clone https://github.com/mariogutierrezlopez/facial-expression-fairness.git
cd facial-expression-fairness

# Instalar dependencias
pip install -r requirements.txt
```

## Uso y ejecución
El proyecto está centralizado en el script `main.py`

Para entrenar el modelo baseline, el flag `--dataset` define el dataset que se va a utilizar, se pueden utilizar los valores `multipie`, `affectnet` o `affwild2`

```bash
python -m src.main --dataset=affectnet
```
Para saltar el entrenamiento y extraer los vectores latentes de un modelo, se utilizan los flags `--test_only` y `--ckpt_path`

```bash
python src/main.py --dataset=affectnet --test_only --ckpt_path=ruta`
```
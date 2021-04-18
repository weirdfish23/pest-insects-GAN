# pest-insects-GAN

## Environment Setup

Activar conda\*

`source $HOME/miniconda3/bin/activate`

Crear el virtual env "env"

`conda create -n pests python=3.8.5 anaconda`

Activar el virutal env

`conda activate pests`

Instalar las librerias y dependencias

`pip install -r requirements.txt`

Revisar el virtual env

`which python`

Descativar el virtual env

`conda deactivate`

## Descargar dataset base

`python3 dowload_base_data.py`

## Aumentar data con técnicas tradicionales

`python3 traditional_data_augmentation.py`

---

By Joel Cabrera Rios

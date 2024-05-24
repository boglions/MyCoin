import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def converti_data_in_float(data):
    timestamp = data.timestamp()
    valore_float = (timestamp / (2 * 10**10)) * 5000
    return valore_float


def get_dataset():
    dataset_train = pd.read_csv('risorse/dataset/dataframe_train serie temporale.csv', sep=',')
    dataset_train.drop('Unnamed: 0', axis=1, inplace=True)
    dataset_validation = pd.read_csv('risorse/dataset/dataframe_validation serie temporale.csv', sep=',')
    dataset_validation.drop('Unnamed: 0', axis=1, inplace=True)
    dataset_test = pd.read_csv('risorse/dataset/dataframe_test serie temporale.csv', sep=',')
    dataset_test.drop('Unnamed: 0', axis=1, inplace=True)

    dataset_predizioni = pd.read_excel('risorse/dataset/dati da predire.xlsx')
    dataset_predizioni['Data'] = pd.to_datetime(dataset_predizioni['Data'])
    dataset_predizioni['Ultimo'] = pd.to_numeric(dataset_predizioni['Ultimo'], downcast='float')
    dataset_predizioni['Data'] = dataset_predizioni['Data'].apply(converti_data_in_float)
    dataset_predizioni['10_giorni_precedenti_Ultimo'] = pd.to_numeric(dataset_predizioni['10_giorni_precedenti_Ultimo'],
                                                                      downcast='float')
    dataset_predizioni['20_giorni_precedenti_Ultimo'] = pd.to_numeric(dataset_predizioni['20_giorni_precedenti_Ultimo'],
                                                                      downcast='float')
    dataset_predizioni['30_giorni_precedenti_Ultimo'] = pd.to_numeric(dataset_predizioni['30_giorni_precedenti_Ultimo'],
                                                                      downcast='float')

    dataframes = [dataset_train, dataset_validation, dataset_test, dataset_predizioni]
    dataset = pd.concat(dataframes, axis=0, ignore_index=True)
    dataset.to_csv('risorse/dataset/ dataset per normalizzatore.csv', columns=dataset.columns, index=False)
    return dataset


normalizzatore = MinMaxScaler(feature_range=(0, 1))
normalizzatore.fit(get_dataset())
dati = pd.DataFrame(normalizzatore.transform(get_dataset()), columns=get_dataset().columns)


def normalizza_dati(dataframe=None):
    dataframe_normalizzato = pd.DataFrame(normalizzatore.transform(dataframe), columns=dataframe.columns)
    return dataframe_normalizzato


def denormalizza_dato(dato=None):
    if dato is not None:
        dato = dato.item()
        dato = np.array([[0, dato, 0, 0, 0]])
        dato_denormalizzato = normalizzatore.inverse_transform(dato)
        dato_denormalizzato = dato_denormalizzato[0, 1]
        return dato_denormalizzato



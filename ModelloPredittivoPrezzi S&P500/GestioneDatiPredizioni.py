import pandas as pd
import Normalizzatore
from LoggerErrori import LoggerErrori
import tensorflow as tf
import numpy as np


class GestioneDatiPredizioni:

    def __init__(self):

        self.log = LoggerErrori().log
        try:
            self.normalizzatore = None
            self.dataset = pd.read_excel('risorse/predizioni/dati da predire.xlsx')
            self.__converti_tipologia_dati()
            self.__normalizza_dati()
            self.__prepara_dataset_predictions()
        except:
            self.log.error("errore: ", exc_info=True)

    def __salva_dati(self, name=None, oggetto=None):
        try:
            oggetto.to_csv(path_or_buf=name, columns=oggetto.columns, index=False)
        # salva su file formato csv i nuovi dati elaborati
        except:
            self.log.error("errore: ", exc_info=True)

    def __converti_data_in_float(self, data):
        try:
            timestamp = data.timestamp()
            valore_float = (timestamp / (2 * 10 ** 10)) * 5000
            return valore_float
        except:
            self.log.error("errore: ", exc_info=True)

    def __converti_tipologia_dati(self):  # converte i dati delle colonne da str a float64 per omogenizzare dati
        try:
            self.dataset['Data'] = self.dataset['Data'].apply(self.__converti_data_in_float)  # converte la colonna data
            # in float
            self.dataset['Ultimo'] = pd.to_numeric(self.dataset['Ultimo'], downcast='float')
            self.dataset['10_giorni_precedenti_Ultimo'] = pd.to_numeric(self.dataset['10_giorni_precedenti_Ultimo'], downcast='float')
            self.dataset['20_giorni_precedenti_Ultimo'] = pd.to_numeric(self.dataset['20_giorni_precedenti_Ultimo'], downcast='float')
            self.dataset['30_giorni_precedenti_Ultimo'] = pd.to_numeric(self.dataset['30_giorni_precedenti_Ultimo'], downcast='float')
            self.__salva_dati(name='risorse/dataset/dati_predicitions_convertiti.csv', oggetto=self.dataset)
            self.dataset = self.dataset.copy()
        except:
            self.log.error("errore: ", exc_info=True)

    def __normalizza_dati(self):  # modifica il valore dei dati nell' intervallo [-1,1] in modo da uniformarli senza
        # perdere il segno negativo
        try:
            self.dataset = Normalizzatore.normalizza_dati(self.dataset)
            self.dataset.drop('Ultimo', axis=1, inplace=True)
            self.__salva_dati(name='risorse/dataset/dati_predicitions_normalizzati.csv', oggetto=self.dataset)
            self.dataset = self.dataset.copy()
        except:
            self.log.error("errore: ", exc_info=True)

    def __prepara_dataset_predictions(self):
        try:
            predict = self.dataset
            V = predict.values.reshape(1, 1,
                                       4)  # Num di campioni: 1 Num di passaggi temporali: 1 Num di caratteristiche: 4
            predict = tf.convert_to_tensor(V, dtype=tf.float64)  # converte l'array NumPy 3D in un tensore TensorFlow 3D
            predict_str = tf.io.serialize_tensor(predict)
            filename_predictions = tf.constant('risorse/dataset/dataset_predictions.dataset')
            tf.io.write_file(filename_predictions, predict_str)
        except:
            self.log.error('errore: ', exc_info=True)

    def get_dataset_predictions(self):
        try:
            predict_str = tf.io.read_file(filename='risorse/dataset/dataset_predictions.dataset')
            return tf.io.parse_tensor(serialized=predict_str, out_type=tf.float64)
        except:
            self.log.error("errore: ", exc_info=True)

from LoggerErrori import LoggerErrori
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import Normalizzatore


class ElaborazioneDati:

    def __init__(self):
        self.log = LoggerErrori().log  # inizializzazione log del modulo log4p per la gestione degli errori/warning/info
        try:
            self.normalizzatore = None
            self.dataframe_train = pd.read_csv('risorse/dataset/dataframe_train spx.CSV',
                                               sep=';')  # importa i dati di train dal csv e li salva in un Dataframe
            self.dataframe_validation = pd.read_csv('risorse/dataset/dataframe_validation spx.csv',
                                                    sep=';')  # importa i dati di validation dal csv e li salva in un
            # Dataframe
            self.dataframe_test = pd.read_csv('risorse/dataset/dataframe_test spx.csv',
                                              sep=';')  # importa i dati di test dal csv e li salva in un Dataframe
            self.dataframe_train = self.__oridna_crescente(dataset=self.dataframe_train, train=True)
            self.dataframe_train = self.__converti_tipologia_dati(dataset=self.dataframe_train,
                                                                  name_dataset='dataframe_train')
            self.__crea_serie_temporale_train()

            self.dataframe_validation = self.__oridna_crescente(dataset=self.dataframe_validation)
            self.dataframe_validation = self.__converti_tipologia_dati(dataset=self.dataframe_validation,
                                                                       name_dataset='dataframe_validation')
            self.__crea_serie_temporale_validation()

            self.dataframe_test = self.__oridna_crescente(dataset=self.dataframe_test)
            self.dataframe_test = self.__converti_tipologia_dati(dataset=self.dataframe_test,
                                                                 name_dataset='dataframe_test')
            self.__crea_serie_temporale_test()

            self.__normalizza_dati()
            self.__prepara_dataset_train()
            self.__prepara_dataset_validation()
            self.__prepara_dataset_test()


        except:
            self.log.error("errore: ", exc_info=True)
        finally:
            self.log.debug('dataset prezzi bitcoin 2017 2023.csv file letto correttamente')
            self.log.debug('dati del dataset modificati correttamente')
            self.log.debug('dati del dataset convertiti correttamente')
            self.log.debug('dati convertiti salvati su file')
            self.log.debug('dati del dataset normalizzati correttamente')
            self.log.debug('dati convertiti salvati su file')
            self.log.debug('database suddivisi correttamente')

    def __oridna_crescente(self, dataset=None,
                           train=False):  # ordina il dataset dalla data più vecchia alla più nuova
        try:
            if train is not False:
                dataset['Data'] = pd.to_datetime(pd.to_datetime(dataset['Data'], format='%d-%b-%y').dt.strftime(
                    '%Y-%m-%d'))
            else:
                dataset['Data'] = pd.to_datetime(
                    pd.to_datetime(dataset['Data'], format='%d.%m.%Y').dt.strftime('%Y-%m-%d'))

            dataset = dataset.sort_values(by='Data', ascending=True)
            return dataset.copy()
        except:
            self.log.error("errore: ", exc_info=True)

    def __converti_data_in_float(self, data):
        try:
            timestamp = data.timestamp()
            valore_float = (timestamp / (2 * 10 ** 10)) * 5000
            return valore_float
        except:
            self.log.error("errore: ", exc_info=True)

    def __converti_tipologia_dati(self, dataset=None,
                                  name_dataset=None,
                                  train=False):  # converte i dati delle colonne da str a float64 per omogenizzare dati

        try:
            dataset['Data'] = dataset['Data'].apply(self.__converti_data_in_float)  # converte la colonna data
            # in float
            dataset['Ultimo'] = pd.to_numeric(dataset['Ultimo'])  # converte in float64

            self.__salva_dati(name=f'risorse/dataset/{name_dataset} con dati convertiti.csv', oggetto=dataset)
            return dataset.copy()
        except:
            self.log.error("errore: ", exc_info=True)

    def __crea_serie_temporale_train(self):  # aggiunge delle colonne al dataframe che per ogni giorno
        # indicano i dati riferiti al 10°, 20°, 30° giorno precedente. La rete utlilizzerà i dati di 10,20,30 giorni
        # precdenti perpredire
        try:
            serie_temporale = self.dataframe_train.copy()
            for i in [10, 20, 30]:  # crea le colonne con i dati di 10,20,30 giorni precedenti per ogni giorno.
                # il metodo shift() sposta i dati in avanti di i posizioni
                serie_temporale[f'{i}_giorni_precedenti_Ultimo'] = self.dataframe_train['Ultimo'].shift(periods=i)

            row_to_drop = []
            for i in range(0, 30):
                row_to_drop.append(i)
            serie_temporale.drop(serie_temporale.index[:30],
                                 inplace=True)  # elimina le prime 30 righe perchè non dispongono dei
            # dati relativi ai 10, 20, 30 giorni precedenti
            serie_temporale.reset_index(drop=True, inplace=True)
            serie_temporale.to_csv(path_or_buf=f'risorse/dataset/dataframe_train serie temporale.csv',
                                   columns=serie_temporale.columns)
            # salva su file la serie temporale
            self.dataframe_train = serie_temporale.copy()
        except:
            self.log.error("errore: ", exc_info=True)

    def __crea_serie_temporale_validation(self):  # aggiunge delle colonne al dataframe che per ogni giorno
        # indicano i dati riferiti al 10°, 20°, 30° giorno precedente. La rete utlilizzerà i dati di 10,20,30 giorni
        # precdenti perpredire
        try:
            serie_temporale = self.dataframe_validation.copy()
            for i in [10, 20, 30]:  # crea le colonne con i dati di 10,20,30 giorni precedenti per ogni giorno.
                # il metodo shift() sposta i dati in avanti di i posizioni
                serie_temporale[f'{i}_giorni_precedenti_Ultimo'] = self.dataframe_validation['Ultimo'].shift(periods=i)

            row_to_drop = []
            for i in range(0, 30):
                row_to_drop.append(i)
            serie_temporale.drop(serie_temporale.index[:30],
                                 inplace=True)  # elimina le prime 30 righe perchè non dispongono dei
            # dati relativi ai 10, 20, 30 giorni precedenti
            serie_temporale.reset_index(drop=True, inplace=True)
            serie_temporale.to_csv(path_or_buf=f'risorse/dataset/dataframe_validation serie temporale.csv',
                                   columns=serie_temporale.columns)
            # salva su file la serie temporale
            self.dataframe_validation = serie_temporale.copy()
        except:
            self.log.error("errore: ", exc_info=True)

    def __crea_serie_temporale_test(self):  # aggiunge delle colonne al dataframe che per ogni giorno
        # indicano i dati riferiti al 10°, 20°, 30° giorno precedente. La rete utlilizzerà i dati di 10,20,30 giorni
        # precdenti perpredire
        try:
            serie_temporale = self.dataframe_test.copy()
            for i in [10, 20, 30]:  # crea le colonne con i dati di 10,20,30 giorni precedenti per ogni giorno.
                # il metodo shift() sposta i dati in avanti di i posizioni
                serie_temporale[f'{i}_giorni_precedenti_Ultimo'] = self.dataframe_test['Ultimo'].shift(periods=i)

            row_to_drop = []
            for i in range(0, 30):
                row_to_drop.append(i)
            serie_temporale.drop(serie_temporale.index[:30],
                                 inplace=True)  # elimina le prime 30 righe perchè non dispongono dei
            # dati relativi ai 10, 20, 30 giorni precedenti
            serie_temporale.reset_index(drop=True, inplace=True)
            serie_temporale.to_csv(path_or_buf=f'risorse/dataset/dataframe_test serie temporale.csv',
                                   columns=serie_temporale.columns)
            # salva su file la serie temporale
            self.dataframe_test = serie_temporale.copy()
        except:
            self.log.error("errore: ", exc_info=True)

    def __salva_dati(self, name=None, oggetto=None):
        try:
            oggetto.to_csv(path_or_buf=name, columns=oggetto.columns, index=False)
        # salva su file formato csv i nuovi dati elaborati

        except:
            self.log.error("errore: ", exc_info=True)

    def __prepara_dataset_train(self):
        try:
            X = self.dataframe_train.iloc[:, [0, 2, 3, 4]]
            # seleziono le colonne di input
            y = self.dataframe_train['Ultimo']  # seleziono la colonna 'Utlimo' (prezzo del Bitcoin) che rappresenta
            # l'output della rete neurale
            X = X.values  # prendo tutti i valori delle colonne di input e converto in un array NumPy
            y = y.values  # prendo tutti valori della colonna di output e converto in un array NumPy
            X = X.reshape(7151, 1,
                          4)  # Num di campioni: 7151 Num di passaggi temporali: 1 Num di caratteristiche: 4
            y = y.reshape(7151, 1, 1)  # Num di campioni: 7151 Num di passaggi temporali: 1 Num di caratteristiche: 1
            xt = tf.convert_to_tensor(X, dtype=tf.float64)  # converte l'array NumPy 3D in un tensore TensorFlow 3D
            yt = tf.convert_to_tensor(y, dtype=tf.float64)  # converte l'array NumPy 3D in un tensore TensorFlow 3D
            xt_str = tf.io.serialize_tensor(xt)  # serializza gli input
            yt_str = tf.io.serialize_tensor(yt)  # serializza gli output
            filename_xt = tf.constant('risorse/dataset/dataset_train_x.dataset')
            filename_yt = tf.constant('risorse/dataset/dataset_train_y.dataset')
            tf.io.write_file(filename_xt, xt_str)  # salva su file
            tf.io.write_file(filename_yt, yt_str)
        except:
            self.log.error('errore: ', exc_info=True)

    def __prepara_dataset_validation(self):
        try:
            X = self.dataframe_validation.iloc[:, [0, 2, 3, 4]]
            # seleziono le colonne di input
            y = self.dataframe_validation['Ultimo']  # seleziono la colonna 'Utlimo' (prezzo del Bitcoin) che
            # rappresenta l'output
            X = X.values  # prendo tutti i valori delle colonne di input e converto in un array NumPy
            y = y.values  # prendo tutti valori della colonna di output e converto in un array NumPy
            X = X.reshape(474, 1, 4)  # Num di campioni: 474 Num di passaggi temporali: 1 Num di caratteristiche: 4
            y = y.reshape(474, 1, 1)  # Num di campioni: 474 Num di passaggi temporali: 1 Num di caratteristiche: 1
            xv = tf.convert_to_tensor(X, dtype=tf.float64)  # converte l'array NumPy 3D in un tensore TensorFlow 3D
            yv = tf.convert_to_tensor(y, dtype=tf.float64)  # converte l'array NumPy 3D in un tensore TensorFlow 3D
            xv_str = tf.io.serialize_tensor(xv)  # serializza gli input
            yv_str = tf.io.serialize_tensor(yv)  # serializza gli output
            filename_xv = tf.constant('risorse/dataset/dataset_validation_x.dataset')
            filename_yv = tf.constant('risorse/dataset/dataset_validation_y.dataset')
            tf.io.write_file(filename_xv, xv_str)  # salva su file
            tf.io.write_file(filename_yv, yv_str)  # salva su file
        except:
            self.log.error('errore: ', exc_info=True)

    def __prepara_dataset_test(self):
        try:
            X = self.dataframe_test.iloc[:, [0, 2, 3, 4]]
            # seleziono le colonne di input
            y = self.dataframe_test['Ultimo']  # seleziono la colonna 'Utlimo' (prezzo del Bitcoin) che
            # rappresenta l'output
            X = X.values  # prendo tutti i valori delle colonne di input e converto in un array NumPy
            y = y.values  # prendo tutti valori della colonna di output e converto in un array NumPy
            X = X.reshape(222, 1, 4)  # Num di campioni: 222 Num di passaggi temporali: 1 Num di caratteristiche: 4
            y = y.reshape(222, 1, 1)  # Num di campioni: 222 Num di passaggi temporali: 1 Num di caratteristiche: 1
            xtest = tf.convert_to_tensor(X, dtype=tf.float64)  # converte l'array NumPy 3D in un tensore TensorFlow 3D
            ytest = tf.convert_to_tensor(y, dtype=tf.float64)  # converte l'array NumPy 3D in un tensore TensorFlow 3D
            xtest_str = tf.io.serialize_tensor(xtest)  # serializza gli input
            ytest_str = tf.io.serialize_tensor(ytest)  # serializza gli output
            filename_xtest = tf.constant('risorse/dataset/dataset_test_x.dataset')
            filename_ytest = tf.constant('risorse/dataset/dataset_test_y.dataset')
            tf.io.write_file(filename_xtest, xtest_str)  # salva su file
            tf.io.write_file(filename_ytest, ytest_str)  # salva su file
        except:
            self.log.error('errore: ', exc_info=True)

    def __normalizza_dati(self):  # modifica il valore dei dati nell' intervallo [-1,1] in modo da uniformarli senza
        # perdere il segno negativo
        try:
            self.dataframe_train = Normalizzatore.normalizza_dati(self.dataframe_train)
            self.dataframe_validation = Normalizzatore.normalizza_dati(self.dataframe_validation)
            self.dataframe_test = Normalizzatore.normalizza_dati(self.dataframe_test)

            self.__salva_dati(name='risorse/dataset/dataframe_train con dati normalizzati.csv',
                              oggetto=self.dataframe_train)
            self.__salva_dati(name='risorse/dataset/dataframe_validation con dati normalizzati.csv',
                              oggetto=self.dataframe_validation)
            self.__salva_dati(name='risorse/dataset/dataframe_test con dati normalizzati.csv',
                              oggetto=self.dataframe_test)
            self.dataframe_train = self.dataframe_train.copy()
            self.dataframe_validation = self.dataframe_validation.copy()
            self.dataframe_test = self.dataframe_test.copy()

        except:
            self.log.error("errore: ", exc_info=True)

    def get_dataframe_train(self):  # restituisce il dataset di train deserializzandolo
        try:
            x_str = tf.io.read_file(filename='risorse/dataset/dataset_train_x.dataset')
            x = tf.io.parse_tensor(serialized=x_str, out_type=tf.float64)
            y_str = tf.io.read_file(filename='risorse/dataset/dataset_train_y.dataset')
            y = tf.io.parse_tensor(serialized=y_str, out_type=tf.float64)
            return x, y
        except:
            self.log.error("errore: ", exc_info=True)

    def get_dataframe_validation(self):  # restituisce il dataset di convalidation deserializzandolo
        try:
            x_str = tf.io.read_file(filename='risorse/dataset/dataset_validation_x.dataset')
            x = tf.io.parse_tensor(serialized=x_str, out_type=tf.float64)
            y_str = tf.io.read_file(filename='risorse/dataset/dataset_validation_y.dataset')
            y = tf.io.parse_tensor(serialized=y_str, out_type=tf.float64)
            return x, y
        except:
            self.log.error("errore: ", exc_info=True)

    def get_dataframe_test(self):  # restituisce il dataset di test deserializzandolo
        try:
            x_str = tf.io.read_file(filename='risorse/dataset/dataset_test_x.dataset')
            x = tf.io.parse_tensor(serialized=x_str, out_type=tf.float64)
            y_str = tf.io.read_file(filename='risorse/dataset/dataset_test_y.dataset')
            y = tf.io.parse_tensor(serialized=y_str, out_type=tf.float64)
            return x, y
        except:
            self.log.error("errore: ", exc_info=True)

    def get_normalizzatore(self):
        self.normalizzatore = MinMaxScaler(feature_range=(-1, 1))
        return  self.normalizzatore.fit(self.dataframe_train)

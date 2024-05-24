import keras
from ElaborazioneDati import ElaborazioneDati
from LoggerErrori import LoggerErrori


class ReteNeurale:

    def __init__(self):
        self.log = LoggerErrori().log
        try:
            self.xtrain, self.ytrain = None, None
            self.xvalidation, self.yvalidation = None, None
            self.xtest, self.ytest = None, None
            self.model = None
            self.__importa_dataset()
            self.__crea_modello()
            self.__allena_modello()
            self.__valutazione()
            self.salva_modello()

        except:
            self.log.error('errore: ', exc_info=True)
        finally:
            self.log.debug('dataframe importati correttamente')

    def __importa_dataset(self):
        try:
            ed = ElaborazioneDati()
            self.xtrain, self.ytrain = ed.get_dataframe_train()
            self.xvalidation, self.yvalidation = ed.get_dataframe_validation()
            self.xtest, self.ytest = ed.get_dataframe_test()
        except:
            self.log.error('errore: ', exc_info=True)

    def __crea_modello(self):
        try:
            self.model = keras.models.Sequential()  # istanziazione modello sequenziale
            self.model.add(keras.layers.GRU(units=128, return_sequences=True, input_shape=(1, 4)))
            # 128 numero neuroni
            # dropout=0.2: vengono spenti il 20% dei pesi per evitare l'overfitting. (regolarizzazione)
            # return_sequences=True: restituisce l'output del neurone al neurone successivo
            # input_shape(): 1 numero di passaggi temporali
            # 4 numero di caratteristiche per ogni passaggio temporale (features)
            self.model.add(keras.layers.Dropout(0.2))
            self.model.add(keras.layers.GRU(units=128, return_sequences=True))
            self.model.add(keras.layers.Dropout(0.2))
            self.model.add(keras.layers.GRU(units=128))
            self.model.add(keras.layers.Dropout(0.2))
            self.model.add(keras.layers.Dense(units=1, activation='linear'))  # layer con numero di neuroni = output attesi
        except:
            self.log.error('errore: ', exc_info=True)

    def __allena_modello(self):
        try:
            self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
            # aggiunge un ottimizzatore e una funzione di perdita per mitigare l'overfitting
            self.model.fit(self.xtrain, self.ytrain, validation_data=(self.xvalidation, self.yvalidation), epochs=4,
                           batch_size=32)
            # Addestra il modello con dati di train e lo valida con quelli di validazione. L'addestramento viene
            # ripetuto per 10 volte e vengono processate 32 righe per volta

        except:
            self.log.error('errore: ', exc_info=True)

    def __valutazione(self):
        try:
            ris = self.model.evaluate(self.xtest,
                                      self.ytest)  # valuta il modello sui dati di test (mai visti nell'addestramento)
            with open('risorse/test del modello/dati_test.txt', 'w') as f:  # salva i risultati su file
                f.write(str("%s: %.2f%%" % (self.model.metrics_names[1], ris[1] * 100)))
                f.close()
        except:
            self.log.error('errore: ', exc_info=True)

    def salva_modello(self):
        self.model.save('risorse/modello/model')  # salva il modello completo su file con estensione prefendita keras

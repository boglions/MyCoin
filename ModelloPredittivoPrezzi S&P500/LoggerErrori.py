import log4p


class LoggerErrori:  # classe per la gestione degli errori con log4p

    def __init__(self):  # crea un oggetto log con lo scopo di gestire gli errori
        logger = log4p.GetLogger(__name__)
        self.log = logger.logger

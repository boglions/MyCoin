from keras.models import load_model
from ttkthemes import ThemedStyle

import Normalizzatore
from GestioneDatiPredizioni import GestioneDatiPredizioni
import tkinter as tk
from tkinter import ttk
import pandas as pd


class GUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Modello di predizione dati")

        self.style = ThemedStyle(self.root)

        self.df = None
        self.gdp = GestioneDatiPredizioni()
        # Pulsante per caricare il file
        self.load_button = tk.Button(root, text="Carica File", command=self.load_file)
        self.load_button.pack(pady=20)

        # Pulsante per eseguire la previsione
        self.predict_button = tk.Button(root, text="Predict", command=self.predict)
        self.predict_button.pack(pady=20)
        self.predict_button["state"] = "disabled"  # Disabilita il pulsante finché non si carica un file

    def load_file(self):
        self.df = pd.read_excel('risorse/predizioni/dati da predire.xlsx')
        self.df.drop('Ultimo', axis=1, inplace=True)
        # Mostra il DataFrame a video
        self.show_dataframe()

        # Abilita il pulsante di previsione
        self.predict_button["state"] = "normal"

    def show_dataframe(self):
        # Crea una finestra per mostrare il DataFrame
        df_window = tk.Toplevel(self.root)
        df_window.title("DataFrame")

        # Creazione di un widget di tabella per mostrare il DataFrame
        tree = ttk.Treeview(df_window)
        tree["columns"] = list(self.df.columns)
        tree["show"] = "headings"

        for col in tree["columns"]:
            tree.heading(col, text=col, anchor=tk.CENTER)
            tree.column(col, width=100, anchor=tk.CENTER)

        for index, row in self.df.iterrows():
            values = [str(row[col]) for col in self.df.columns]
            tree.insert("", index, values=values)

        # Imposta lo stile della tabella
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TTreeview", font=('Segoe UI', 10))
        style.configure("TTreeview.Heading", font=('Segoe UI', 10, 'bold'))

        vsb = ttk.Scrollbar(df_window, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)

        tree.pack(expand=True, fill="both")
        vsb.pack(side="right", fill="y")

        # Mostra la tabella
        tree.pack(expand=True, fill="both")

    def predict(self):
        model = load_model('risorse/modello/model')
        predictions = model.predict(self.gdp.get_dataset_predictions())
        predictions_transformed = Normalizzatore.denormalizza_dato(predictions)
        # Mostra l'output della previsione
        prediction_window = tk.Toplevel(self.root)
        prediction_window.title("Output della previsione")
        output_label = tk.Label(prediction_window, text=f"Output della previsione: {predictions_transformed}")
        output_label.pack(pady=10)


if __name__ == "__main__":
    root = tk.Tk()
    app = GUIApp(root)
    root.geometry("800x600")
    root.mainloop()





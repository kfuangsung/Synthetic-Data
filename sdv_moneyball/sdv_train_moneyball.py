import numpy as np
import pandas as pd
import openml
from datetime import datetime
from sdv.tabular import CopulaGAN
from sdv.metadata.table import Table


def load_data(id):
    dataset = openml.datasets.get_dataset(id)
    data, _, _, _ = dataset.get_data(dataset_format="dataframe")
    print(data.info())
    return data


def get_moneyball_metadata():
    metadata = {
        "fields": {
            "Team": {"type": "categorical"},
            "League": {"type": "categorical"},
            "Playoffs": {"type": "categorical"},
            "RankSeason": {"type": "categorical"},
            "RankPlayoffs": {"type": "categorical"},
            "G": {"type": "categorical"},
            "Year": {"type": "numerical", "subtype": "integer"},
            "RS": {"type": "numerical", "subtype": "integer"},
            "RA": {"type": "numerical", "subtype": "integer"},
            "W": {"type": "numerical", "subtype": "integer"},
            "OBP": {"type": "numerical", "subtype": "float"},
            "SLG": {"type": "numerical", "subtype": "float"},
            "BA": {"type": "numerical", "subtype": "float"},
            "OOBP": {"type": "numerical", "subtype": "float"},
            "OSLG": {"type": "numerical", "subtype": "float"},
        },
        "constraints": [],
        "model_kwargs": {},
        "name": None,
        "primary_key": None,
        "sequence_index": None,
        "entity_columns": [],
        "context_columns": [],
    }
    table_metadata = Table.from_dict(metadata)
    return table_metadata


def convert_category_data(data):
    # SDV does not understand 'pd.CategoricalDtype'
    # convert to str object
    for c in data.columns:
        if isinstance(data.loc[:, c].dtype, pd.CategoricalDtype):
            data.loc[:, c] = data.loc[:, c].astype(np.dtype("str"))
    return data


def train_model(data, epochs=300, batch_size=500, metadata=None):
    model = CopulaGAN(
        table_metadata=metadata,
        verbose=True,
        epochs=epochs,
        batch_size=batch_size,
        embedding_dim=128,
        generator_dim=(256,) * 5,
        discriminator_dim=(256,) * 5,
        cuda=True,
    )
    model.fit(data)
    return model


def save_model(model, save_path):
    model.save(f"{save_path}.pkl")
    print(f"{save_path} is saved")


def main():
    dataset_id = 41021
    epochs = int(1e5)
    batch_size = 1000
    save_path = "sdv_copulagan_moneyball"

    data = load_data(dataset_id)
    metadata = get_moneyball_metadata()
    data = convert_category_data(data)
    d0 = datetime.now()
    print(f"Start training: {d0}")
    model = train_model(
        data=data, metadata=metadata, epochs=epochs, batch_size=batch_size
    )
    save_model(model, save_path)
    print(f"Timed: {datetime.now()- d0}")


if __name__ == "__main__":
    main()

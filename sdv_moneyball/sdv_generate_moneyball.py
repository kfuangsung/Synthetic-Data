import os
from sdv.tabular import CopulaGAN


def load_model(file_path):
    return CopulaGAN.load(file_path)


def generate_data(model, num_rows, save_path=None):
    if save_path is not None and os.path.exists(save_path):
        os.remove(save_path)
    model.sample(num_rows=num_rows, output_file_path=save_path)


def main():
    model_path = "sdv_copulagan_moneyball.pkl"
    n_rows = int(1e5)
    save_path = "synthetic_moneyball.csv"
    model = load_model(model_path)
    generate_data(model, n_rows, save_path)


if __name__ == "__main__":
    main()

# Flower Example using PyTorch

This project uses Flower and PyTorch to train a federated learning model.

## Project Setup

Start by navigating to the project folder.

```shell
cd creditcard_pooja
```

The project structure is as below:

```shell
-- pyproject.toml
-- client.py
-- server.py
-- README.md
-- cs-training.csv
```

Project dependencies (such as `torch` and `flwr`) are defined in `pyproject.toml`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
python3 -c "import flwr"
```

If you don't see any errors you're good to go!

# Run Federated Learning with PyTorch and Flower

Afterwards you are ready to start the Flower server as well as the clients. You can simply run 

```shell
poetry run python server.py & poetry run python client.py & poetry run python client.py
```

to start the server and two clients in one go.

Alternatively, start the server in a terminal as follows:

```shell
python3 server.py
```

Open two more terminal windows and run the following commands to start the Flower clients which will participate in the learning.
Start client 1 in the first terminal:

```shell
python3 client.py
```

Start client 2 in the second terminal:

```shell
python3 client.py
```

You will see that PyTorch is starting a federated training. Further information on [Flower Quickstarter documentation](https://flower.dev/docs/quickstart_pytorch.html).

# Prism

Query-aware time series compression with exact segment sidecars.

## Setup

```bash
pip install -r requirements.txt
make            # builds prism/libfastcodecs.so and prism/libexactacc.so (gcc)
```

## Run

Download `weather.csv` (or any other dataset of the long-horizon forecasting benchmarks,
e.g. from https://github.com/thuml/Autoformer) into `data/`, then

```bash
python run.py --csv data/weather.csv
```

The script runs the thirteen query classes with the default configuration and reports,
for every format, the latency, the encoded items touched, the bytes fetched and the
values decoded, and checks every answer against the uncompressed data.

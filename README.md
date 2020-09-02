# Auto2Encoder data exploration

## Development

### Setup virtualenv

```
pip install virtualenv;
python -m virtualenv env;
source env/bin/activate;
```

### Install dependencies

```
pip install -r requirements.txt;
pip install -e .;
```

## Usage

### Clean

Used to clip a data set to a specified time frame as well as shifting data points.
The transformed data set is written to `out/cleaner/`.

```
a2e/data/clean.py --config=./data/800rpm.yaml --clip=True --shift="2020-08-27T10:00:00+00:00"
```

### Explore

Used to explore different aspects of a data set. 
The plots and stats are written to `out/explorer/`.

```
a2e/data/explore.py --config=./data/400rpm.yaml"
```

import pandas as pd
import numpy as np
import scipy.stats

DATA_PATH = "sample2.csv"  # Make sure MLflow step downloads or copies this to working dir

def test_column_names():
    data = pd.read_csv(DATA_PATH)
    expected_columns = [
        "id","name","host_id","host_name","neighbourhood_group","neighbourhood",
        "latitude","longitude","room_type","price","minimum_nights","number_of_reviews",
        "last_review","reviews_per_month","calculated_host_listings_count","availability_365"
    ]
    assert np.array_equal(expected_columns, data.columns.to_numpy())

def test_neighborhood_names():
    data = pd.read_csv(DATA_PATH)
    known_names = ["Bronx","Brooklyn","Manhattan","Queens","Staten Island"]
    assert set(known_names) == set(data['neighbourhood_group'].unique())

def test_proper_boundaries():
    data = pd.read_csv(DATA_PATH)
    idx = data['longitude'].between(-74.25, -73.50) & data['latitude'].between(40.5, 41.2)
    assert np.sum(~idx) == 0

def test_row_count():
    data = pd.read_csv(DATA_PATH)
    assert 15000 < data.shape[0] < 1000000

def test_price_range():
    data = pd.read_csv(DATA_PATH)
    min_price = 10
    max_price = 350
    assert data['price'].between(min_price, max_price).all()

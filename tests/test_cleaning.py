import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from cleaning import (
    parse_bhk,
    parse_total_sqft,
    clean_locality,
    remove_bhk_outliers,
    remove_bath_outliers
)

class TestBhkParser:
    def test_standard_bhk(self):
        assert parse_bhk("2 BHK") == 2
        assert parse_bhk("3 BHK") == 3

    def test_bedroom_string(self):
        assert parse_bhk("4 Bedroom") == 4
        assert parse_bhk("6 Bedroom") == 6

    def test_rk_string(self):
        assert parse_bhk("1 RK") == 1

    def test_null_or_invalid(self):
        assert parse_bhk(None) is None
        assert parse_bhk(np.nan) is None
        assert parse_bhk("InvalidString") is None

class TestSqftParser:
    def test_clean_number(self):
        assert parse_total_sqft("1056") == 1056.0
        assert parse_total_sqft(1200) == 1200.0

    def test_range_mean(self):
        # Range '2100 - 2850' should return average = 2475.0
        assert parse_total_sqft("2100 - 2850") == 2475.0
        assert parse_total_sqft("1000 - 1200") == 1100.0

    def test_unit_conversions(self):
        # 100 Sq. Yards * 9 = 900.0
        assert parse_total_sqft("100Sq. Yards") == 900.0
        # 10 Sq. Meter * 10.7639 = 107.64
        assert parse_total_sqft("10Sq. Meter") == 107.64
        # 1 Acre * 43560.0 = 43560.0
        assert parse_total_sqft("1Acres") == 43560.0

    def test_invalid_sqft(self):
        assert parse_total_sqft(None) is None
        assert parse_total_sqft("Unparseable") is None

class TestLocalityCleaner:
    def test_strip_whitespace(self):
        assert clean_locality("  Aundh  ") == "Aundh"
        assert clean_locality("Baner") == "Baner"

    def test_missing_locality(self):
        assert clean_locality(None) == "other"
        assert clean_locality(np.nan) == "other"
        assert clean_locality("") == "other"

class TestOutlierFiltering:
    def test_remove_bhk_density_outliers(self):
        # Property 1: 500 sqft, 1 bhk -> 500 sqft/bhk (valid)
        # Property 2: 500 sqft, 3 bhk -> 166.7 sqft/bhk (< 300, invalid)
        df = pd.DataFrame({
            'total_sqft': [500.0, 500.0],
            'bhk': [1, 3]
        })
        filtered = remove_bhk_outliers(df)
        assert len(filtered) == 1
        assert filtered.iloc[0]['bhk'] == 1

    def test_remove_bath_outliers(self):
        # Property 1: 2 bhk, 2 bath -> bath <= bhk + 2 (valid)
        # Property 2: 2 bhk, 5 bath -> 5 > 2 + 2 (invalid)
        df = pd.DataFrame({
            'bhk': [2, 2],
            'bath': [2.0, 5.0]
        })
        filtered = remove_bath_outliers(df)
        assert len(filtered) == 1
        assert filtered.iloc[0]['bath'] == 2.0

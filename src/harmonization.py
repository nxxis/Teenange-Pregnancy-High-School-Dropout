"""
Geographic and race/ethnicity harmonization for the teen birth / dropout project.

Per design_document.pdf section 9 ("Data Harmonization Strategy"):
  - state FIPS codes are the primary geographic key, not state names
  - a standardized race/ethnicity field is only created after documenting how
    each source defines race and Hispanic origin
  - an `original_category` field is retained so source values are recoverable
  - CDC "single race" categories are NOT assumed identical to NCES categories

This module resolves design_document.pdf section 19, decision #3 ("which
race/ethnicity categories are sufficiently comparable across CDC and
NCES/ACS"): NCES Table 219.85b explicitly excludes Hispanic ethnicity from
its race categories (its own footnote: "Race categories exclude persons of
Hispanic ethnicity"). CDC WONDER's "Mother's Single Race 6" pull does NOT
exclude Hispanic ethnicity (Hispanic origin was left "All Origins"), so each
WONDER race bucket mixes Hispanic and non-Hispanic mothers. This is a real,
irreducible mismatch at the individual-category level -- it is documented
here via `race_comparable_to_nces`, not silently mapped away.
"""

import pandas as pd

STATE_FIPS = {
    "Alabama": "01", "Alaska": "02", "Arizona": "04", "Arkansas": "05",
    "California": "06", "Colorado": "08", "Connecticut": "09", "Delaware": "10",
    "District of Columbia": "11", "Florida": "12", "Georgia": "13", "Hawaii": "15",
    "Idaho": "16", "Illinois": "17", "Indiana": "18", "Iowa": "19", "Kansas": "20",
    "Kentucky": "21", "Louisiana": "22", "Maine": "23", "Maryland": "24",
    "Massachusetts": "25", "Michigan": "26", "Minnesota": "27", "Mississippi": "28",
    "Missouri": "29", "Montana": "30", "Nebraska": "31", "Nevada": "32",
    "New Hampshire": "33", "New Jersey": "34", "New Mexico": "35", "New York": "36",
    "North Carolina": "37", "North Dakota": "38", "Ohio": "39", "Oklahoma": "40",
    "Oregon": "41", "Pennsylvania": "42", "Rhode Island": "44", "South Carolina": "45",
    "South Dakota": "46", "Tennessee": "47", "Texas": "48", "Utah": "49",
    "Vermont": "50", "Virginia": "51", "Washington": "53", "West Virginia": "54",
    "Wisconsin": "55", "Wyoming": "56",
}

US_STATES_PLUS_DC = set(STATE_FIPS.keys())

# USPS postal abbreviations, needed for choropleth map location codes
# (distinct from FIPS -- kept separate since they serve different purposes).
STATE_ABBREV = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
}


def add_state_abbrev(df: pd.DataFrame, state_col: str = "state") -> pd.DataFrame:
    df = df.copy()
    df["state_abbrev"] = df[state_col].map(STATE_ABBREV)
    return df

# Standard Census Bureau 4-region scheme, used for RQ4's geographic
# comparison in place of state fixed effects (51 state dummies would leave
# too few residual degrees of freedom at this sample size). DC is grouped
# with the South region, matching Census Bureau convention.
CENSUS_REGION = {
    "Connecticut": "Northeast", "Maine": "Northeast", "Massachusetts": "Northeast",
    "New Hampshire": "Northeast", "Rhode Island": "Northeast", "Vermont": "Northeast",
    "New Jersey": "Northeast", "New York": "Northeast", "Pennsylvania": "Northeast",

    "Illinois": "Midwest", "Indiana": "Midwest", "Michigan": "Midwest",
    "Ohio": "Midwest", "Wisconsin": "Midwest", "Iowa": "Midwest",
    "Kansas": "Midwest", "Minnesota": "Midwest", "Missouri": "Midwest",
    "Nebraska": "Midwest", "North Dakota": "Midwest", "South Dakota": "Midwest",

    "Delaware": "South", "District of Columbia": "South", "Florida": "South",
    "Georgia": "South", "Maryland": "South", "North Carolina": "South",
    "South Carolina": "South", "Virginia": "South", "West Virginia": "South",
    "Alabama": "South", "Kentucky": "South", "Mississippi": "South",
    "Tennessee": "South", "Arkansas": "South", "Louisiana": "South",
    "Oklahoma": "South", "Texas": "South",

    "Arizona": "West", "Colorado": "West", "Idaho": "West", "Montana": "West",
    "Nevada": "West", "New Mexico": "West", "Utah": "West", "Wyoming": "West",
    "Alaska": "West", "California": "West", "Hawaii": "West",
    "Oregon": "West", "Washington": "West",
}


# Approximate geographic centers (lat, lon in degrees), used only to build
# a k-nearest-neighbor spatial weights matrix for the Moran's I residual
# check in robustness.py -- precision to ~0.5 degree is ample for ranking
# neighbors. Not a join key.
STATE_CENTROID = {
    "Alabama": (32.8, -86.8), "Alaska": (64.7, -152.5), "Arizona": (34.3, -111.7),
    "Arkansas": (34.9, -92.4), "California": (37.2, -119.5), "Colorado": (39.0, -105.5),
    "Connecticut": (41.6, -72.7), "Delaware": (39.0, -75.5), "District of Columbia": (38.9, -77.0),
    "Florida": (28.6, -82.4), "Georgia": (32.7, -83.4), "Hawaii": (20.3, -156.4),
    "Idaho": (44.4, -114.6), "Illinois": (40.0, -89.2), "Indiana": (39.9, -86.3),
    "Iowa": (42.1, -93.5), "Kansas": (38.5, -98.4), "Kentucky": (37.5, -85.3),
    "Louisiana": (31.1, -92.0), "Maine": (45.4, -69.2), "Maryland": (39.0, -76.8),
    "Massachusetts": (42.3, -71.8), "Michigan": (44.3, -85.4), "Minnesota": (46.3, -94.3),
    "Mississippi": (32.7, -89.7), "Missouri": (38.4, -92.5), "Montana": (47.0, -109.6),
    "Nebraska": (41.5, -99.8), "Nevada": (39.3, -116.6), "New Hampshire": (43.7, -71.6),
    "New Jersey": (40.2, -74.7), "New Mexico": (34.4, -106.1), "New York": (42.9, -75.5),
    "North Carolina": (35.6, -79.4), "North Dakota": (47.5, -100.5), "Ohio": (40.3, -82.8),
    "Oklahoma": (35.6, -97.5), "Oregon": (43.9, -120.6), "Pennsylvania": (40.9, -77.8),
    "Rhode Island": (41.7, -71.5), "South Carolina": (33.9, -80.9), "South Dakota": (44.4, -100.2),
    "Tennessee": (35.9, -86.4), "Texas": (31.5, -99.3), "Utah": (39.3, -111.7),
    "Vermont": (44.1, -72.7), "Virginia": (37.5, -78.9), "Washington": (47.4, -120.5),
    "West Virginia": (38.6, -80.6), "Wisconsin": (44.6, -89.9), "Wyoming": (43.0, -107.6),
}


def add_census_region(df: pd.DataFrame, state_col: str = "state") -> pd.DataFrame:
    df = df.copy()
    df["census_region"] = df[state_col].map(CENSUS_REGION)
    return df


def add_fips(df: pd.DataFrame, state_col: str = "state") -> pd.DataFrame:
    """Add a `fips` column from a state-name column. Rows for states not in the
    50-states-plus-DC list (e.g. Puerto Rico, 'United States') get NaN and
    should be dropped explicitly by the caller, not silently kept."""
    df = df.copy()
    df["fips"] = df[state_col].map(STATE_FIPS)
    return df


# --- Race/ethnicity crosswalk -------------------------------------------

# Canonical categories used across the merged dataset. NCES's own 8-category
# scheme is used as the canonical set because it is the more granular of the
# two (it separates Hispanic ethnicity); WONDER categories are mapped onto it
# with the mismatch flagged, not hidden.
CANONICAL_RACE_ETHNICITY = [
    "Total", "White", "Black", "Hispanic", "Asian",
    "Pacific Islander", "American Indian/Alaska Native", "Two or more races",
]

# NCES Table 219.85b columns -> canonical label.
# NCES footnote: "Race categories exclude persons of Hispanic ethnicity."
# So every category here except "Hispanic" itself and "Total" is non-Hispanic.
NCES_RACE_CROSSWALK = {
    "Total": {
        "canonical": "Total",
        "race_definition": "all races/ethnicities combined (per NCES footnote, includes small groups not shown separately)",
        "race_comparable_to_nces": True,
    },
    "White": {
        "canonical": "White",
        "race_definition": "White, non-Hispanic",
        "race_comparable_to_nces": True,
    },
    "Black": {
        "canonical": "Black",
        "race_definition": "Black, non-Hispanic",
        "race_comparable_to_nces": True,
    },
    "Hispanic": {
        "canonical": "Hispanic",
        "race_definition": "Hispanic, any race",
        "race_comparable_to_nces": True,
    },
    "Asian": {
        "canonical": "Asian",
        "race_definition": "Asian, non-Hispanic",
        "race_comparable_to_nces": True,
    },
    "Pacific Islander": {
        "canonical": "Pacific Islander",
        "race_definition": "Native Hawaiian/Pacific Islander, non-Hispanic",
        "race_comparable_to_nces": True,
    },
    "American Indian/Alaska Native": {
        "canonical": "American Indian/Alaska Native",
        "race_definition": "American Indian/Alaska Native, non-Hispanic",
        "race_comparable_to_nces": True,
    },
    "Two or more races": {
        "canonical": "Two or more races",
        "race_definition": "Two or more races, non-Hispanic",
        "race_comparable_to_nces": True,
    },
}

# CDC WONDER "Mother's Single Race 6" categories -> canonical label.
# Mother's Hispanic Origin was left "All Origins" in the WONDER query
# (design_document.pdf section 7.1, point 3), so NONE of these categories
# exclude Hispanic ethnicity -- they mix Hispanic and non-Hispanic mothers.
# `race_comparable_to_nces` is False everywhere. "Total" is not available
# from WONDER at all (suppression constraints), so it is omitted here.
WONDER_RACE_CROSSWALK = {
    "White": {
        "canonical": "White",
        "race_definition": "White, any ethnicity (Hispanic origin not separated)",
        "race_comparable_to_nces": False,
    },
    "Black or African American": {
        "canonical": "Black",
        "race_definition": "Black, any ethnicity (Hispanic origin not separated)",
        "race_comparable_to_nces": False,
    },
    "Asian": {
        "canonical": "Asian",
        "race_definition": "Asian, any ethnicity (Hispanic origin not separated)",
        "race_comparable_to_nces": False,
    },
    "Native Hawaiian or Other Pacific Islander": {
        "canonical": "Pacific Islander",
        "race_definition": "NHOPI, any ethnicity (Hispanic origin not separated)",
        "race_comparable_to_nces": False,
    },
    "American Indian or Alaska Native": {
        "canonical": "American Indian/Alaska Native",
        "race_definition": "AI/AN, any ethnicity (Hispanic origin not separated)",
        "race_comparable_to_nces": False,
    },
    "More than one race": {
        "canonical": "Two or more races",
        "race_definition": "Two or more races, any ethnicity (Hispanic origin not separated)",
        "race_comparable_to_nces": False,
    },
}

# WONDER rows that are always 0 births / "Not Available" population for every
# state and year -- structural placeholders in WONDER's race scheme, not real
# suppressed data. Drop these before harmonizing.
WONDER_PLACEHOLDER_CATEGORIES = {"Unknown or Not Stated", "Not Available", "Not Reported"}


def harmonize_nces_race(df: pd.DataFrame, race_col: str = "race_ethnicity") -> pd.DataFrame:
    """Attach canonical race/ethnicity + comparability metadata for NCES rows,
    retaining the original category in `original_category`."""
    df = df.copy()
    df["original_category"] = df[race_col]
    df["race_ethnicity"] = df[race_col].map(lambda r: NCES_RACE_CROSSWALK[r]["canonical"])
    df["race_definition"] = df["original_category"].map(lambda r: NCES_RACE_CROSSWALK[r]["race_definition"])
    df["race_comparable_to_nces"] = df["original_category"].map(lambda r: NCES_RACE_CROSSWALK[r]["race_comparable_to_nces"])
    return df


def harmonize_wonder_race(df: pd.DataFrame, race_col: str = "race_ethnicity") -> pd.DataFrame:
    """Attach canonical race/ethnicity + comparability metadata for WONDER
    rows, retaining the original category in `original_category`. Rows in
    WONDER_PLACEHOLDER_CATEGORIES must already be dropped by the caller."""
    df = df.copy()
    df["original_category"] = df[race_col]
    df["race_ethnicity"] = df[race_col].map(lambda r: WONDER_RACE_CROSSWALK[r]["canonical"])
    df["race_definition"] = df["original_category"].map(lambda r: WONDER_RACE_CROSSWALK[r]["race_definition"])
    df["race_comparable_to_nces"] = df["original_category"].map(lambda r: WONDER_RACE_CROSSWALK[r]["race_comparable_to_nces"])
    return df


def check_duplicates(df: pd.DataFrame, keys: list[str], label: str) -> None:
    """Design doc section 9, point 8: check duplicates after every merge,
    keyed on state + year/period + demographic group. Raises rather than
    warns -- a duplicate key silently corrupts every downstream rate."""
    dupes = df[df.duplicated(subset=keys, keep=False)]
    if len(dupes):
        raise ValueError(
            f"{label}: {len(dupes)} duplicate rows found on keys {keys}.\n"
            f"{dupes[keys].drop_duplicates().to_string()}"
        )

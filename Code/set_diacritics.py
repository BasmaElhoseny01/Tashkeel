import pandas as pd


def get_diacritics_set() -> set[str]:
    """
    This function gets the set of used diacritics
    """
    filename = 'pickle/diacritics.pickle'
    dataframe = pd.read_pickle(filename)
    diacritics = set(dataframe)
    return diacritics

def get_id_diacritics_dict() -> dict[str, int]:
    """
    This function gets the dictionary that maps each diacritic to its id
    """
    filename = 'pickle/diacritic2id.pickle'
    dataframe = pd.read_pickle(filename)
    id_diacritic = dict(dataframe)
    return id_diacritic
import os

def get_tdata_corr_txt_dir():
    """Return the directory with text correlation sample files"""

    return os.path.join(os.path.dirname(__file__),'tdata_corr_txt')

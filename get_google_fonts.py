from pathlib import Path
import pandas as pd
from string import ascii_letters
from git import Repo, RemoteProgress
from tqdm import tqdm
import __fonts__

class CloneProgress(RemoteProgress):
    # from Cosmos Zhu: https://stackoverflow.com/a/65576165
    def __init__(self):
        super().__init__()
        self.pbar = tqdm()

    def update(self, op_code, cur_count, max_count=None, message=''):
        self.pbar.total = max_count
        self.pbar.n = cur_count
        self.pbar.refresh()

def download_google_fonts():
    print('Cloning google fonts repository')
    repo = Repo.clone_from('https://github.com/google/fonts', to_path='google-fonts', no_checkout=True, progress=CloneProgress())
    print('Checking out commit')
    repo.git.checkout('ce84a48f1dc57b5dd6b4d46b3ac5204fa48d0b99')  # 29/10/2025 (matches google-fonts-analytics-archive)
    return None

def download_google_fonts_analytics():
    print('Cloning google fonts analytics archive repository')
    repo = Repo.clone_from('https://github.com/radames/google-fonts-analytics-archive/', to_path='google-fonts-analytics-archive', no_checkout=True, progress=CloneProgress())
    print('Checking out commit')
    repo.git.checkout('feb507c623e23441736af18d8ca818f78f757cfa')  # 29/10/2025 (matches google-fonts)
    return None

def get_fonts_df():
    test_letters = [*ascii_letters, 'ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü', 'ß']
    test_font_size = 50
    bad_fonts = pd.read_csv(Path('data') / 'bad_fonts.csv')
    bad_fonts = bad_fonts.loc[bad_fonts.reason!='outline', :]  # keep the outlined fonts

    # get list of fonts that work for the test letters
    fonts_df = __fonts__.get_google_font_df(char_list=test_letters, font_size=test_font_size, location='', exclude_ttfs=bad_fonts.ttf.tolist(), max_identical=3)

    return fonts_df

def main():
    if not Path('google-fonts').exists():
        download_google_fonts()
    
    if not Path('google-fonts-analytics-archive').exists():
        download_google_fonts_analytics()
    
    freqs_path = Path('freqs')
    freqs_path.mkdir(exist_ok=True)
    out_path = freqs_path / 'font_frequencies.csv'
    fonts_df = get_fonts_df()
    fonts_df.to_csv(out_path)
    print(f'Saved font frequencies to {out_path}')
    return None

if __name__ == "__main__":
    main()

from pathlib import Path
import numpy as np
import pandas as pd
from string import ascii_letters
from tqdm import tqdm

letters = [*ascii_letters, 'ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü', 'ß']

def get_freqs(letters=letters):
    # calculate letter frequencies from SUBTLEX-DE word frequencies
    wf_df = pd.read_csv(Path('data') / 'SUBTLEX-DE_cleaned_with_Google00_frequencies.csv', encoding='utf-8')

    # flat array of all letters, collapsed across words
    subtlex_chars = np.array([L for w in wf_df.Word for L in w])

    # get the corresponding counts of word frequency, from the words the letters belong to
    nchars = wf_df.Word.str.len()
    subtlex_char_wts = np.repeat(wf_df.WFfreqcount.to_numpy(), repeats=nchars)
    
    # now calculate total letter counts as the total number of times the letter has appeared, weighted by the frequency of the words they appeared in
    subtlex_char_counts = np.array([subtlex_char_wts[subtlex_chars==L].sum() for L in letters])

    out_df = pd.DataFrame({'letter': letters,
                           'n': subtlex_char_counts,
                           'p': subtlex_char_counts/subtlex_char_counts.sum()})

    return out_df

def main():
    freqs_path = Path('freqs')
    freqs_path.mkdir(exist_ok=True)
    out_path = freqs_path / 'letter_frequencies.csv'
    lf_df = get_freqs()
    lf_df.to_csv(out_path)
    print(f'Saved letter frequencies to {out_path}')
    return None

if __name__ == "__main__":
    main()

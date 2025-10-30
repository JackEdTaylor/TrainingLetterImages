import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from string import ascii_letters
from pathlib import Path
import shutil
from tqdm import tqdm
import warnings

def get_letter_dims(letter, font_file='arial.ttf', font_size=128, rotation=0.0):
    # takes the bbox, and then calculates what the height and width will be after rotation
    font = ImageFont.truetype(font_file, font_size)
    _, _, w, h = font.getbbox(letter, anchor='lt')
    W = w * np.abs(np.sin(rotation)) + h * np.abs(np.cos(rotation))
    H = w * np.abs(np.cos(rotation)) + h * np.abs(np.sin(rotation))
    return W, H

def render_text_im(letter, font_file='arial.ttf', x=128, y=128, font_size=128, rotation=0.0, canvas_dims=(256, 256), variation='Regular'):
    font = ImageFont.truetype(font_file, font_size)

    # if a variation is requested, try to set to the requested variation
    if variation is not None:
        try:
            font.set_variation_by_name(variation)
        except OSError:
            warnings.warn('Got OSError setting variation - is it really a variation font?')
            pass  # e.g., will throw an OSError if not a variation font
        except ValueError:
            warnings.warn(f'Got ValueError setting variation to "{variation}" - does the font support this variation?')
            pass  # e.g., will throw a ValueError if the requested variation is missing

    im   = Image.new('L', canvas_dims, color=0)
    draw = ImageDraw.Draw(im)
    draw.text((x, y), letter, fill=255, font=font, anchor='mm')
    im   = im.rotate(rotation, center=(x, y))  # rotate using the text location as the centre of rotation
    return im

def get_google_font_list():
    fonts_df = pd.read_csv(Path('freqs') / 'font_frequencies.csv')
    return fonts_df.ttf_path.tolist()

def main():
    np.random.seed(25102025)

    # settings
    n_samples = 8  # per combination of font and letter
    fonts = get_google_font_list()
    letters = [*ascii_letters, 'ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü', 'ß']
    print(f'Will generate {n_samples * len(fonts) * len(letters)} images in total.')

    canvas_dims = (256, 256)
    rotation_bounds = (-15, 15)
    size_bounds = (12, 0.5*max(canvas_dims))
    decimals = 3  # number of decimals to round all variables to
    ims_path = Path('ims')

    if ims_path.exists():
        print('Removing existing ims directory...')
        shutil.rmtree(ims_path)
    
    # generate all images
    for L in (pbar:=tqdm(letters, desc='Generating images')):
        pbar.set_postfix_str(L)
        case_lab = 'lwr' if L.islower() else 'upr'
        save_dir = ims_path / Path(f'{L}_{case_lab}')
        save_dir.mkdir(parents=True, exist_ok=True)

        for F in fonts:
            rotation_vals = np.random.uniform(low=rotation_bounds[0], high=rotation_bounds[1], size=n_samples).round(decimals)
            size_vals = np.random.uniform(low=size_bounds[0], high=size_bounds[1], size=n_samples).round(decimals)

            # get the letter dimensions after rotation (used to ensure that the letter stays on the canvas)
            letter_dims = np.array([get_letter_dims(letter=L, font_file=F, font_size=S, rotation=R)
                                    for S, R in zip(size_vals, rotation_vals)])

            if np.any(letter_dims[:, 0]>canvas_dims[0]) or np.any(letter_dims[:, 1]>canvas_dims[1]):
                warnings.warn(f'Letter {L} in font {Path(F).stem} exceeds canvas dimensions!')

            # set the bounds for x and y location so that the letters don't exceed the canvas
            # (depends on size and rotation values)
            x_trans_bounds = np.array([0 + letter_dims[:, 0]/2, canvas_dims[0] - letter_dims[:, 0]/2])
            y_trans_bounds = np.array([0 + letter_dims[:, 1]/2, canvas_dims[1] - letter_dims[:, 1]/2])

            x_vals = np.round( np.random.uniform(low=x_trans_bounds[0, :], high=x_trans_bounds[1, :], size=n_samples), decimals )
            y_vals = np.round( np.random.uniform(low=y_trans_bounds[0, :], high=y_trans_bounds[1, :], size=n_samples), decimals )

            # generate images for this letter and font
            ims = [render_text_im(letter=L, font_file=F, x=X, y=Y, font_size=S, rotation=R, canvas_dims=canvas_dims)
                   for X, Y, S, R in zip(x_vals, y_vals, size_vals, rotation_vals)]
            
            # save to file
            file_names = [f'font-{Path(F).stem.replace(".", "-")}_x{X}_y{Y}_sz{S}_rot{R}'.replace('.', 'p')
                          for X, Y, S, R in zip(x_vals, y_vals, size_vals, rotation_vals)]

            for im, fn in zip(ims, file_names):
                im.save(save_dir / f'{fn}.png')

    return None

if __name__ == "__main__":
    main()

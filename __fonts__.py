import __fonts_public_pb2__
import generate_images

import numpy as np
import os.path as op
import glob
import re
import pandas as pd
from tqdm import tqdm
import json
from google.protobuf import text_format

# function to import the metadata file
def get_pb_metadata(pb_path):
    with open(pb_path, 'r', encoding='utf-8') as f:
        f_dat = f.read()
    
    pb_metadata = __fonts_public_pb2__.FamilyProto()
    text_format.Merge(f_dat, pb_metadata, allow_unknown_field=True)

    return(pb_metadata)

# function to extract from the metadata file which .ttf, if any, is regular latin
def get_regular_ttf(pb_metadata):
    # check supports latin text; return None if not
    subset_dat = pb_metadata.subsets
    if 'latin' not in subset_dat:
        return(None)

    # get which ttf is a regular style
    font_dat = pb_metadata.fonts
    is_regular_style = [(fd.style=='normal') & (fd.weight==400) for fd in font_dat]  # check style and weight (standard weight is 400, i.e., not bold)
    all_ttfs = [fd.filename for fd in font_dat]
    regular_ttfs = [x for (x, is_reg) in zip(all_ttfs, is_regular_style) if is_reg]

    # if there are multiple possible ttf files, and only one has the '-Regular' suffix, use the Regular one
    regular_suffix_regex = re.compile(r'.*-Regular\.')
    regular_suffix_ttfs = list(filter(regular_suffix_regex.match, regular_ttfs))
    if len(regular_ttfs) > 1 and len(regular_suffix_ttfs) == 1:
        regular_ttfs = regular_suffix_ttfs

    # return None if no regular fonts
    if len(regular_ttfs) == 0:
        return(None)
    
    # error if >1 acceptable font file
    if len(regular_ttfs) > 1:
        raise ValueError('Expected 1 acceptable font file in {}, but there were {}'.format(pb_metadata.name, len(regular_ttfs)))
    
    return(regular_ttfs[0])

# function to test fonts and identify any that fail to produce any of the characters, or produce more than max_identical identical arrays
def font_succeeds(char_list, font='arial.ttf', font_size=50, canvas_dims=(100, 100), max_dims=(np.inf, np.inf), max_identical=3):
    # check for variable font from file name, and request the regular variation if it is one
    # (commented out as we just try it anyway, and get an exception for any error)
    # if bool(re.match(r'^.*\[.*\]\.ttf$', font)):
    #     variation = 'Regular'
    # else:
    #     variation = None

    try:
        x = [generate_images.render_text_im(char, font_file=font, font_size=font_size, canvas_dims=canvas_dims, variation='Regular', rotation=0.0, x=canvas_dims[0]/2, y=canvas_dims[1]/2) for char in char_list]

        x_arrs = [np.array(x_i) for x_i in x]

        # N unique arrays
        x_3d = np.array(x_arrs)
        _, unique_counts = np.unique(x_3d, axis=0, return_counts=True)
        if np.max(unique_counts) > max_identical:
            return False

        # does it produce real values?
        if not all([np.all(np.isreal(x_i)) for x_i in x_arrs]):
            return False
        
        # do all arrays sum to >0?
        if not np.all(np.any(x_3d>0, axis=(1,2))):
            return False
        
        return True
    except:
        # if it fails to draw or be manipulated as a numpy array
        # (includes exceeding canvas dimensions in render_text_im)
        return False

# function to get a dataframe with all font info for all suitable fonts
def get_google_font_df(char_list, font_size, max_dims=(np.inf, np.inf), max_identical=3, max_canvas_size_factor=3, location='', exclude_ttfs=[]):
    # get list of all ttfs and infer from that the directories that contain fonts
    ttf_files = glob.glob(op.join(location, 'google-fonts', '*', '*', '*.ttf'))
    font_df = pd.DataFrame({'font_dir': list(set([op.dirname(x) for x in ttf_files]))})

    # get path of .pb metadata, and remove from dataframe if none is present
    font_df.loc[:, 'n_metadatas'] = font_df.apply(lambda r: len(glob.glob(op.join(r.font_dir, '*.pb'))), axis=1)  # count number of metadata files
    font_df = font_df.loc[font_df['n_metadatas']==1, :]  # remove entries with 0 or >1 metadata files
    font_df.loc[:, 'metadata_path'] = font_df.apply(lambda r: glob.glob(op.join(r.font_dir, '*.pb'))[0], axis=1)  # take first entry

    # get the metadata
    metadata = [get_pb_metadata(r.metadata_path) for _, r in font_df.iterrows()]

    # get the family name
    font_df.loc[:, 'family'] = [m.name for m in metadata]

    # get the category
    font_df.loc[:, 'category'] = [m.category for m in metadata]

    # get the regular ttf file
    font_df.loc[:, 'ttf'] = [get_regular_ttf(m) for m in metadata]

    # remove entries with no regular ttf file
    font_df = font_df.loc[font_df['ttf'].notnull()]

    # get the full path of the ttf file
    font_df.loc[:, 'ttf_path'] = font_df.apply(lambda r: op.join(r.font_dir, r.ttf), axis=1)

    # any font paths containing these strings will be excluded before being tested
    # NOTE: we keep variable fonts, assuming that they are handled by the request to draw the "Regular" variant
    exclude_fonts_containing = [
        'librebarcode',  # a special font for writing text to bar codes
        'jsMath-cmsy10', 'jsMath-cmex10',  # maps letters to maths symbols
        'redacted',  # replaces text with blocked-out symbols
        'FlowCircular', 'FlowRounded', 'FlowBlock',  # replaces text with blocks that have circular edges
        'ZillaSlabHighlight',  # inverts 0s and 1s
        'Ponnala']  # problem rendering this font

    keep_font = [not np.any([excl_i in font_path_j for excl_i in exclude_fonts_containing]) and ttf_j not in exclude_ttfs for font_path_j, ttf_j in zip(font_df.ttf_path, font_df.ttf)]

    font_df = font_df.loc[keep_font, :]
    
    # test fonts and remove any that fail to produce any of the characters, or produce identical characters in any cases
    canvas_dims = [int(round(font_size*max_canvas_size_factor))] * 2
    tqdm.pandas(desc='Testing fonts on characters')
    font_df['font_okay'] = font_df.progress_apply(lambda r: font_succeeds(char_list=char_list, font_size=font_size, canvas_dims=canvas_dims, font=r.ttf_path, max_dims=max_dims, max_identical=max_identical), axis=1)

    font_df = font_df.loc[font_df['font_okay'], :]

    # join to the usage metadata
    with open(op.join(location, 'google-fonts-analytics-archive', 'stats.json'), encoding='utf-8') as f:
        font_stats = pd.DataFrame(json.load(f))

    # join the dataframes ("family" should be the only shared column name)
    # use an inner join to remove any fonts that we don't have both stats and font files for
    merged_font_df = pd.merge(left=font_df, right=font_stats, on='family', how='inner')

    # sort rows by family
    merged_font_df = merged_font_df.sort_values('family').reset_index()

    # remove old index column from before merge
    merged_font_df = merged_font_df.drop(columns=['index'])

    # remove any columns that are no longer useful
    merged_font_df = merged_font_df.drop(columns=['n_metadatas', 'font_okay'])

    return merged_font_df

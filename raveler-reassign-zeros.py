#!/usr/bin/env python

import os
import shutil
import numpy as np
import skimage as ski
import skimage.io
import skimage.util
import skimage.measure

indir = '/groups/flyem/data/temp/ordishc/exports/raveler_export_for davi'
outdir = '/groups/saalfeld/home/nuneziglesiasj/data/raveler_export_davi_v7'
os.makedirs(os.path.join(outdir, 'superpixel_maps'), exist_ok=True)

sp_fns = sorted(os.listdir(os.path.join(indir, 'superpixel_maps')))
sp_to_seg_file = 'superpixel_to_segment_map.txt'
seg_to_body_file = 'segment_to_body_map.txt'


# read images using generators
sections = (ski.io.imread(os.path.join(indir, 'superpixel_maps', f))
            for f in sp_fns)

# find max body id
seg2bod = np.loadtxt(os.path.join(indir, seg_to_body_file),
                     dtype=int, delimiter='\t')
max_body_id = np.max(seg2bod[:, 1])
start_body = max_body_id + 1

sp2seg = np.loadtxt(os.path.join(indir, sp_to_seg_file),
                    dtype=int, delimiter='\t')
max_segment_id = np.max(sp2seg[:, 2])
start_segment = max_segment_id + 1
# section boundaries in superpixel-to-segment map table
sec_idxs = np.unique(sp2seg[:, 0], return_index=True)[1]
sec_idxs = np.concatenate((sec_idxs, [len(sp2seg)]))

# we will append to these, concatenate, sort, rewrite.
sp2seg_new = [sp2seg]
seg2bod_new = [seg2bod]

for i, (filename, superpixels) in enumerate(zip(sp_fns, sections)):

    print('Processing section %i...' % i)

    # find max superpixel id and create background map
    superpixel_map = np.sum(superpixels * [1, 1<<8, 1<<16, 0], axis=-1)
    max_superpixel_id = np.max(superpixel_map)
    bg_superpixels = (superpixel_map == 0)

    # find all connected components of background
    #   * replace 0 with 1, everything else with 0
    #   * find connected components -> label image
    #   * add max superpixel id to label image, except where 0

    labels = ski.measure.label(bg_superpixels, background=0)
    num_components = np.max(labels)

    labels[labels == -1] = 0
    labels[labels > 0] += max_superpixel_id

    replace = labels > 0
    labels = labels[replace]
    superpixels[replace, 0] = labels % (1<<8)
    superpixels[replace, 1] = (labels % (1<<16)) // (1<<8)
    superpixels[replace, 2] = labels // (1<<16)

    ski.io.imsave(os.path.join(outdir, 'superpixel_maps', filename),
                  superpixels)

    # make maps by stacking arrays -- these will be written out with savetxt
    unique_sps = np.unique(labels)
    unique_seg = np.arange(start_segment, start_segment + num_components,
                           dtype=int)
    unique_bod = np.arange(start_body, start_body + num_components,
                           dtype=int)
    section = np.zeros_like(unique_sps)
    section.fill(i)

    # add to the sp2seg and seg2bod maps
    sp2seg_new.append(np.array([section, unique_sps, unique_seg]).T)
    seg2bod_new.append(np.array([unique_seg, unique_bod]).T)

    # update the current starting segment and body
    start_segment += num_components
    start_body += num_components

sp2seg = np.concatenate(sp2seg_new, axis=0)
# sort by section
sp2seg = sp2seg[np.argsort(sp2seg[:, 0]), :]
seg2bod = np.concatenate(seg2bod, axis=0)

np.savetxt(os.path.join(outdir, sp_to_seg_file), sp2seg,
           fmt='%i', delimiter='\t')
np.savetxt(os.path.join(outdir, seg_to_body_file), seg2bod,
           fmt='%i', delimiter='\t')

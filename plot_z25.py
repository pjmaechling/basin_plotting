#!/usr/bin/env python3

import argparse
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot Z2.5 depth data for California CVMs.')

    parser.add_argument('--data_file', required=True, help='Space-delimited file with Z2.5 data (depth in column 4)')
    parser.add_argument('--meta_file', required=True, help='JSON metadata file (lat_list, lon_list, nx, ny, max_depth)')
    parser.add_argument('--output_file', required=True, help='Output PNG filename')
    parser.add_argument('--cmap', default='viridis', help='Matplotlib colormap name')
    parser.add_argument('--alpha', type=float, default=1.0, help='Transparency (0=transparent,1=opaque)')
    parser.add_argument('--scale_mode', choices=['metadata','datamax','user'], default='metadata',
                        help='Color scale mode: metadata max, data max, or user-specified')
    parser.add_argument('--user_max', type=float, default=None, help='User-defined max value (if scale_mode=user)')
    parser.add_argument('--title', default='Z2.5 depth data for California CVMs.',
                        help='Plot title (use quotes for multi-word titles)')

    return parser.parse_args()



def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Parse command-line arguments
    args = parse_arguments()

    # Load metadata
    with open(args.meta_file) as f:
        meta = json.load(f)

    lat_list = np.array(meta['lat_list'], dtype=float)
    lon_list = np.array(meta['lon_list'], dtype=float)
    nx = int(meta['nx'])
    ny = int(meta['ny'])

    if 'max depth' in meta:
        meta_max = float(meta['max depth'])
    else:
        logger.warning("'max_depth' not found in metadata. Using datamax mode instead.")
        args.scale_mode = 'datamax'
        meta_max = None

    logger.info(f"Metadata: nx={nx}, ny={ny}, lat-range=({lat_list.min()}, {lat_list.max()})")

    # Read data file
    data = np.loadtxt(args.data_file)

    # New depth extraction logic
    depths_flat = np.where(
        data[:,2] == 1,
        data[:,3],  # use column 4 when column 3 == 1
        data[:,4]   # use column 5 when column 3 > 1
    )

    if depths_flat.size != nx * ny:
        logger.error(f"Data size mismatch: expected {nx*ny} points, got {depths_flat.size}")
        return

    depths = depths_flat.reshape((ny, nx))
    depths = np.ma.masked_equal(depths, -1.0)

    # Determine if we need to flip vertically:
    lat_ascending = (lat_list[0] < lat_list[-1])
    if not lat_ascending:
        depths = np.flipud(depths)
        logger.info("Flipped data vertically to match Mercator projection orientation.")

    # Determine color scale vmax
    if args.scale_mode == 'metadata':
        if meta_max is None:
            logger.error("Metadata mode selected but no 'max_depth' found. Exiting.")
            return
        vmax = meta_max
    elif args.scale_mode == 'datamax':
        if depths.mask.all():
            logger.warning("All data are missing; setting max to 0")
            vmax = 0.0
        else:
            vmax = float(depths.max())
    elif args.scale_mode == 'user':
        if args.user_max is None:
            logger.error("User max scale mode requires --user_max value")
            return
        vmax = args.user_max
    else:
        logger.error("Invalid scale_mode. Exiting.")
        return

    vmin = 0
    logger.info(f"Using color scale vmin={vmin}, vmax={vmax}")

    # Set up Mercator map
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.Mercator())

    # Calculate extent in geographic coords
    lon_min, lon_max = lon_list.min(), lon_list.max()
    lat_min, lat_max = lat_list.min(), lat_list.max()
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    logger.info("Set map extent.")

    # Add base map features
    ax.add_feature(cfeature.LAND, facecolor='none')
    ax.add_feature(cfeature.OCEAN, facecolor='none')
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='black')
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
    states = cfeature.NaturalEarthFeature('cultural',
        'admin_1_states_provinces_lines', '10m', facecolor='none')
    ax.add_feature(states, edgecolor='gray', linewidth=0.5)

    # Add major roads
    try:
        roads = cfeature.NaturalEarthFeature(category='cultural',
                                             name='roads',
                                             scale='10m',
                                             facecolor='none')
        ax.add_feature(roads, edgecolor='brown', linewidth=0.5)
        logger.info("Added major roads (brown lines).")
    except Exception as e:
        logger.warning(f"Could not load major roads: {e}")

    # Plot the depth data using imshow
    cmap = plt.get_cmap(args.cmap)
    im = ax.imshow(depths, origin='lower',
                   extent=[lon_min, lon_max, lat_min, lat_max],
                   interpolation='bilinear',
                   cmap=cmap, alpha=args.alpha,
                   vmin=vmin, vmax=vmax,
                   transform=ccrs.PlateCarree())
    logger.info("Rendered depth data with imshow on Mercator projection.")

    # Add major cities (pop > 500k) from Natural Earth
    shpfile = shpreader.natural_earth(resolution='10m', category='cultural', name='populated_places')
    reader = shpreader.Reader(shpfile)
    cities_plotted = 0
    for rec in reader.records():
        name = rec.attributes['NAME']
        if name == "Oakland":
            continue # Skip this city because it overlaps with other city labels
        pop = rec.attributes.get('POP_MAX')
        if pop and pop >= 500000:
            lon_pt, lat_pt = rec.geometry.x, rec.geometry.y
            if lon_min <= lon_pt <= lon_max and lat_min <= lat_pt <= lat_max:
                ax.plot(lon_pt, lat_pt, 'ro', markersize=4, transform=ccrs.PlateCarree())
                ax.text(lon_pt + 0.1, lat_pt + 0.1, rec.attributes['NAME'],
                        fontsize=8, transform=ccrs.PlateCarree())
                cities_plotted += 1

    logger.info(f"Plotted {cities_plotted} major cities on the map.")

    # Add colorbar and title
    plt.colorbar(im, ax=ax, label='Depth (m)')
    plt.title(args.title)

    plt.tight_layout()

    # Save figure
    try:
        plt.savefig(args.output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Map saved to {args.output_file}")
    except Exception as e:
        logger.error(f"Error saving figure: {e}")

if __name__ == '__main__':
    main()

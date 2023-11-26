import copy

from nonpoisson import paths
import scipy
from scipy.interpolate import griddata
from scipy.stats import entropy
import scipy.interpolate as sci
from matplotlib.lines import Line2D
from scipy.ndimage import binary_closing
from entropymdlp.mdlp import MDLP
import skimage
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
import functools
from sklearn.preprocessing import KBinsDiscretizer
from skimage.measure import label
from skimage.filters.rank import median
from skimage.morphology import disk, remove_small_objects, remove_small_holes
import seaborn as sns
from openquake.hazardlib.geo import Polygon as oqpoly
from openquake.hazardlib.geo.utils import OrthographicProjection
from shapely.geometry import Polygon as shpoly
from shapely.geometry import Point as shpoint
from shapely.geometry.multipolygon import MultiPolygon
import fiona
import rasterio
import rasterio.features
import rasterio.merge
import rasterio.crs
import pandas

sns.set_style("darkgrid", {"axes.facecolor": ".9", 'font.family': 'Ubuntu'})


def extrapolate_nans(x, y, v, method='nearest'):
    """
    Extrapolate the NaNs or masked values in a grid INPLACE using nearest
    value.

    Parameters:

    * x, y : 1D arrays
        Arrays with the x and y coordinates of the data points.
    * v : 1D array
        Array with the scalar value assigned to the data points.

    Returns:

    * v : 1D array
        The array with NaNs or masked values extrapolated.
    """

    if np.ma.is_masked(v):
        nans = v.mask
    else:
        nans = np.isnan(v)
    notnans = np.logical_not(nans)
    v[nans] = griddata((x[notnans], y[notnans]), v[notnans],
                       (x[nans], y[nans]),
                       method=method).ravel()
    return v


class GNSSImporter:

    @staticmethod
    def hw_final():
        Data = np.genfromtxt(paths.gps_hw_final, delimiter=',', skip_header=1)
        points = np.genfromtxt(paths.points, delimiter=',', skip_header=1)
        interp = scipy.interpolate.griddata(Data[:, :2], Data[:, 4:], points,
                                            method='nearest')
        interp = np.vstack(
            [extrapolate_nans(points[:, 0], points[:, 1], interp[:, 0]),
             extrapolate_nans(points[:, 0], points[:, 1], interp[:, 1]),
             extrapolate_nans(points[:, 0], points[:, 1], interp[:, 2])]).T
        haines_final = np.hstack((points, interp[:, [0, 2, 1]]))
        np.savetxt(paths.gps_proc_models['hw_final'], haines_final,
                   delimiter=',', header='lon,lat,exx,eyy,exy', comments='',
                   fmt='%.8e')

        return haines_final


class GeodeticModel(object):

    def __init__(self, model_name, suffix=''):

        self.name = model_name
        self.paths = paths.replace_path(
            {results_type: paths.get('spatial', results_type, model_name)
             for results_type in paths.subfolders if results_type in
             paths.results_path['spatial'].keys()}, suffix)
        self.raster = None
        self.lats = []
        self.lons = []
        self.data = {}
        self.bin_edges = {}
        self.polygons = {}

    def import_data(self, process_raw=True, res_0=0.02):

        from nonpoisson import geo
        if process_raw:
            getattr(GNSSImporter, self.name)()
        with open(paths.gps_proc_models[self.name], 'r') as file_:
            attributes = file_.readline().strip().split(',')[2:]
        geo.rasterize_results(paths.gps_proc_models[self.name],
                              self.paths['raster'],
                              self.paths['shp'],
                              attributes=attributes,
                              res=(res_0, res_0),
                              del_shp=True)
        self.raster = geo.read_raster(self.paths['raster'],
                                      indexes=range(len(attributes)))
        self.raster['attributes'] = attributes
        lons = []
        lats = []
        for j in range(self.raster['dims'][1] - 1):
            for i in range(self.raster['dims'][0] - 1):
                lons.append(
                    self.raster['affine'][1] * i + self.raster['affine'][0])
                lats.append(
                    self.raster['affine'][-3] + self.raster['affine'][-1] * (
                            self.raster['dims'][1] - j - 1))

        self.lons = np.array(lons)
        self.lats = np.array(lats)

    def get_strain(self):
        """
        Reads a raw strain 2d data containing (exx, eyy, exy) and calculate
        strain tensor properties assuming plane strain or stress
        Implemented:
        - J2
        - Areal strain
        - 3D strain state phi (eps_2 - eps_3)/(eps_1 - eps_3)
        - 2D strain state psi (eps_1 + eps_3)/(abs(eps_1) + abs(eps_3)
        """

        if not all(
                x in ('exx', 'eyy', 'exy') for x in self.raster['attributes']):
            self.data = {j: self.raster['data'][:, i] for i, j in
                         enumerate(self.raster['attributes'])}
            return

        data = self.raster['data']
        mask = self.raster['mask']

        Regime = []
        Area = []
        Phi = []
        Psi = []
        J2 = []
        Tau_max = []
        SS = []  # Savage and Simpson

        for i, j in zip(data, mask):
            if not j:

                eps_i = np.array([[i[0], i[2]], [i[2], i[1]]])
                val, vec = np.linalg.eig(eps_i)
                if val[0] < 0 and val[1] < 0:
                    regime = -1
                elif val[0] > 0 and val[1] > 0:
                    regime = 1
                else:
                    regime = 0
                Regime.append(regime)
                sigma_m = (i[0] + i[1]) / 2.
                Area.append(sigma_m)

                val_2d = np.sort(np.array([val[0], val[1]]))
                val_3d = np.sort(np.array([val[0], val[1], 0]))
                j2 = np.sqrt(3 * 0.5 * (
                    np.sum([(i - sigma_m) ** 2 for i in val_3d])))
                J2.append(j2)
                phi = (val_3d[1] - val_3d[2]) / (val_3d[0] - val_3d[2])

                tau_max = np.abs(val_3d[0] - val_3d[2]) / 2.
                Tau_max.append(tau_max)
                Phi.append(phi)
                Psi.append((val_2d[0] + val_2d[1]) / (
                        np.abs(val_2d[0]) + np.abs(val_2d[1])))
                ss = np.max([np.abs(val_2d[0]), np.abs(val_2d[1]),
                             np.abs(val_2d[0] + val_2d[1])])
                SS.append(ss)

            else:
                Regime.append(np.nan)
                Area.append(np.nan)
                Phi.append(np.nan)
                Psi.append(np.nan)
                J2.append(np.nan)
                Tau_max.append(np.nan)
                SS.append(np.nan)

        data_dict = {'eps': data, 'regime': np.array(Regime),
                     'eps_a': np.array(Area), 'phi': np.array(Phi),
                     'psi': np.array(Psi), 'j2': np.array(J2),
                     'tau_max': np.array(Tau_max), 'ss': np.array(SS)}
        self.data = data_dict

    @staticmethod
    def _get_range(array, confidence, one_sided=False):
        """
        Returns range of an array, within a confidence interval
        """
        array = array[~np.isnan(array[:])]
        if one_sided:
            lower = np.quantile(array, 1 - confidence)
            upper = np.quantile(array, 1)
        else:
            lower = np.quantile(array, (1 - confidence) / 2.)
            upper = np.quantile(array, 1 - (1 - confidence) / 2.)
        return np.array([lower, upper])

    @staticmethod
    def _get_bins(array, k=3, range_=None, method='eq_interval', n_disc=25,
                  n_bins=100):
        """
        Returns binning by equal intervals, equal quantiles, max entropy or
         K-means

        :param array: Data to be binned
        :param k: Number of intervals
        :param range_: pre-established range within the binning is performed,
         for equal intervals
        :param method: 'eq_interal' or 'eq_quantile'
        :return: array_(k-1)  - Inner boundaries of the binning
        """

        bins = None
        if range_ is None:
            range_ = [np.nanquantile(array, 0.025),
                      np.nanquantile(array, 0.975)]

        if method == 'eq_interval':
            bins = np.linspace(range_[0], range_[1], k + 1)[1:-1]

        elif method == 'eq_quantile':
            bins = []
            for i in np.linspace(0, 1, k + 1)[1:-1]:
                bins.append(np.nanquantile(array, i))
        elif method == 'eq_entropy':

            values = array[~np.isnan(array)]
            cut_points = np.linspace(range_[0], range_[1], n_disc,
                                     endpoint=False)[1:]
            discretizations = [np.hstack([range_[0], i, range_[1]]) for i in
                               itertools.combinations(cut_points, k - 1)]
            print('max ent iterations %i' % len(discretizations))
            h = np.zeros((len(discretizations), k))
            n_i = np.zeros((len(discretizations), k))
            for i, intervals in enumerate(discretizations):
                for j in range(len(intervals) - 1):
                    element = values[np.where(
                        np.logical_and(values >= intervals[j],
                                       values < intervals[j + 1]))]
                    n_i[i, j] = len(element)
                    h[i, j] = entropy(np.histogram(element, bins=n_bins)[0])

            H = np.sum(h * n_i / len(values), axis=1)
            ind = np.argmax(H)

            bins = [i for i in discretizations[ind][1:-1]]

        elif method == 'kmeans':

            array = array[~np.isnan(array)].reshape((-1, 1))
            est = KBinsDiscretizer(n_bins=k, encode='ordinal',
                                   strategy='kmeans')
            est.fit(array)
            bins = est.bin_edges_[0][1:-1]

        elif method == 'mdlp':
            mdlp = MDLP()
            bins = mdlp.cut_points(array[:2000], array[1000:3000])

        return bins

    @staticmethod
    def _categorize_array(array, mask, bins):

        cat_array = np.digitize(array, bins).astype(float)
        cat_array[mask] = np.nan
        return cat_array

    @staticmethod
    def _categorized_product(cat_1, cat_2, bins_1, bins_2):

        a, b = np.meshgrid(np.arange(len(bins_1) + 1),
                           np.arange(len(bins_2) + 1))
        ind = np.vstack((a.ravel(), b.ravel())).T.astype(float)
        product = []
        for i, j in zip(cat_1, cat_2):
            if np.isnan(i) or np.isnan(j):
                product.append(np.nan)
            else:
                index = np.where(np.all(ind == [i, j], axis=1))[0][0]
                product.append(index)
        product = np.array(product)

        return product, ind

    def bin_measure(self, measures, bin_discr, method='kmeans'):

        if isinstance(measures, str):
            measures = [measures]

        for measure in measures:
            print('Binning measure %s' % measure)
            self._get_range(self.data[measure], 0.99)
            self.bin_edges[measure] = []
            min_tree = []
            for binx in bin_discr:
                print('Processing bin %i' % binx)
                bin_edges = self._get_bins(array=self.data[measure],
                                           k=binx,
                                           method=method)
                categorized_array = self._categorize_array(
                    array=self.data[measure],
                    mask=self.raster['mask'],
                    bins=bin_edges)

                min_tree.append(categorized_array)
                self.bin_edges[measure].append(bin_edges)

            self.data[measure + '_disc'] = np.array(min_tree).T

    @staticmethod
    def _image_processing(input_array, n_bins, array_mask, dims, smooth=1):

        array = input_array.reshape((dims[1] - 1, dims[0] - 1))
        nanmask = array_mask.reshape((dims[1] - 1, dims[0] - 1))
        index_masks = []

        for i in range(n_bins):
            mask = np.where(array >= n_bins - i)
            mask_2d = np.zeros((dims[1] - 1, dims[0] - 1))
            mask_2d[mask[0], mask[1]] = 1
            index_masks.append(mask_2d)

        images = []

        for mask in index_masks:
            closed = binary_closing(mask, structure=np.ones((smooth, smooth)))
            new_img = remove_small_objects(closed, 1000)
            new_img = remove_small_holes(new_img, 900)
            if len(np.unique(new_img)) != len(np.unique(closed)):
                print('woops')
                new_img = closed
            images.append(new_img)

        images.reverse()
        images2 = []
        for mask in images:
            closed = binary_closing(mask, structure=np.ones((smooth, smooth)))
            medi = median(skimage.img_as_ubyte(closed), disk(5),
                          mask=1 - nanmask)
            new_img = remove_small_objects(medi.astype(bool), 900)
            new_img = remove_small_holes(new_img, 800)
            if len(np.unique(new_img)) != len(np.unique(closed)):
                print('woops2')
                new_img = medi
            images2.append(new_img.astype(int))
        images = images2

        compound_img = np.zeros((dims[1] - 1, dims[0] - 1))
        labeled_img = np.zeros((dims[1] - 1, dims[0] - 1))
        offset = 0
        for i, im in enumerate(images):
            ind = np.where(im == 1)
            compound_img[ind[0], ind[1]] = i + 1

            labeled = label(im, background=0)
            labeled_img[ind[0], ind[1]] = labeled[ind[0], ind[1]] + offset
            offset += np.max(labeled)

        compound_array = compound_img.ravel()
        compound_array[array_mask] = np.nan

        labeled_array = labeled_img.ravel()
        labeled_array[array_mask] = np.nan

        return compound_array

    def image_proc(self, measures, smooth=4):

        if isinstance(measures, str):
            measures = [measures]
        for measure in measures:
            for n_bin in range(len(self.bin_edges[measure])):
                array = self.data[measure + '_disc'][:, n_bin]
                array_proc = self._image_processing(input_array=array,
                                                    n_bins=int(
                                                        np.nanmax(array)),
                                                    array_mask=self.raster[
                                                        'mask'],
                                                    dims=self.raster['dims'],
                                                    smooth=smooth)
                self.data[measure + '_disc'][:, n_bin] = array_proc

    def extrapolate_to_polygon(self, polygon, measures):

        from nonpoisson import geo
        if isinstance(measures, str):
            measures = [measures]

        geo.rasterize_polygons(polygon, paths.get('spatial',
                                                  'raster', 'new_grid'),
                               ['id'], res=(0.02, 0.02), del_shp=False)
        raster_new = geo.read_raster(paths.get('spatial', 'raster',
                                               'new_grid'), [0])

        lons_new = []
        lats_new = []

        for j in range(raster_new['dims'][1] - 1):
            for i in range(raster_new['dims'][0] - 1):
                lons_new.append(
                    raster_new['affine'][1] * i + raster_new['affine'][0])
                lats_new.append(
                    raster_new['affine'][-3] + raster_new['affine'][-1] * (
                            raster_new['dims'][1] - j - 1))
        coords = []
        hash_idx = []
        i = 0

        for x, y, m in zip(lons_new, lats_new, raster_new['mask']):
            if not m:
                coords.append([x, y])
                hash_idx.append(i)
            i += 1
        coords = np.array(coords)
        grid_old = np.vstack((self.lons[~self.raster['mask']],
                              self.lats[~self.raster['mask']])).T

        new_data = dict()

        for measure in measures:
            array_old = self.data[measure + '_disc'][~self.raster['mask']]
            interp = sci.NearestNDInterpolator(grid_old, array_old)
            array = interp(coords[:, 0], coords[:, 1])

            array_new = np.nan * np.ones((len(lons_new), array_old.shape[1]))

            array_new[np.array(hash_idx), :] = array
            new_data[measure + '_disc'] = array_new
        new_data['grid'] = np.vstack((lons_new, lats_new)).T
        self.raster = raster_new
        self.lats = np.array(lats_new)
        self.lons = np.array(lons_new)
        self.data = new_data

    def intersect_by_polygon(self, polygon, measure, binx):

        if isinstance(polygon, str):
            polygon = fiona.open(polygon)
        poly_coords = polygon[0]['geometry']['coordinates'][0]
        nz_poly = shpoly(shell=np.array(poly_coords))

        polygons_0 = self.polygons[(measure, binx)]
        new_poly = []
        for i, pol in enumerate(polygons_0):
            coords_poly = pol[0]['coordinates']
            if len(coords_poly) == 1:
                shp_poly = shpoly(shell=coords_poly[0])
            else:
                shp_poly = shpoly(shell=coords_poly[0], holes=coords_poly[1:])
            intersection = nz_poly.intersection(shp_poly)

            if isinstance(intersection, shpoly):
                aux = copy.deepcopy(pol)

                c = intersection.exterior.xy

                if len(c[0]) == 0:
                    pass
                else:
                    aux[0]['coordinates'] = [
                        [(i, j) for i, j in zip(c[0], c[1])]]
                    new_poly.append(aux)
            if isinstance(intersection, MultiPolygon):
                aux = copy.deepcopy(pol)

                for p in intersection.geoms:
                    aux_p = copy.deepcopy(aux)
                    c = p.exterior.xy
                    if len(c[0]) == 0:
                        continue
                    interiors = p.interiors
                    aux_p[0]['coordinates'] = [
                        [(m, n) for m, n in zip(c[0], c[1])]]
                    if len(interiors) != 0:
                        for z in interiors:
                            aux_p[0]['coordinates'].append(
                                [(m, n) for m, n in zip(z.xy[0], z.xy[1])])
                    new_poly.append(aux_p)

        self.polygons[(measure, binx)] = new_poly

        shp_schema = {'geometry': 'MultiPolygon',
                      'properties': {'pixelvalue': 'float'}}

        from shapely.geometry import shape, mapping, Point
        with fiona.open('test.shp', 'w', 'ESRI Shapefile', shp_schema,
                        'epsg:4326') as shp:
            for poly, value in [(shape(geom), value) for geom, value in
                                new_poly]:
                multipolygon = MultiPolygon([poly])
                shp.write({'geometry': mapping(multipolygon),
                           'properties': {'pixelvalue': float(value)}})


    def include_region(self, polygon, measures):

        if isinstance(measures, str):
            measures = [measures]

        poly_points = np.genfromtxt(polygon)
        poly_region = shpoly(poly_points)

        points_inside = []
        for n, i in enumerate(self.data['grid']):
            point = shpoint(i)
            if poly_region.contains(point):
                points_inside.append(n)
        points_inside = np.array(points_inside)
        for measure in measures:
            new_bins = np.nanmax(self.data[measure + '_disc'], axis=0) + 1
            self.data[measure + '_disc'][points_inside] = new_bins

            # print(self.data[measure + '_disc'][points_inside])
            # array_old = self.data[measure + '_disc']

        # new_data['grid'] = np.vstack((lons_new, lats_new)).T

        # self.data = new_data

    def bins_polygonize(self, measures, n_bins, load=False, post_proc=True):

        if isinstance(measures, str):
            measures = [measures]
        for measure in measures:
            for n_bin in n_bins:
                if post_proc:
                    prefix = 'post_proc/'
                else:
                    prefix = ''
                filename = paths.get('spatial', 'shp',
                                     f'{prefix}{self.name}' + '_%s_%i' % (measure, n_bin))
                if load:
                    shp_schema = {'geometry': 'MultiPolygon',
                                  'properties': {'pixelvalue': 'float'}}
                    epsg = 'EPSG:4326'
                    crs = rasterio.crs.CRS.from_epsg(epsg.split(':')[-1])
                    polygons = []
                    with fiona.open(filename, 'r', 'ESRI Shapefile',
                                    shp_schema, crs) as shp:
                        for poly in shp:
                            polygons.append((poly['geometry'],
                                             poly['properties']['pixelvalue']))

                else:
                    from nonpoisson import geo
                    bins = [len(i) for i in self.bin_edges[measure]]
                    bin_col = np.where(np.array(bins) == n_bin - 1)[0][0]
                    array = self.data[measure + '_disc'][:, bin_col]

                    polygons = geo.polygonize_array(array, self.raster,
                                                    savepath=filename)
                self.polygons[(measure, n_bin)] = polygons

    def get_oqpolygons(self, measure, n_bins):
        oqpolygons = []
        polygon_values = []
        for i, n_poly in enumerate(self.polygons[(measure, n_bins)]):
            shell_sphe = n_poly[0]['coordinates'][0]
            holes_sphe = n_poly[0]['coordinates'][1:]
            poly_value = n_poly[1]
            proj = OrthographicProjection.from_lons_lats(
                np.array([i[0] for i in shell_sphe]),
                np.array([i[1] for i in shell_sphe]))
            poly = shpoly(shell=np.array(proj(*np.array(shell_sphe).T)).T,
                          holes=[np.array(proj(*np.array(i).T)).T for i in
                                 holes_sphe])
            oqpolygons.append(oqpoly._from_2d(poly, proj))
            polygon_values.append(poly_value)

        return oqpolygons, polygon_values

    @staticmethod
    def plot_histogram(A, bins, range_=None, title=None, xlims=[None, None]
                       , ylabel='PDF',
                       var=None, save_path='histogram.png',
                       legend=True, legend_size=12, dpi=300):

        fig = plt.figure(figsize=(6, 5))
        hist = plt.hist(A, bins=75, color='steelblue', density=True, alpha=0.6)
        plt.title(title)
        if range_ is None:
            range_ = [np.nanquantile(A, 0.005), np.nanquantile(A, 0.995)]
        plt.axvline(range_[0], color='red', linestyle='--')
        plt.axvline(range_[1], color='red', linestyle='--')

        legend_elements = [Line2D([0], [0], color='steelblue',
                                  lw=5, label=var + r' distribution'),
                           Line2D([0], [0], color='red', lw=1,
                                  linestyle='--', label=r'99% mass intervals')]
        if len(bins):
            legend_elements.append(Line2D([0], [0], color='green',
                                          lw=1, linestyle='--',
                                          label='Bin edges'))
        for i in bins:
            plt.axvline(i, linewidth=2, color='green', linestyle='--')
        if legend:
            fig.get_axes()[0].legend(handles=legend_elements,
                                     fontsize=legend_size,
                                     loc='best',
                                     borderaxespad=0.8)

        limits = np.array([range_[0], *bins, range_[1]])
        midpoint_bins = [(limits[i] + limits[i + 1]) / 2 for i in
                         range(len(limits) - 1)]
        for i, m in enumerate(midpoint_bins):
            plt.text(m, np.max(hist[0]), f'{i + 1}', color='darkgreen',
                     fontsize=16, va='bottom', ha='center')

        fig.get_axes()[0].set_xlabel(var, fontsize=28)
        fig.get_axes()[0].set_ylabel(ylabel, fontsize=28)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlim(xlims)
        plt.ylim([0.01, None])
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi)
        plt.show()
        return hist

    def write_vti(self, attributes=None, vtk_name=None, epsg='epsg:2193',
                  res=(1000, 1000)):

        from nonpoisson import geo

        if attributes is None:
            attributes = self.data.keys()
        else:
            attributes = [i for i in self.data.keys() if
                          any([j in i for j in attributes])]

        if vtk_name is None:
            vtk_name = self.paths['vtk']

        struct = {}
        n = 0
        for key in attributes:
            if len(self.data[key].shape) == 1:
                struct[key] = [n]
                n += 1
            elif len(self.data[key].shape) == 2:
                indexes = []
                for i in range(self.data[key].shape[1]):
                    indexes.append(n)
                    n += 1
                struct[key] = indexes

        n_atts = np.max(
            list(functools.reduce(lambda i, j: i + j, struct.values()))) + 1
        data_resampled = np.zeros((self.raster['mask'].shape[0], n_atts))
        for key, item in struct.items():
            for i, it in enumerate(item):
                if len(item) == 1:
                    data_resampled[:, it] = self.data[key].squeeze()
                else:
                    data_resampled[:, it] = self.data[key][:, i]

        geo.numpy2raster(self.paths['raster'], data_resampled,
                         self.raster['affine'], self.raster['dims'])
        geo.reproject_rio(self.paths['raster'], self.paths['raster'],
                          epsg, res=res, bounds=None, resample='nearest')
        geo.raster2vti(vtk_name, self.paths['raster'], struct)

    @classmethod
    def load(cls, model_name, filename=None):
        """
        Loads a serialized Model_results object
        :param filename:
        :return:
        """
        if filename:
            with open(filename, 'rb') as f:
                obj = pickle.load(f)

            return obj
        else:
            with open(paths.get('spatial', 'serial', model_name), 'rb') as f:
                obj = pickle.load(f)
            return obj

    def save(self, filename=None):
        """
        Serializes Model_results object into a file
        :param filename: If None, save in results folder named with self.name
        """
        if filename:
            with open(filename, 'wb') as obj:
                pickle.dump(self, obj)
        else:
            with open(self.paths['serial'], 'wb') as obj:
                pickle.dump(self, obj)

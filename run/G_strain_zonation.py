import os
import time
from os.path import join
from nonpoisson import paths
from nonpoisson.zonation import GeodeticModel


def create_strain_model():
    print('Processing model: ', paths.model_names)
    metrics = ['j2', 'tau_max', 'ss']
    name = paths.model_names[0]
    model = GeodeticModel(name)
    model.import_data()
    model.get_strain()

    # Create paraview files
    paraview_path = join(paths.ms1_figs['fig11'], 'paraview')
    os.makedirs(paraview_path, exist_ok=True)
    model.write_vti(
        metrics,
        vtk_name=join(paraview_path, 'strain_map.vti'),
        epsg='epsg:2193')
    model.save()


def make_zonation():
    metrics = ['j2', 'tau_max', 'ss']
    bin_numbers = [3, 4, 5, 6]

    name = paths.model_names[0]
    print('Processing Model %s' % name)
    start = time.process_time()

    model = GeodeticModel.load(name)
    model.bin_measure(metrics, bin_numbers, method='eq_entropy')
    model.image_proc(metrics, smooth=10)
    model.bins_polygonize(metrics, bin_numbers)

    # Plot strain histogram binning (for Figure 12)
    for b, number in enumerate(bin_numbers):
        legend = False
        if b == 4:
            legend = True
        model.plot_histogram(
            model.data['j2'],
            model.bin_edges['j2'][b],
            var=r'$J_2$', xlims=[0, 0.5],
            ylabel='$f_{J_2}$', legend=legend, legend_size=20,
            save_path=join(paths.ms1_figs['fig12'],
                           f'hist_maxent_{b+1}.png'))

    # Create paraview files (for Figure 12)
    paraview_bins_path = join(paths.ms1_figs['fig12'], 'paraview')
    os.makedirs(paraview_bins_path, exist_ok=True)
    model.write_vti(metrics, vtk_name=join(paraview_bins_path,
                                           'strain_bins_map.vti'))

    # Create final Shapefiles for NZ forecasts (with TVZ and CSEP region)
    model.extrapolate_to_polygon(paths.region_nz_collection, metrics)
    model.include_region(paths.region_tvz_corr, metrics)
    model.bins_polygonize(metrics, bin_numbers)
    print('\tTime of processing: %.1f' % (time.process_time() - start))


if __name__ == '__main__':

    create_strain_model()
    make_zonation()

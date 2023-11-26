# state file generated using paraview version 5.11.1
import paraview

paraview.compatibility.major = 5
paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()
import matplotlib.pyplot as plt


def makefig_j2():
    # ----------------------------------------------------------------
    # setup views used in the visualization
    # ----------------------------------------------------------------

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # Create a new 'Render View'
    renderView1 = CreateView('RenderView')
    renderView1.ViewSize = [645, 786]
    renderView1.InteractionMode = '2D'
    renderView1.AxesGrid = 'GridAxes3DActor'
    renderView1.OrientationAxesVisibility = 0
    renderView1.CenterOfRotation = [1532215.0607010862, 5444402.662813947, 0.0]
    renderView1.StereoType = 'Crystal Eyes'
    renderView1.CameraPosition = [1567009.4788373362, 5462720.144816798,
                                  4937900.0]
    renderView1.CameraFocalPoint = [1567009.4788373362, 5462720.144816798, 0.0]
    renderView1.CameraFocalDisk = 1.0
    renderView1.CameraParallelScale = 741825.0763008723
    renderView1.BackEnd = 'OSPRay raycaster'
    renderView1.OSPRayMaterialLibrary = materialLibrary1

    SetActiveView(None)

    # ----------------------------------------------------------------
    # setup view layouts
    # ----------------------------------------------------------------

    # create new layout object 'Layout #1'
    layout1 = CreateLayout(name='Layout #1')
    layout1.AssignView(0, renderView1)
    layout1.SetSize(645, 786)

    # ----------------------------------------------------------------
    # restore active view
    SetActiveView(renderView1)
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # setup the data processing pipelines
    # ----------------------------------------------------------------

    # create a new 'XML Image Data Reader'
    strain_bins_mapvti = XMLImageDataReader(
        registrationName='strain_bins_map.vti',
        FileName=['paraview/strain_bins_map.vti'])
    strain_bins_mapvti.CellArrayStatus = ['j2', 'tau_max', 'ss', 'j2_disc',
                                          'tau_max_disc', 'ss_disc', 'mask']
    strain_bins_mapvti.TimeArray = 'None'

    # create a new 'Clip'
    clip1 = Clip(registrationName='Clip1', Input=strain_bins_mapvti)
    clip1.ClipType = 'Scalar'
    clip1.HyperTreeGridClipper = 'Plane'
    clip1.Scalars = ['CELLS', 'mask']
    clip1.Value = 0.2901383340385938
    clip1.Invert = 0

    # init the 'Plane' selected for 'HyperTreeGridClipper'
    clip1.HyperTreeGridClipper.Origin = [1532215.0607010862, 5444402.662813947,
                                         0.0]

    # create a new 'XML Image Data Reader'
    basemap_2193vti = XMLImageDataReader(
        registrationName='basemap_2193.vti',
        FileName=['paraview/basemap_2193.vti'])
    basemap_2193vti.CellArrayStatus = ['basemap', 'mask']
    basemap_2193vti.TimeArray = 'None'

    # ----------------------------------------------------------------
    # setup the visualization in view 'renderView1'
    # ----------------------------------------------------------------

    # show data from basemap_2193vti
    basemap_2193vtiDisplay = Show(basemap_2193vti, renderView1,
                                  'UniformGridRepresentation')

    # get 2D transfer function for 'basemap'
    basemapTF2D = GetTransferFunction2D('basemap')
    basemapTF2D.ScalarRangeInitialized = 1
    basemapTF2D.Range = [0.0, 441.6729559300637, 0.0, 1.0]

    # get color transfer function/color map for 'basemap'
    basemapLUT = GetColorTransferFunction('basemap')
    basemapLUT.TransferFunction2D = basemapTF2D
    basemapLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941,
                            220.83647796503186, 0.865003, 0.865003, 0.865003,
                            441.6729559300637, 0.705882, 0.0156863, 0.14902]
    basemapLUT.ScalarRangeInitialized = 1.0

    # get opacity transfer function/opacity map for 'basemap'
    basemapPWF = GetOpacityTransferFunction('basemap')
    basemapPWF.Points = [0.0, 0.0, 0.5, 0.0, 441.6729559300637, 1.0, 0.5, 0.0]
    basemapPWF.ScalarRangeInitialized = 1

    # trace defaults for the display properties.
    basemap_2193vtiDisplay.Representation = 'Slice'
    basemap_2193vtiDisplay.ColorArrayName = ['CELLS', 'basemap']
    basemap_2193vtiDisplay.LookupTable = basemapLUT
    basemap_2193vtiDisplay.MapScalars = 0
    basemap_2193vtiDisplay.SelectTCoordArray = 'None'
    basemap_2193vtiDisplay.SelectNormalArray = 'None'
    basemap_2193vtiDisplay.SelectTangentArray = 'None'
    basemap_2193vtiDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    basemap_2193vtiDisplay.SelectOrientationVectors = 'None'
    basemap_2193vtiDisplay.ScaleFactor = 276600.0
    basemap_2193vtiDisplay.SelectScaleArray = 'mask'
    basemap_2193vtiDisplay.GlyphType = 'Arrow'
    basemap_2193vtiDisplay.GlyphTableIndexArray = 'mask'
    basemap_2193vtiDisplay.GaussianRadius = 13830.0
    basemap_2193vtiDisplay.SetScaleArray = [None, '']
    basemap_2193vtiDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    basemap_2193vtiDisplay.OpacityArray = [None, '']
    basemap_2193vtiDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    basemap_2193vtiDisplay.DataAxesGrid = 'GridAxesRepresentation'
    basemap_2193vtiDisplay.PolarAxes = 'PolarAxesRepresentation'
    basemap_2193vtiDisplay.ScalarOpacityUnitDistance = 19311.9664193523
    basemap_2193vtiDisplay.ScalarOpacityFunction = basemapPWF
    basemap_2193vtiDisplay.TransferFunction2D = basemapTF2D
    basemap_2193vtiDisplay.OpacityArrayName = ['CELLS', 'mask']
    basemap_2193vtiDisplay.ColorArray2Name = ['CELLS', 'mask']
    basemap_2193vtiDisplay.IsosurfaceValues = [0.5]
    basemap_2193vtiDisplay.SliceFunction = 'Plane'
    basemap_2193vtiDisplay.SelectInputVectors = [None, '']
    basemap_2193vtiDisplay.WriteLog = ''

    # init the 'Plane' selected for 'SliceFunction'
    basemap_2193vtiDisplay.SliceFunction.Origin = [1575927.8735133181,
                                                   5400949.511356351, -10.0]

    # show data from clip1
    clip1Display = Show(clip1, renderView1, 'UnstructuredGridRepresentation')

    # get 2D transfer function for 'j2_disc'
    j2_discTF2D = GetTransferFunction2D('j2_disc')
    j2_discTF2D.ScalarRangeInitialized = 1
    j2_discTF2D.Range = [-0.5, 3.5, 0.0, 1.0]

    # get color transfer function/color map for 'j2_disc'
    j2_discLUT = GetColorTransferFunction('j2_disc')
    j2_discLUT.AutomaticRescaleRangeMode = 'Grow and update every timestep'
    j2_discLUT.AnnotationsInitialized = 1
    j2_discLUT.TransferFunction2D = j2_discTF2D
    j2_discLUT.RGBPoints = [-0.5, 0.054902, 0.109804, 0.121569,
                            -0.29999999999999993, 0.07451, 0.172549, 0.180392,
                            -0.09999999999999992, 0.086275, 0.231373, 0.219608,
                            0.09999999999999998, 0.094118, 0.278431, 0.25098,
                            0.30000000000000016, 0.109804, 0.34902, 0.278431,
                            0.5, 0.113725, 0.4, 0.278431, 0.7, 0.117647,
                            0.45098, 0.270588, 0.8999999999999997, 0.117647,
                            0.490196, 0.243137, 1.1000000000000003, 0.113725,
                            0.521569, 0.203922, 1.3, 0.109804, 0.54902,
                            0.152941, 1.5, 0.082353, 0.588235, 0.082353,
                            1.7000000000000002, 0.109804, 0.631373, 0.05098,
                            1.9, 0.211765, 0.678431, 0.082353, 2.1, 0.317647,
                            0.721569, 0.113725, 2.2999999999999994, 0.431373,
                            0.760784, 0.160784, 2.5, 0.556863, 0.8, 0.239216,
                            2.7000000000000006, 0.666667, 0.839216, 0.294118,
                            2.9, 0.784314, 0.878431, 0.396078, 3.1, 0.886275,
                            0.921569, 0.533333, 3.2999999999999994, 0.960784,
                            0.94902, 0.670588, 3.5, 1.0, 0.984314, 0.901961]
    j2_discLUT.ColorSpace = 'Lab'
    j2_discLUT.NanColor = [0.1725490196078431, 0.6352941176470588,
                           0.3725490196078431]
    j2_discLUT.NanOpacity = 0.0
    j2_discLUT.NumberOfTableValues = 4
    j2_discLUT.ScalarRangeInitialized = 1.0
    j2_discLUT.VectorComponent = 1
    j2_discLUT.VectorMode = 'Component'
    j2_discLUT.Annotations = ['2', '', '1', '', '0', '']
    j2_discLUT.ActiveAnnotatedValues = ['0', '1', '2']
    j2_discLUT.IndexedColors = [0.8980392156862745, 0.9607843137254902,
                                0.9764705882352941, 0.6, 0.8470588235294118,
                                0.788235294117647, 0.06274509803921569,
                                0.22745098039215686, 0.12941176470588237]
    j2_discLUT.IndexedOpacities = [1.0, 1.0, 1.0]

    # get opacity transfer function/opacity map for 'j2_disc'
    j2_discPWF = GetOpacityTransferFunction('j2_disc')
    j2_discPWF.Points = [-0.5, 0.7336956858634949, 0.5, 0.0,
                         0.07709246873855591, 0.8478261232376099, 0.5, 0.0,
                         1.094713568687439, 0.875, 0.5, 0.0, 3.5, 1.0, 0.5,
                         0.0]
    j2_discPWF.ScalarRangeInitialized = 1

    # trace defaults for the display properties.
    clip1Display.Representation = 'Surface'
    clip1Display.ColorArrayName = ['CELLS', 'j2_disc']
    clip1Display.LookupTable = j2_discLUT
    clip1Display.Opacity = 0.82
    clip1Display.SelectTCoordArray = 'None'
    clip1Display.SelectNormalArray = 'None'
    clip1Display.SelectTangentArray = 'None'
    clip1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    clip1Display.SelectOrientationVectors = 'None'
    clip1Display.ScaleFactor = 147400.0
    clip1Display.SelectScaleArray = 'None'
    clip1Display.GlyphType = 'Arrow'
    clip1Display.GlyphTableIndexArray = 'None'
    clip1Display.GaussianRadius = 7370.0
    clip1Display.SetScaleArray = [None, '']
    clip1Display.ScaleTransferFunction = 'PiecewiseFunction'
    clip1Display.OpacityArray = [None, '']
    clip1Display.OpacityTransferFunction = 'PiecewiseFunction'
    clip1Display.DataAxesGrid = 'GridAxesRepresentation'
    clip1Display.PolarAxes = 'PolarAxesRepresentation'
    clip1Display.ScalarOpacityFunction = j2_discPWF
    clip1Display.ScalarOpacityUnitDistance = 18060.107275319242
    clip1Display.OpacityArrayName = ['CELLS', 'j2']
    clip1Display.SelectInputVectors = [None, '']
    clip1Display.WriteLog = ''

    # setup the color legend parameters for each legend in this view

    # get color legend/bar for j2_discLUT in view renderView1
    j2_discLUTColorBar = GetScalarBar(j2_discLUT, renderView1)
    j2_discLUTColorBar.WindowLocation = 'Any Location'
    j2_discLUTColorBar.Position = [0.7845736434108527, 0.056895674300254426]
    j2_discLUTColorBar.Title = '$J_2$'
    j2_discLUTColorBar.ComponentTitle = 'Bin'
    j2_discLUTColorBar.HorizontalTitle = 1
    j2_discLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    j2_discLUTColorBar.TitleFontSize = 32
    j2_discLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    j2_discLUTColorBar.LabelFontSize = 20
    j2_discLUTColorBar.ScalarBarThickness = 18
    j2_discLUTColorBar.ScalarBarLength = 0.22821882951653955
    j2_discLUTColorBar.DrawScalarBarOutline = 1
    j2_discLUTColorBar.ScalarBarOutlineColor = [0.0, 0.0, 0.0]
    j2_discLUTColorBar.LabelFormat = '%.2f'
    j2_discLUTColorBar.DrawTickMarks = 0
    j2_discLUTColorBar.UseCustomLabels = 1
    j2_discLUTColorBar.CustomLabels = [0.0, 1.0, 2.0, 3.0]
    j2_discLUTColorBar.AddRangeLabels = 0
    j2_discLUTColorBar.RangeLabelFormat = '%.2f'
    j2_discLUTColorBar.DrawAnnotations = 0

    # set color bar visibility
    j2_discLUTColorBar.Visibility = 1

    # show color legend
    clip1Display.SetScalarBarVisibility(renderView1, True)

    # ----------------------------------------------------------------
    # setup color maps and opacity mapes used in the visualization
    # note: the Get..() functions create a new object, if needed
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # restore active source
    SetActiveSource(clip1)
    # ----------------------------------------------------------------
    SaveExtracts(ExtractsOutputDirectory='extracts')
    res = [1290, 1572]
    SaveScreenshot('figure_j2.png',
                   renderView1, ImageResolution=[2 * i for i in res])


def makefig_tau_max():
    # ----------------------------------------------------------------
    # setup views used in the visualization
    # ----------------------------------------------------------------

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # Create a new 'Render View'
    renderView1 = CreateView('RenderView')
    renderView1.ViewSize = [645, 786]
    renderView1.InteractionMode = '2D'
    renderView1.AxesGrid = 'GridAxes3DActor'
    renderView1.OrientationAxesVisibility = 0
    renderView1.CenterOfRotation = [1532215.0607010862, 5444402.662813947, 0.0]
    renderView1.StereoType = 'Crystal Eyes'
    renderView1.CameraPosition = [1567009.4788373362, 5462720.144816798,
                                  4937900.0]
    renderView1.CameraFocalPoint = [1567009.4788373362, 5462720.144816798, 0.0]
    renderView1.CameraFocalDisk = 1.0
    renderView1.CameraParallelScale = 741825.0763008723
    renderView1.BackEnd = 'OSPRay raycaster'
    renderView1.OSPRayMaterialLibrary = materialLibrary1

    SetActiveView(None)

    # ----------------------------------------------------------------
    # setup view layouts
    # ----------------------------------------------------------------

    # create new layout object 'Layout #1'
    layout1 = CreateLayout(name='Layout #1')
    layout1.AssignView(0, renderView1)
    layout1.SetSize(645, 786)

    # ----------------------------------------------------------------
    # restore active view
    SetActiveView(renderView1)
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # setup the data processing pipelines
    # ----------------------------------------------------------------

    # create a new 'XML Image Data Reader'
    basemap_2193vti = XMLImageDataReader(registrationName='basemap_2193.vti',
                                         FileName=[
                                             '/home/pciturri/PycharmProjects/nonpoisson_nz/figures/forecasts_ms/figure12/paraview/basemap_2193.vti'])
    basemap_2193vti.CellArrayStatus = ['basemap', 'mask']
    basemap_2193vti.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    strain_bins_mapvti = XMLImageDataReader(
        registrationName='strain_bins_map.vti', FileName=[
            '/home/pciturri/PycharmProjects/nonpoisson_nz/figures/forecasts_ms/figure12/paraview/strain_bins_map.vti'])
    strain_bins_mapvti.CellArrayStatus = ['j2', 'tau_max', 'ss', 'j2_disc',
                                          'tau_max_disc', 'ss_disc', 'mask']
    strain_bins_mapvti.TimeArray = 'None'

    # create a new 'Clip'
    clip1 = Clip(registrationName='Clip1', Input=strain_bins_mapvti)
    clip1.ClipType = 'Scalar'
    clip1.HyperTreeGridClipper = 'Plane'
    clip1.Scalars = ['CELLS', 'mask']
    clip1.Value = 0.2901383340385938
    clip1.Invert = 0

    # init the 'Plane' selected for 'HyperTreeGridClipper'
    clip1.HyperTreeGridClipper.Origin = [1532215.0607010862, 5444402.662813947,
                                         0.0]

    # ----------------------------------------------------------------
    # setup the visualization in view 'renderView1'
    # ----------------------------------------------------------------

    # show data from basemap_2193vti
    basemap_2193vtiDisplay = Show(basemap_2193vti, renderView1,
                                  'UniformGridRepresentation')

    # get 2D transfer function for 'basemap'
    basemapTF2D = GetTransferFunction2D('basemap')
    basemapTF2D.ScalarRangeInitialized = 1
    basemapTF2D.Range = [0.0, 441.6729559300637, 0.0, 1.0]

    # get color transfer function/color map for 'basemap'
    basemapLUT = GetColorTransferFunction('basemap')
    basemapLUT.TransferFunction2D = basemapTF2D
    basemapLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941,
                            220.83647796503186, 0.865003, 0.865003, 0.865003,
                            441.6729559300637, 0.705882, 0.0156863, 0.14902]
    basemapLUT.ScalarRangeInitialized = 1.0

    # get opacity transfer function/opacity map for 'basemap'
    basemapPWF = GetOpacityTransferFunction('basemap')
    basemapPWF.Points = [0.0, 0.0, 0.5, 0.0, 441.6729559300637, 1.0, 0.5, 0.0]
    basemapPWF.ScalarRangeInitialized = 1

    # trace defaults for the display properties.
    basemap_2193vtiDisplay.Representation = 'Slice'
    basemap_2193vtiDisplay.ColorArrayName = ['CELLS', 'basemap']
    basemap_2193vtiDisplay.LookupTable = basemapLUT
    basemap_2193vtiDisplay.MapScalars = 0
    basemap_2193vtiDisplay.SelectTCoordArray = 'None'
    basemap_2193vtiDisplay.SelectNormalArray = 'None'
    basemap_2193vtiDisplay.SelectTangentArray = 'None'
    basemap_2193vtiDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    basemap_2193vtiDisplay.SelectOrientationVectors = 'None'
    basemap_2193vtiDisplay.ScaleFactor = 276600.0
    basemap_2193vtiDisplay.SelectScaleArray = 'mask'
    basemap_2193vtiDisplay.GlyphType = 'Arrow'
    basemap_2193vtiDisplay.GlyphTableIndexArray = 'mask'
    basemap_2193vtiDisplay.GaussianRadius = 13830.0
    basemap_2193vtiDisplay.SetScaleArray = [None, '']
    basemap_2193vtiDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    basemap_2193vtiDisplay.OpacityArray = [None, '']
    basemap_2193vtiDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    basemap_2193vtiDisplay.DataAxesGrid = 'GridAxesRepresentation'
    basemap_2193vtiDisplay.PolarAxes = 'PolarAxesRepresentation'
    basemap_2193vtiDisplay.ScalarOpacityUnitDistance = 19311.9664193523
    basemap_2193vtiDisplay.ScalarOpacityFunction = basemapPWF
    basemap_2193vtiDisplay.TransferFunction2D = basemapTF2D
    basemap_2193vtiDisplay.OpacityArrayName = ['CELLS', 'mask']
    basemap_2193vtiDisplay.ColorArray2Name = ['CELLS', 'mask']
    basemap_2193vtiDisplay.IsosurfaceValues = [0.5]
    basemap_2193vtiDisplay.SliceFunction = 'Plane'
    basemap_2193vtiDisplay.SelectInputVectors = [None, '']
    basemap_2193vtiDisplay.WriteLog = ''

    # init the 'Plane' selected for 'SliceFunction'
    basemap_2193vtiDisplay.SliceFunction.Origin = [1575927.8735133181,
                                                   5400949.511356351, -10.0]

    # show data from clip1
    clip1Display = Show(clip1, renderView1, 'UnstructuredGridRepresentation')

    # get 2D transfer function for 'tau_max_disc'
    tau_max_discTF2D = GetTransferFunction2D('tau_max_disc')
    tau_max_discTF2D.ScalarRangeInitialized = 1
    tau_max_discTF2D.Range = [-0.5, 3.5, 0.0, 1.0]

    # get color transfer function/color map for 'tau_max_disc'
    tau_max_discLUT = GetColorTransferFunction('tau_max_disc')
    tau_max_discLUT.TransferFunction2D = tau_max_discTF2D
    tau_max_discLUT.RGBPoints = [-0.5, 1.11641e-07, 0.0, 1.62551e-06,
                                 -0.24902000000000002, 0.0413146, 0.0619808,
                                 0.209857, 0.0019599999999999618, 0.0185557,
                                 0.101341, 0.350684, 0.252942, 0.00486405,
                                 0.149847, 0.461054, 0.503922, 0.0836345,
                                 0.210845, 0.517906, 0.754902, 0.173222,
                                 0.276134, 0.541793, 1.005882, 0.259857,
                                 0.343877, 0.535869, 1.256862, 0.362299,
                                 0.408124, 0.504293, 1.5078431399999999,
                                 0.468266, 0.468276, 0.468257,
                                 1.7588240000000002, 0.582781, 0.527545,
                                 0.374914, 2.009804, 0.691591, 0.585251,
                                 0.274266, 2.2607839999999997, 0.784454,
                                 0.645091, 0.247332, 2.5117640000000008,
                                 0.862299, 0.710383, 0.27518, 2.762746,
                                 0.920863, 0.782923, 0.351563, 3.013726,
                                 0.955792, 0.859699, 0.533541,
                                 3.2647060000000003, 0.976162, 0.93433,
                                 0.780671, 3.5, 1.0, 1.0, 0.999983]
    tau_max_discLUT.ColorSpace = 'Lab'
    tau_max_discLUT.NanColor = [1.0, 0.0, 0.0]
    tau_max_discLUT.NumberOfTableValues = 4
    tau_max_discLUT.ScalarRangeInitialized = 1.0
    tau_max_discLUT.VectorComponent = 1
    tau_max_discLUT.VectorMode = 'Component'

    # get opacity transfer function/opacity map for 'tau_max_disc'
    tau_max_discPWF = GetOpacityTransferFunction('tau_max_disc')
    tau_max_discPWF.Points = [-0.5, 0.0, 0.5, 0.0, 3.5, 1.0, 0.5, 0.0]
    tau_max_discPWF.ScalarRangeInitialized = 1

    # trace defaults for the display properties.
    clip1Display.Representation = 'Surface'
    clip1Display.ColorArrayName = ['CELLS', 'tau_max_disc']
    clip1Display.LookupTable = tau_max_discLUT
    clip1Display.Opacity = 0.82
    clip1Display.SelectTCoordArray = 'None'
    clip1Display.SelectNormalArray = 'None'
    clip1Display.SelectTangentArray = 'None'
    clip1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    clip1Display.SelectOrientationVectors = 'None'
    clip1Display.ScaleFactor = 147400.0
    clip1Display.SelectScaleArray = 'None'
    clip1Display.GlyphType = 'Arrow'
    clip1Display.GlyphTableIndexArray = 'None'
    clip1Display.GaussianRadius = 7370.0
    clip1Display.SetScaleArray = [None, '']
    clip1Display.ScaleTransferFunction = 'PiecewiseFunction'
    clip1Display.OpacityArray = [None, '']
    clip1Display.OpacityTransferFunction = 'PiecewiseFunction'
    clip1Display.DataAxesGrid = 'GridAxesRepresentation'
    clip1Display.PolarAxes = 'PolarAxesRepresentation'
    clip1Display.ScalarOpacityFunction = tau_max_discPWF
    clip1Display.ScalarOpacityUnitDistance = 18060.107275319242
    clip1Display.OpacityArrayName = ['CELLS', 'j2']
    clip1Display.SelectInputVectors = [None, '']
    clip1Display.WriteLog = ''

    # setup the color legend parameters for each legend in this view

    # get color legend/bar for tau_max_discLUT in view renderView1
    tau_max_discLUTColorBar = GetScalarBar(tau_max_discLUT, renderView1)
    tau_max_discLUTColorBar.WindowLocation = 'Any Location'
    tau_max_discLUTColorBar.Position = [0.7845736434108527,
                                        0.056895674300254426]
    tau_max_discLUTColorBar.Title = '$\\gamma_{max}$ Bin'
    tau_max_discLUTColorBar.ComponentTitle = ''
    tau_max_discLUTColorBar.HorizontalTitle = 1
    tau_max_discLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    tau_max_discLUTColorBar.TitleFontSize = 32
    tau_max_discLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    tau_max_discLUTColorBar.LabelFontSize = 20
    tau_max_discLUTColorBar.ScalarBarThickness = 18
    tau_max_discLUTColorBar.ScalarBarLength = 0.22821882951653955
    tau_max_discLUTColorBar.DrawScalarBarOutline = 1
    tau_max_discLUTColorBar.ScalarBarOutlineColor = [0.0, 0.0, 0.0]
    tau_max_discLUTColorBar.LabelFormat = '%.2f'
    tau_max_discLUTColorBar.DrawTickMarks = 0
    tau_max_discLUTColorBar.UseCustomLabels = 1
    tau_max_discLUTColorBar.CustomLabels = [0.0, 1.0, 2.0, 3.0]
    tau_max_discLUTColorBar.AddRangeLabels = 0
    tau_max_discLUTColorBar.RangeLabelFormat = '%.2f'
    tau_max_discLUTColorBar.DrawAnnotations = 0

    # set color bar visibility
    tau_max_discLUTColorBar.Visibility = 1

    # show color legend
    clip1Display.SetScalarBarVisibility(renderView1, True)

    # ----------------------------------------------------------------
    # setup color maps and opacity mapes used in the visualization
    # note: the Get..() functions create a new object, if needed
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # restore active source
    SetActiveSource(clip1)
    # ----------------------------------------------------------------

    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='extracts')
    res = [1290, 1572]
    SaveScreenshot('figure_gammamax.png',
                   renderView1, ImageResolution=[2 * i for i in res])


def makefig_ss():
    # ----------------------------------------------------------------
    # setup views used in the visualization
    # ----------------------------------------------------------------

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # Create a new 'Render View'
    renderView1 = CreateView('RenderView')
    renderView1.ViewSize = [645, 786]
    renderView1.InteractionMode = '2D'
    renderView1.AxesGrid = 'GridAxes3DActor'
    renderView1.OrientationAxesVisibility = 0
    renderView1.CenterOfRotation = [1532215.0607010862, 5444402.662813947, 0.0]
    renderView1.StereoType = 'Crystal Eyes'
    renderView1.CameraPosition = [1567009.4788373362, 5462720.144816798,
                                  4937900.0]
    renderView1.CameraFocalPoint = [1567009.4788373362, 5462720.144816798, 0.0]
    renderView1.CameraFocalDisk = 1.0
    renderView1.CameraParallelScale = 741825.0763008723
    renderView1.BackEnd = 'OSPRay raycaster'
    renderView1.OSPRayMaterialLibrary = materialLibrary1

    SetActiveView(None)

    # ----------------------------------------------------------------
    # setup view layouts
    # ----------------------------------------------------------------

    # create new layout object 'Layout #1'
    layout1 = CreateLayout(name='Layout #1')
    layout1.AssignView(0, renderView1)
    layout1.SetSize(645, 786)

    # ----------------------------------------------------------------
    # restore active view
    SetActiveView(renderView1)
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # setup the data processing pipelines
    # ----------------------------------------------------------------

    # create a new 'XML Image Data Reader'
    basemap_2193vti = XMLImageDataReader(registrationName='basemap_2193.vti',
                                         FileName=[
                                             '/home/pciturri/PycharmProjects/nonpoisson_nz/figures/forecasts_ms/figure12/paraview/basemap_2193.vti'])
    basemap_2193vti.CellArrayStatus = ['basemap', 'mask']
    basemap_2193vti.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    strain_bins_mapvti = XMLImageDataReader(
        registrationName='strain_bins_map.vti', FileName=[
            '/home/pciturri/PycharmProjects/nonpoisson_nz/figures/forecasts_ms/figure12/paraview/strain_bins_map.vti'])
    strain_bins_mapvti.CellArrayStatus = ['j2', 'tau_max', 'ss', 'j2_disc',
                                          'tau_max_disc', 'ss_disc', 'mask']
    strain_bins_mapvti.TimeArray = 'None'

    # create a new 'Clip'
    clip1 = Clip(registrationName='Clip1', Input=strain_bins_mapvti)
    clip1.ClipType = 'Scalar'
    clip1.HyperTreeGridClipper = 'Plane'
    clip1.Scalars = ['CELLS', 'mask']
    clip1.Value = 0.2901383340385938
    clip1.Invert = 0

    # init the 'Plane' selected for 'HyperTreeGridClipper'
    clip1.HyperTreeGridClipper.Origin = [1532215.0607010862, 5444402.662813947,
                                         0.0]

    # ----------------------------------------------------------------
    # setup the visualization in view 'renderView1'
    # ----------------------------------------------------------------

    # show data from basemap_2193vti
    basemap_2193vtiDisplay = Show(basemap_2193vti, renderView1,
                                  'UniformGridRepresentation')

    # get 2D transfer function for 'basemap'
    basemapTF2D = GetTransferFunction2D('basemap')
    basemapTF2D.ScalarRangeInitialized = 1
    basemapTF2D.Range = [0.0, 441.6729559300637, 0.0, 1.0]

    # get color transfer function/color map for 'basemap'
    basemapLUT = GetColorTransferFunction('basemap')
    basemapLUT.TransferFunction2D = basemapTF2D
    basemapLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941,
                            220.83647796503186, 0.865003, 0.865003, 0.865003,
                            441.6729559300637, 0.705882, 0.0156863, 0.14902]
    basemapLUT.ScalarRangeInitialized = 1.0

    # get opacity transfer function/opacity map for 'basemap'
    basemapPWF = GetOpacityTransferFunction('basemap')
    basemapPWF.Points = [0.0, 0.0, 0.5, 0.0, 441.6729559300637, 1.0, 0.5, 0.0]
    basemapPWF.ScalarRangeInitialized = 1

    # trace defaults for the display properties.
    basemap_2193vtiDisplay.Representation = 'Slice'
    basemap_2193vtiDisplay.ColorArrayName = ['CELLS', 'basemap']
    basemap_2193vtiDisplay.LookupTable = basemapLUT
    basemap_2193vtiDisplay.MapScalars = 0
    basemap_2193vtiDisplay.SelectTCoordArray = 'None'
    basemap_2193vtiDisplay.SelectNormalArray = 'None'
    basemap_2193vtiDisplay.SelectTangentArray = 'None'
    basemap_2193vtiDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    basemap_2193vtiDisplay.SelectOrientationVectors = 'None'
    basemap_2193vtiDisplay.ScaleFactor = 276600.0
    basemap_2193vtiDisplay.SelectScaleArray = 'mask'
    basemap_2193vtiDisplay.GlyphType = 'Arrow'
    basemap_2193vtiDisplay.GlyphTableIndexArray = 'mask'
    basemap_2193vtiDisplay.GaussianRadius = 13830.0
    basemap_2193vtiDisplay.SetScaleArray = [None, '']
    basemap_2193vtiDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    basemap_2193vtiDisplay.OpacityArray = [None, '']
    basemap_2193vtiDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    basemap_2193vtiDisplay.DataAxesGrid = 'GridAxesRepresentation'
    basemap_2193vtiDisplay.PolarAxes = 'PolarAxesRepresentation'
    basemap_2193vtiDisplay.ScalarOpacityUnitDistance = 19311.9664193523
    basemap_2193vtiDisplay.ScalarOpacityFunction = basemapPWF
    basemap_2193vtiDisplay.TransferFunction2D = basemapTF2D
    basemap_2193vtiDisplay.OpacityArrayName = ['CELLS', 'mask']
    basemap_2193vtiDisplay.ColorArray2Name = ['CELLS', 'mask']
    basemap_2193vtiDisplay.IsosurfaceValues = [0.5]
    basemap_2193vtiDisplay.SliceFunction = 'Plane'
    basemap_2193vtiDisplay.SelectInputVectors = [None, '']
    basemap_2193vtiDisplay.WriteLog = ''

    # init the 'Plane' selected for 'SliceFunction'
    basemap_2193vtiDisplay.SliceFunction.Origin = [1575927.8735133181,
                                                   5400949.511356351, -10.0]

    # show data from clip1
    clip1Display = Show(clip1, renderView1, 'UnstructuredGridRepresentation')

    # get 2D transfer function for 'ss_disc'
    ss_discTF2D = GetTransferFunction2D('ss_disc')
    ss_discTF2D.ScalarRangeInitialized = 1
    ss_discTF2D.Range = [-0.5, 3.5, 0.0, 1.0]

    # get color transfer function/color map for 'ss_disc'
    ss_discLUT = GetColorTransferFunction('ss_disc')
    ss_discLUT.AutomaticRescaleRangeMode = 'Never'
    ss_discLUT.TransferFunction2D = ss_discTF2D
    ss_discLUT.RGBPoints = [-0.5, 0.001462, 0.000466, 0.013866, -0.484312,
                            0.002267, 0.00127, 0.01857, -0.468628, 0.003299,
                            0.002249, 0.024239, -0.45294, 0.004547, 0.003392,
                            0.030909, -0.437256, 0.006006, 0.004692, 0.038558,
                            -0.421568, 0.007676, 0.006136, 0.046836,
                            -0.40588399999999997, 0.009561, 0.007713, 0.055143,
                            -0.390196, 0.011663, 0.009417, 0.06346, -0.374508,
                            0.013995, 0.011225, 0.071862, -0.35882400000000003,
                            0.016561, 0.013136, 0.080282, -0.343136, 0.019373,
                            0.015133, 0.088767, -0.32745199999999997, 0.022447,
                            0.017199, 0.097327, -0.31176400000000004, 0.025793,
                            0.019331, 0.10593, -0.29608, 0.029432, 0.021503,
                            0.114621, -0.280392, 0.033385, 0.023702, 0.123397,
                            -0.26470399999999994, 0.037668, 0.025921, 0.132232,
                            -0.24902000000000002, 0.042253, 0.028139, 0.141141,
                            -0.23333199999999998, 0.046915, 0.030324, 0.150164,
                            -0.217648, 0.051644, 0.032474, 0.159254,
                            -0.20195999999999997, 0.056449, 0.034569, 0.168414,
                            -0.186276, 0.06134, 0.03659, 0.177642,
                            -0.17058800000000002, 0.066331, 0.038504, 0.186962,
                            -0.15489999999999998, 0.071429, 0.040294, 0.196354,
                            -0.139216, 0.076637, 0.041905, 0.205799,
                            -0.12352800000000003, 0.081962, 0.043328, 0.215289,
                            -0.107844, 0.087411, 0.044556, 0.224813,
                            -0.09215599999999996, 0.09299, 0.045583, 0.234358,
                            -0.07647200000000004, 0.098702, 0.046402, 0.243904,
                            -0.060784000000000005, 0.104551, 0.047008, 0.25343,
                            -0.04509999999999997, 0.110536, 0.047399, 0.262912,
                            -0.029411999999999994, 0.116656, 0.047574,
                            0.272321, -0.013724000000000014, 0.122908,
                            0.047536, 0.281624, 0.0019599999999999618,
                            0.129285, 0.047293, 0.290788, 0.017647999999999997,
                            0.135778, 0.046856, 0.299776, 0.03333200000000003,
                            0.142378, 0.046242, 0.308553, 0.04901999999999995,
                            0.149073, 0.045468, 0.317085, 0.06470399999999998,
                            0.15585, 0.044559, 0.325338, 0.08039200000000002,
                            0.162689, 0.043554, 0.333277, 0.09608000000000005,
                            0.169575, 0.042489, 0.340874, 0.11176399999999997,
                            0.176493, 0.041402, 0.348111, 0.127452, 0.183429,
                            0.040329, 0.354971, 0.14313600000000004, 0.190367,
                            0.039309, 0.361447, 0.15882399999999997, 0.197297,
                            0.0384, 0.367535, 0.174508, 0.204209, 0.037632,
                            0.373238, 0.19019599999999992, 0.211095, 0.03703,
                            0.378563, 0.20588399999999984, 0.217949, 0.036615,
                            0.383522, 0.221568, 0.224763, 0.036405, 0.388129,
                            0.23725600000000002, 0.231538, 0.036405, 0.3924,
                            0.25294000000000005, 0.238273, 0.036621, 0.396353,
                            0.268628, 0.244967, 0.037055, 0.400007, 0.284312,
                            0.25162, 0.037705, 0.403378, 0.30000000000000016,
                            0.258234, 0.038571, 0.406485, 0.3156880000000001,
                            0.26481, 0.039647, 0.409345, 0.331372, 0.271347,
                            0.040922, 0.411976, 0.34706000000000004, 0.27785,
                            0.042353, 0.414392, 0.36274399999999984, 0.284321,
                            0.043933, 0.416608, 0.378432, 0.290763, 0.045644,
                            0.418637, 0.394116, 0.297178, 0.04747, 0.420491,
                            0.40980399999999995, 0.303568, 0.049396, 0.422182,
                            0.42549199999999987, 0.309935, 0.051407, 0.423721,
                            0.441176, 0.316282, 0.05349, 0.425116,
                            0.45686400000000005, 0.32261, 0.055634, 0.426377,
                            0.4725480000000001, 0.328921, 0.057827, 0.427511,
                            0.488236, 0.335217, 0.06006, 0.428524,
                            0.5039199999999999, 0.3415, 0.062325, 0.429425,
                            0.5196080000000001, 0.347771, 0.064616, 0.430217,
                            0.535296, 0.354032, 0.066925, 0.430906, 0.55098,
                            0.360284, 0.069247, 0.431497, 0.566668, 0.366529,
                            0.071579, 0.431994, 0.582352, 0.372768, 0.073915,
                            0.4324, 0.5980399999999999, 0.379001, 0.076253,
                            0.432719, 0.6137239999999999, 0.385228, 0.078591,
                            0.432955, 0.6294120000000001, 0.391453, 0.080927,
                            0.433109, 0.6451, 0.397674, 0.083257, 0.433183,
                            0.660784, 0.403894, 0.08558, 0.433179, 0.676472,
                            0.410113, 0.087896, 0.433098, 0.692156, 0.416331,
                            0.090203, 0.432943, 0.7078439999999999, 0.422549,
                            0.092501, 0.432714, 0.723528, 0.428768, 0.09479,
                            0.432412, 0.7392160000000001, 0.434987, 0.097069,
                            0.432039, 0.7548999999999999, 0.441207, 0.099338,
                            0.431594, 0.770588, 0.447428, 0.101597, 0.43108,
                            0.786276, 0.453651, 0.103848, 0.430498, 0.80196,
                            0.459875, 0.106089, 0.429846, 0.8176479999999999,
                            0.4661, 0.108322, 0.429125, 0.833332, 0.472328,
                            0.110547, 0.428334, 0.8490200000000001, 0.478558,
                            0.112764, 0.427475, 0.8647039999999999, 0.484789,
                            0.114974, 0.426548, 0.8803919999999998, 0.491022,
                            0.117179, 0.425552, 0.8960800000000002, 0.497257,
                            0.119379, 0.424488, 0.911764, 0.503493, 0.121575,
                            0.423356, 0.9274519999999999, 0.50973, 0.123769,
                            0.422156, 0.943136, 0.515967, 0.12596, 0.420887,
                            0.9588239999999999, 0.522206, 0.12815, 0.419549,
                            0.9745079999999999, 0.528444, 0.130341, 0.418142,
                            0.9901960000000001, 0.534683, 0.132534, 0.416667,
                            1.005884, 0.54092, 0.134729, 0.415123, 1.021568,
                            0.547157, 0.136929, 0.413511, 1.037256, 0.553392,
                            0.139134, 0.411829, 1.0529400000000002, 0.559624,
                            0.141346, 0.410078, 1.0686279999999997, 0.565854,
                            0.143567, 0.408258, 1.084312, 0.572081, 0.145797,
                            0.406369, 1.1000000000000003, 0.578304, 0.148039,
                            0.404411, 1.1156879999999998, 0.584521, 0.150294,
                            0.402385, 1.131372, 0.590734, 0.152563, 0.40029,
                            1.14706, 0.59694, 0.154848, 0.398125, 1.162744,
                            0.603139, 0.157151, 0.395891, 1.178432, 0.60933,
                            0.159474, 0.393589, 1.194116, 0.615513, 0.161817,
                            0.391219, 1.209804, 0.621685, 0.164184, 0.388781,
                            1.225492, 0.627847, 0.166575, 0.386276, 1.241176,
                            0.633998, 0.168992, 0.383704, 1.256864, 0.640135,
                            0.171438, 0.381065, 1.2725479999999998, 0.64626,
                            0.173914, 0.378359, 1.2882360000000002, 0.652369,
                            0.176421, 0.375586, 1.30392, 0.658463, 0.178962,
                            0.372748, 1.319608, 0.66454, 0.181539, 0.369846,
                            1.3352960000000003, 0.670599, 0.184153, 0.366879,
                            1.35098, 0.676638, 0.186807, 0.363849, 1.366668,
                            0.682656, 0.189501, 0.360757, 1.382352, 0.688653,
                            0.192239, 0.357603, 1.39804, 0.694627, 0.195021,
                            0.354388, 1.413724, 0.700576, 0.197851, 0.351113,
                            1.429412, 0.7065, 0.200728, 0.347777, 1.4451,
                            0.712396, 0.203656, 0.344383, 1.460784, 0.718264,
                            0.206636, 0.340931, 1.476472, 0.724103, 0.20967,
                            0.337424, 1.4921560000000003, 0.729909, 0.212759,
                            0.333861, 1.507844, 0.735683, 0.215906, 0.330245,
                            1.5235280000000002, 0.741423, 0.219112, 0.326576,
                            1.5392160000000001, 0.747127, 0.222378, 0.322856,
                            1.5549, 0.752794, 0.225706, 0.319085,
                            1.5705879999999999, 0.758422, 0.229097, 0.315266,
                            1.5862759999999998, 0.76401, 0.232554, 0.311399,
                            1.60196, 0.769556, 0.236077, 0.307485, 1.617648,
                            0.775059, 0.239667, 0.303526, 1.6333319999999998,
                            0.780517, 0.243327, 0.299523, 1.6490200000000002,
                            0.785929, 0.247056, 0.295477, 1.664704, 0.791293,
                            0.250856, 0.29139, 1.6803919999999999, 0.796607,
                            0.254728, 0.287264, 1.6960799999999998, 0.801871,
                            0.258674, 0.283099, 1.711764, 0.807082, 0.262692,
                            0.278898, 1.727452, 0.812239, 0.266786, 0.274661,
                            1.7431359999999998, 0.817341, 0.270954, 0.27039,
                            1.7588240000000002, 0.822386, 0.275197, 0.266085,
                            1.774508, 0.827372, 0.279517, 0.26175, 1.790196,
                            0.832299, 0.283913, 0.257383, 1.8058839999999998,
                            0.837165, 0.288385, 0.252988, 1.821568, 0.841969,
                            0.292933, 0.248564, 1.837256, 0.846709, 0.297559,
                            0.244113, 1.8529399999999998, 0.851384, 0.30226,
                            0.239636, 1.8686280000000002, 0.855992, 0.307038,
                            0.235133, 1.884312, 0.860533, 0.311892, 0.230606,
                            1.9, 0.865006, 0.316822, 0.226055,
                            1.9156879999999998, 0.869409, 0.321827, 0.221482,
                            1.931372, 0.873741, 0.326906, 0.216886, 1.94706,
                            0.878001, 0.33206, 0.212268, 1.9627439999999998,
                            0.882188, 0.337287, 0.207628, 1.9784320000000002,
                            0.886302, 0.342586, 0.202968, 1.994116, 0.890341,
                            0.347957, 0.198286, 2.009804, 0.894305, 0.353399,
                            0.193584, 2.025492, 0.898192, 0.358911, 0.18886,
                            2.041176, 0.902003, 0.364492, 0.184116, 2.056864,
                            0.905735, 0.37014, 0.17935, 2.072548, 0.90939,
                            0.375856, 0.174563, 2.088236, 0.912966, 0.381636,
                            0.169755, 2.10392, 0.916462, 0.387481, 0.164924,
                            2.119608, 0.919879, 0.393389, 0.16007, 2.135296,
                            0.923215, 0.399359, 0.155193, 2.15098, 0.92647,
                            0.405389, 0.150292, 2.166668, 0.929644, 0.411479,
                            0.145367, 2.182352, 0.932737, 0.417627, 0.140417,
                            2.19804, 0.935747, 0.423831, 0.13544,
                            2.2137240000000005, 0.938675, 0.430091, 0.130438,
                            2.229412, 0.941521, 0.436405, 0.125409,
                            2.2450999999999994, 0.944285, 0.442772, 0.120354,
                            2.2607839999999997, 0.946965, 0.449191, 0.115272,
                            2.276472, 0.949562, 0.45566, 0.110164, 2.292156,
                            0.952075, 0.462178, 0.105031, 2.307844, 0.954506,
                            0.468744, 0.099874, 2.323528, 0.956852, 0.475356,
                            0.094695, 2.339216, 0.959114, 0.482014, 0.089499,
                            2.3549, 0.961293, 0.488716, 0.084289, 2.370588,
                            0.963387, 0.495462, 0.079073, 2.386276, 0.965397,
                            0.502249, 0.073859, 2.4019600000000003, 0.967322,
                            0.509078, 0.068659, 2.417648, 0.969163, 0.515946,
                            0.063488, 2.433332, 0.970919, 0.522853, 0.058367,
                            2.4490199999999995, 0.97259, 0.529798, 0.053324,
                            2.464704, 0.974176, 0.53678, 0.048392, 2.480392,
                            0.975677, 0.543798, 0.043618, 2.4960800000000005,
                            0.977092, 0.55085, 0.03905, 2.511764, 0.978422,
                            0.557937, 0.034931, 2.527452, 0.979666, 0.565057,
                            0.031409, 2.543136, 0.980824, 0.572209, 0.028508,
                            2.558824, 0.981895, 0.579392, 0.02625, 2.574508,
                            0.982881, 0.586606, 0.024661, 2.590196, 0.983779,
                            0.593849, 0.02377, 2.605884, 0.984591, 0.601122,
                            0.023606, 2.621568, 0.985315, 0.608422, 0.024202,
                            2.6372559999999994, 0.985952, 0.61575, 0.025592,
                            2.6529399999999996, 0.986502, 0.623105, 0.027814,
                            2.668628, 0.986964, 0.630485, 0.030908, 2.684312,
                            0.987337, 0.63789, 0.034916, 2.7000000000000006,
                            0.987622, 0.64532, 0.039886, 2.715688, 0.987819,
                            0.652773, 0.045581, 2.731372, 0.987926, 0.66025,
                            0.05175, 2.74706, 0.987945, 0.667748, 0.058329,
                            2.762744, 0.987874, 0.675267, 0.065257, 2.778432,
                            0.987714, 0.682807, 0.072489, 2.7941160000000003,
                            0.987464, 0.690366, 0.07999, 2.809804, 0.987124,
                            0.697944, 0.087731, 2.825492, 0.986694, 0.70554,
                            0.095694, 2.8411759999999995, 0.986175, 0.713153,
                            0.103863, 2.856864, 0.985566, 0.720782, 0.112229,
                            2.872548, 0.984865, 0.728427, 0.120785,
                            2.8882360000000005, 0.984075, 0.736087, 0.129527,
                            2.90392, 0.983196, 0.743758, 0.138453, 2.919608,
                            0.982228, 0.751442, 0.147565, 2.9352959999999997,
                            0.981173, 0.759135, 0.156863, 2.95098, 0.980032,
                            0.766837, 0.166353, 2.966668, 0.978806, 0.774545,
                            0.176037, 2.982352, 0.977497, 0.782258, 0.185923,
                            2.99804, 0.976108, 0.789974, 0.196018, 3.013724,
                            0.974638, 0.797692, 0.206332, 3.029412, 0.973088,
                            0.805409, 0.216877, 3.0451, 0.971468, 0.813122,
                            0.227658, 3.060784, 0.969783, 0.820825, 0.238686,
                            3.0764720000000003, 0.968041, 0.828515, 0.249972,
                            3.0921560000000006, 0.966243, 0.836191, 0.261534,
                            3.107844, 0.964394, 0.843848, 0.273391, 3.123528,
                            0.962517, 0.851476, 0.285546, 3.139216, 0.960626,
                            0.859069, 0.29801, 3.1549, 0.95872, 0.866624,
                            0.31082, 3.170588, 0.956834, 0.874129, 0.323974,
                            3.186276, 0.954997, 0.881569, 0.337475, 3.20196,
                            0.953215, 0.888942, 0.351369, 3.217648, 0.951546,
                            0.896226, 0.365627, 3.2333319999999994, 0.950018,
                            0.903409, 0.380271, 3.24902, 0.948683, 0.910473,
                            0.395289, 3.264704, 0.947594, 0.917399, 0.410665,
                            3.2803920000000004, 0.946809, 0.924168, 0.426373,
                            3.29608, 0.946392, 0.930761, 0.442367, 3.311764,
                            0.946403, 0.937159, 0.458592, 3.3274519999999996,
                            0.946903, 0.943348, 0.47497, 3.343136, 0.947937,
                            0.949318, 0.491426, 3.358824, 0.949545, 0.955063,
                            0.50786, 3.374508, 0.95174, 0.960587, 0.524203,
                            3.390196, 0.954529, 0.965896, 0.540361, 3.405884,
                            0.957896, 0.971003, 0.556275, 3.421568, 0.961812,
                            0.975924, 0.571925, 3.437256, 0.966249, 0.980678,
                            0.587206, 3.45294, 0.971162, 0.985282, 0.602154,
                            3.4686280000000003, 0.976511, 0.989753, 0.61676,
                            3.4843120000000005, 0.982257, 0.994109, 0.631017,
                            3.5, 0.988362, 0.998364, 0.644924]
    ss_discLUT.NanColor = [0.0, 1.0, 0.0]
    ss_discLUT.NumberOfTableValues = 4
    ss_discLUT.ScalarRangeInitialized = 1.0
    ss_discLUT.VectorComponent = 1
    ss_discLUT.VectorMode = 'Component'

    # get opacity transfer function/opacity map for 'ss_disc'
    ss_discPWF = GetOpacityTransferFunction('ss_disc')
    ss_discPWF.Points = [-0.5, 0.0, 0.5, 0.0, 3.5, 1.0, 0.5, 0.0]
    ss_discPWF.ScalarRangeInitialized = 1

    # trace defaults for the display properties.
    clip1Display.Representation = 'Surface'
    clip1Display.ColorArrayName = ['CELLS', 'ss_disc']
    clip1Display.LookupTable = ss_discLUT
    clip1Display.Opacity = 0.82
    clip1Display.SelectTCoordArray = 'None'
    clip1Display.SelectNormalArray = 'None'
    clip1Display.SelectTangentArray = 'None'
    clip1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    clip1Display.SelectOrientationVectors = 'None'
    clip1Display.ScaleFactor = 147400.0
    clip1Display.SelectScaleArray = 'None'
    clip1Display.GlyphType = 'Arrow'
    clip1Display.GlyphTableIndexArray = 'None'
    clip1Display.GaussianRadius = 7370.0
    clip1Display.SetScaleArray = [None, '']
    clip1Display.ScaleTransferFunction = 'PiecewiseFunction'
    clip1Display.OpacityArray = [None, '']
    clip1Display.OpacityTransferFunction = 'PiecewiseFunction'
    clip1Display.DataAxesGrid = 'GridAxesRepresentation'
    clip1Display.PolarAxes = 'PolarAxesRepresentation'
    clip1Display.ScalarOpacityFunction = ss_discPWF
    clip1Display.ScalarOpacityUnitDistance = 18060.107275319242
    clip1Display.OpacityArrayName = ['CELLS', 'j2']
    clip1Display.SelectInputVectors = [None, '']
    clip1Display.WriteLog = ''

    # setup the color legend parameters for each legend in this view

    # get color legend/bar for ss_discLUT in view renderView1
    ss_discLUTColorBar = GetScalarBar(ss_discLUT, renderView1)
    ss_discLUTColorBar.WindowLocation = 'Any Location'
    ss_discLUTColorBar.Position = [0.7845736434108527, 0.056895674300254426]
    ss_discLUTColorBar.Title = '$SS$ Bin'
    ss_discLUTColorBar.ComponentTitle = ''
    ss_discLUTColorBar.HorizontalTitle = 1
    ss_discLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    ss_discLUTColorBar.TitleFontSize = 32
    ss_discLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    ss_discLUTColorBar.LabelFontSize = 20
    ss_discLUTColorBar.ScalarBarThickness = 18
    ss_discLUTColorBar.ScalarBarLength = 0.2282
    ss_discLUTColorBar.DrawScalarBarOutline = 1
    ss_discLUTColorBar.ScalarBarOutlineColor = [0.0, 0.0, 0.0]
    ss_discLUTColorBar.LabelFormat = '%.2f'
    ss_discLUTColorBar.DrawTickMarks = 0
    ss_discLUTColorBar.UseCustomLabels = 1
    ss_discLUTColorBar.CustomLabels = [0.0, 1.0, 2.0, 3.0]
    ss_discLUTColorBar.AddRangeLabels = 0
    ss_discLUTColorBar.RangeLabelFormat = '%.2f'

    # set color bar visibility
    ss_discLUTColorBar.Visibility = 1

    # show color legend
    clip1Display.SetScalarBarVisibility(renderView1, True)

    # ----------------------------------------------------------------
    # setup color maps and opacity mapes used in the visualization
    # note: the Get..() functions create a new object, if needed
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # restore active source
    SetActiveSource(clip1)
    # ----------------------------------------------------------------

    SaveExtracts(ExtractsOutputDirectory='extracts')

    res = [1290, 1572]
    SaveScreenshot('figure_ss.png',
                   renderView1, ImageResolution=[2 * i for i in res])


def make_joint_figure():
    binmaps = [plt.imread(f"figure_{i}.png") for i in ['j2', 'ss', 'gammamax']]
    letters = ['a)', 'b)', 'c)']
    fig, axs = plt.subplots(1, 3,
                            figsize=(12, 5),
                            gridspec_kw={'wspace': -0.02},
                            constrained_layout=True)

    for i, j in enumerate(binmaps):
        axs[i].imshow(j)
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)
        axs[i].text(100, 250, letters[i], fontsize=16,
                    **{'fontname': 'Ubuntu'})

    plt.savefig("strain_measures_bin.jpg", dpi=400, bbox_inches='tight')
    plt.savefig("fig13.jpg", dpi=1200, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    makefig_j2()
    makefig_ss()
    makefig_tau_max()

    make_joint_figure()

# state file generated using paraview version 5.11.1
import paraview

paraview.compatibility.major = 5
paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()
import matplotlib.pyplot as plt


def makefig_m(size_factor=1):
    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # Create a new 'Render View'
    renderView1 = CreateView('RenderView')
    renderView1.ViewSize = [558, 786]
    renderView1.InteractionMode = '2D'
    renderView1.AxesGrid = 'GridAxes3DActor'
    renderView1.OrientationAxesVisibility = 0
    renderView1.CenterOfRotation = [1590196.015518637, 5465886.7749801995, 10.0]
    renderView1.StereoType = 'Crystal Eyes'
    renderView1.CameraPosition = [1592444.049301613, 5466129.775278663, 5175760.0]
    renderView1.CameraFocalPoint = [1592444.049301613, 5466129.775278663, 10.0]
    renderView1.CameraFocalDisk = 1.0
    renderView1.CameraParallelScale = 785071.8935111373
    renderView1.BackEnd = 'OSPRay raycaster'
    renderView1.OSPRayMaterialLibrary = materialLibrary1

    SetActiveView(None)

    # ----------------------------------------------------------------
    # setup view layouts
    # ----------------------------------------------------------------

    # create new layout object 'Layout #1'
    layout1 = CreateLayout(name='Layout #1')
    layout1.AssignView(0, renderView1)
    layout1.SetSize(558, 786)

    # ----------------------------------------------------------------
    # restore active view
    SetActiveView(renderView1)
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # setup the data processing pipelines
    # ----------------------------------------------------------------

    # create a new 'XML Image Data Reader'
    pua_4vti = XMLImageDataReader(
        registrationName='pua_4.vti',
         FileName=['paraview/pua_4.vti'])
    pua_4vti.CellArrayStatus = ['poly', 'bin', 'eventcount', 'rate_learning', 'bval', 'lmmin', 'mmin', 'mmax', 'model', 'params', 'rates_bin', 'rate', 'mask']
    pua_4vti.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    pua_5vti = XMLImageDataReader(
        registrationName='pua_5.vti',
        FileName=['paraview/pua_5.vti'])
    pua_5vti.CellArrayStatus = ['poly', 'bin', 'eventcount', 'rate_learning', 'bval', 'lmmin', 'mmin', 'mmax', 'model', 'params', 'rates_bin', 'rate', 'mask']
    pua_5vti.TimeArray = 'None'

    # create a new 'XML MultiBlock Data Reader'
    regionvtm = XMLMultiBlockDataReader(
        registrationName='region.vtm',
        FileName=['paraview/region.vtm'])
    regionvtm.CellArrayStatus = ['id']
    regionvtm.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    basemap_2193vti = XMLImageDataReader(
        registrationName='basemap_2193.vti',
        FileName=['paraview/basemap_2193.vti'])
    basemap_2193vti.CellArrayStatus = ['basemap', 'mask']
    basemap_2193vti.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    hybridvti = XMLImageDataReader(
        registrationName='hybrid.vti',
        FileName=['paraview/hybrid.vti'])
    hybridvti.CellArrayStatus = ['poly', 'bin', 'eventcount', 'rate_learning', 'bval', 'lmmin', 'mmin', 'mmax', 'model', 'params', 'rates_bin', 'rate', 'mask']
    hybridvti.TimeArray = 'None'

    # create a new 'XML MultiBlock Data Reader'
    coastlinevtm = XMLMultiBlockDataReader(
        registrationName='coastline.vtm',
        FileName=['paraview/coastline.vtm'])
    coastlinevtm.CellArrayStatus = ['t50_fid', 'elevation']
    coastlinevtm.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    pua_3vti = XMLImageDataReader(
        registrationName='pua_3.vti',
        FileName=['paraview/pua_3.vti'])
    pua_3vti.CellArrayStatus = ['poly', 'bin', 'eventcount', 'rate_learning', 'bval', 'lmmin', 'mmin', 'mmax', 'model', 'params', 'rates_bin', 'rate', 'mask']
    pua_3vti.TimeArray = 'None'

    # ----------------------------------------------------------------
    # setup the visualization in view 'renderView1'
    # ----------------------------------------------------------------

    # show data from basemap_2193vti
    basemap_2193vtiDisplay = Show(basemap_2193vti, renderView1, 'UniformGridRepresentation')

    # get 2D transfer function for 'basemap'
    basemapTF2D = GetTransferFunction2D('basemap')
    basemapTF2D.ScalarRangeInitialized = 1
    basemapTF2D.Range = [0.0, 441.6729559300637, 0.0, 1.0]

    # get color transfer function/color map for 'basemap'
    basemapLUT = GetColorTransferFunction('basemap')
    basemapLUT.TransferFunction2D = basemapTF2D
    basemapLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 220.83647796503186, 0.865003, 0.865003, 0.865003, 441.6729559300637, 0.705882, 0.0156863, 0.14902]
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
    basemap_2193vtiDisplay.SliceFunction.Origin = [1575927.8735133181, 5400949.511356351, -10.0]

    # show data from coastlinevtm
    coastlinevtmDisplay = Show(coastlinevtm, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    coastlinevtmDisplay.Representation = 'Surface'
    coastlinevtmDisplay.AmbientColor = [0.0, 0.0, 0.0]
    coastlinevtmDisplay.ColorArrayName = ['POINTS', '']
    coastlinevtmDisplay.DiffuseColor = [0.0, 0.0, 0.0]
    coastlinevtmDisplay.SelectTCoordArray = 'None'
    coastlinevtmDisplay.SelectNormalArray = 'None'
    coastlinevtmDisplay.SelectTangentArray = 'None'
    coastlinevtmDisplay.LineWidth = size_factor
    coastlinevtmDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    coastlinevtmDisplay.SelectOrientationVectors = 'None'
    coastlinevtmDisplay.ScaleFactor = 144610.3415327726
    coastlinevtmDisplay.SelectScaleArray = 'None'
    coastlinevtmDisplay.GlyphType = 'Arrow'
    coastlinevtmDisplay.GlyphTableIndexArray = 'None'
    coastlinevtmDisplay.GaussianRadius = 7230.51707663863
    coastlinevtmDisplay.SetScaleArray = [None, '']
    coastlinevtmDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    coastlinevtmDisplay.OpacityArray = [None, '']
    coastlinevtmDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    coastlinevtmDisplay.DataAxesGrid = 'GridAxesRepresentation'
    coastlinevtmDisplay.PolarAxes = 'PolarAxesRepresentation'
    coastlinevtmDisplay.SelectInputVectors = [None, '']
    coastlinevtmDisplay.WriteLog = ''

    # show data from regionvtm
    regionvtmDisplay = Show(regionvtm, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    regionvtmDisplay.Representation = 'Surface'
    regionvtmDisplay.AmbientColor = [0.0, 0.0, 0.0]
    regionvtmDisplay.ColorArrayName = [None, '']
    regionvtmDisplay.DiffuseColor = [0.0, 0.0, 0.0]
    regionvtmDisplay.SelectTCoordArray = 'None'
    regionvtmDisplay.SelectNormalArray = 'None'
    regionvtmDisplay.SelectTangentArray = 'None'
    regionvtmDisplay.LineWidth = size_factor
    regionvtmDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    regionvtmDisplay.SelectOrientationVectors = 'None'
    regionvtmDisplay.ScaleFactor = 153998.45800642233
    regionvtmDisplay.SelectScaleArray = 'None'
    regionvtmDisplay.GlyphType = 'Arrow'
    regionvtmDisplay.GlyphTableIndexArray = 'None'
    regionvtmDisplay.GaussianRadius = 7699.922900321116
    regionvtmDisplay.SetScaleArray = [None, '']
    regionvtmDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    regionvtmDisplay.OpacityArray = [None, '']
    regionvtmDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    regionvtmDisplay.DataAxesGrid = 'GridAxesRepresentation'
    regionvtmDisplay.PolarAxes = 'PolarAxesRepresentation'
    regionvtmDisplay.SelectInputVectors = [None, '']
    regionvtmDisplay.WriteLog = ''

    # show data from hybridvti
    hybridvtiDisplay = Show(hybridvti, renderView1, 'UniformGridRepresentation')

    # get 2D transfer function for 'rate'
    rateTF2D = GetTransferFunction2D('rate')
    rateTF2D.ScalarRangeInitialized = 1
    rateTF2D.Range = [4.9999999999999996e-06, 0.003000000000000001, 7.564979114249576e-07, 0.0007075337856316542]

    # get color transfer function/color map for 'rate'
    rateLUT = GetColorTransferFunction('rate')
    rateLUT.AutomaticRescaleRangeMode = 'Never'
    rateLUT.TransferFunction2D = rateTF2D
    rateLUT.RGBPoints = [4.9999999999999996e-06, 0.02, 0.3813, 0.9981, 5.822593385447654e-06, 0.02000006, 0.424267768, 0.96906969, 6.780518746451757e-06, 0.02, 0.467233763, 0.940033043, 7.896040720598767e-06, 0.02, 0.5102, 0.911, 9.195086894196724e-06, 0.02000006, 0.546401494, 0.872669438, 1.0707850425753277e-05, 0.02, 0.582600362, 0.83433295, 1.2469491812270778e-05, 0.02, 0.6188, 0.796, 1.4520956109204277e-05, 0.02000006, 0.652535156, 0.749802434, 1.690992459836571e-05, 0.02, 0.686267004, 0.703599538, 1.9691923022972557e-05, 0.02, 0.72, 0.6574, 2.293161214806088e-05, 0.02000006, 0.757035456, 0.603735359, 2.6704290642190078e-05, 0.02, 0.794067037, 0.55006613, 3.1097645211257534e-05, 0.02, 0.8311, 0.4964, 3.621378866201322e-05, 0.021354336738172372, 0.8645368555261631, 0.4285579460761159, 4.217163326508741e-05, 0.023312914349117714, 0.897999359924484, 0.36073871343115577, 4.910965458056455e-05, 0.015976108242848862, 0.9310479513349017, 0.2925631815088092, 5.718910998448275e-05, 0.27421074700988196, 0.952562960995083, 0.15356836602739213, 6.659778670305754e-05, 0.4933546281681699, 0.9619038625309482, 0.11119493614749336, 7.755436646853534e-05, 0.6439, 0.9773, 0.0469, 9.031350824245546e-05, 0.762401813, 0.984669591, 0.034600153, 0.0001051717671418187, 0.880901185, 0.992033407, 0.022299877, 0.00012247448713915892, 0.9995285432627147, 0.9995193706781492, 0.0134884641450013, 0.00014262382774051213, 0.999402998, 0.955036376, 0.079066628, 0.00016608811120182637, 0.9994, 0.910666223, 0.148134024, 0.0001934127075370494, 0.9994, 0.8663, 0.2172, 0.00022523271031334936, 0.999269665, 0.818035981, 0.217200652, 0.0002622876978513912, 0.999133332, 0.769766184, 0.2172, 0.00030543892291876073, 0.999, 0.7215, 0.2172, 0.0003556893304490061, 0.99913633, 0.673435546, 0.217200652, 0.0004142068685493377, 0.999266668, 0.625366186, 0.2172, 0.000482351634604472, 0.9994, 0.5773, 0.2172, 0.0005617074874215727, 0.999402998, 0.521068455, 0.217200652, 0.0006541188601634542, 0.9994, 0.464832771, 0.2172, 0.0007617336296968577, 0.9994, 0.4086, 0.2172, 0.0008870530387491915, 0.9947599917687346, 0.33177297300202935, 0.2112309638520206, 0.001032989831192457, 0.9867129505479589, 0.2595183410914934, 0.19012239549291934, 0.001202935951667177, 0.9912458875646419, 0.14799417507952672, 0.21078892136920357, 0.001400841383058897, 0.949903037, 0.116867171, 0.252900603, 0.0016313059542120158, 0.903199533, 0.078432949, 0.291800389, 0.001899686251727252, 0.8565, 0.04, 0.3307, 0.002212220120746587, 0.798902627, 0.04333345, 0.358434298, 0.0025761716484426584, 0.741299424, 0.0466667, 0.386166944, 0.003000000000000001, 0.6837, 0.05, 0.4139]
    rateLUT.UseLogScale = 1
    rateLUT.UseOpacityControlPointsFreehandDrawing = 1
    rateLUT.ColorSpace = 'RGB'
    rateLUT.NanColor = [1.0, 0.0, 0.0]
    rateLUT.NanOpacity = 0.0
    rateLUT.ScalarRangeInitialized = 1.0

    # get opacity transfer function/opacity map for 'rate'
    ratePWF = GetOpacityTransferFunction('rate')
    ratePWF.Points = [4.9999999999999996e-06, 0.0, 0.5, 0.0, 0.0004178252980862393, 1.0, 0.5, 0.0, 0.0020051646169275193, 1.0, 0.5, 0.0, 0.003000000000000001, 1.0, 0.5, 0.0]
    ratePWF.UseLogScale = 1
    ratePWF.ScalarRangeInitialized = 1

    # trace defaults for the display properties.
    hybridvtiDisplay.Representation = 'Slice'
    hybridvtiDisplay.ColorArrayName = ['CELLS', 'rate']
    hybridvtiDisplay.LookupTable = rateLUT
    hybridvtiDisplay.LineWidth = 2.0
    hybridvtiDisplay.SelectTCoordArray = 'None'
    hybridvtiDisplay.SelectNormalArray = 'None'
    hybridvtiDisplay.SelectTangentArray = 'None'
    hybridvtiDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    hybridvtiDisplay.SelectOrientationVectors = 'None'
    hybridvtiDisplay.ScaleFactor = 154350.0
    hybridvtiDisplay.SelectScaleArray = 'None'
    hybridvtiDisplay.GlyphType = 'Arrow'
    hybridvtiDisplay.GlyphTableIndexArray = 'None'
    hybridvtiDisplay.GaussianRadius = 7717.5
    hybridvtiDisplay.SetScaleArray = [None, '']
    hybridvtiDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    hybridvtiDisplay.OpacityArray = [None, '']
    hybridvtiDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    hybridvtiDisplay.DataAxesGrid = 'GridAxesRepresentation'
    hybridvtiDisplay.PolarAxes = 'PolarAxesRepresentation'
    hybridvtiDisplay.ScalarOpacityUnitDistance = 20818.636043244984
    hybridvtiDisplay.ScalarOpacityFunction = ratePWF
    hybridvtiDisplay.TransferFunction2D = rateTF2D
    hybridvtiDisplay.OpacityArrayName = ['CELLS', 'bin']
    hybridvtiDisplay.ColorArray2Name = ['CELLS', 'bin']
    hybridvtiDisplay.SliceFunction = 'Plane'
    hybridvtiDisplay.SelectInputVectors = [None, '']
    hybridvtiDisplay.WriteLog = ''

    # init the 'Plane' selected for 'SliceFunction'
    hybridvtiDisplay.SliceFunction.Origin = [1589946.015518637, 5470136.7749801995, 10.0]

    # setup the color legend parameters for each legend in this view

    # get color legend/bar for rateLUT in view renderView1
    rateLUTColorBar = GetScalarBar(rateLUT, renderView1)
    rateLUTColorBar.WindowLocation = 'Any Location'
    rateLUTColorBar.Position = [0.08282111399957778, 0.5012722646310435]
    rateLUTColorBar.Title = '$\\frac{\\mu}{N_{tot}}$'
    rateLUTColorBar.ComponentTitle = ''
    rateLUTColorBar.HorizontalTitle = 1
    rateLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    rateLUTColorBar.TitleFontSize = 40
    rateLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    rateLUTColorBar.LabelFontSize = 20
    rateLUTColorBar.ScalarBarLength = 0.32999999999999996
    rateLUTColorBar.AutomaticLabelFormat = 0
    rateLUTColorBar.LabelFormat = '%.0e'
    rateLUTColorBar.RangeLabelFormat = '%.0e'

    # set color bar visibility
    rateLUTColorBar.Visibility = 1

    # show color legend
    hybridvtiDisplay.SetScalarBarVisibility(renderView1, True)

    # ----------------------------------------------------------------
    # setup color maps and opacity mapes used in the visualization
    # note: the Get..() functions create a new object, if needed
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # restore active source
    SetActiveSource(hybridvti)
    # ----------------------------------------------------------------

    SaveExtracts(ExtractsOutputDirectory='extracts')

    SaveScreenshot('figure_m.png',
                   renderView1,
                   ImageResolution=[size_factor * i for i in renderView1.ViewSize])


def makefig_p3(size_factor=1):
    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # Create a new 'Render View'
    renderView1 = CreateView('RenderView')
    renderView1.ViewSize = [558, 786]
    renderView1.InteractionMode = '2D'
    renderView1.AxesGrid = 'GridAxes3DActor'
    renderView1.OrientationAxesVisibility = 0
    renderView1.CenterOfRotation = [1590196.015518637, 5465886.7749801995, 10.0]
    renderView1.StereoType = 'Crystal Eyes'
    renderView1.CameraPosition = [1592444.049301613, 5466129.775278663, 5175760.0]
    renderView1.CameraFocalPoint = [1592444.049301613, 5466129.775278663, 10.0]
    renderView1.CameraFocalDisk = 1.0
    renderView1.CameraParallelScale = 785071.8935111373
    renderView1.BackEnd = 'OSPRay raycaster'
    renderView1.OSPRayMaterialLibrary = materialLibrary1

    SetActiveView(None)

    # ----------------------------------------------------------------
    # setup view layouts
    # ----------------------------------------------------------------

    # create new layout object 'Layout #1'
    layout1 = CreateLayout(name='Layout #1')
    layout1.AssignView(0, renderView1)
    layout1.SetSize(558, 786)

    # ----------------------------------------------------------------
    # restore active view
    SetActiveView(renderView1)
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # setup the data processing pipelines
    # ----------------------------------------------------------------

    # create a new 'XML Image Data Reader'
    pua_4vti = XMLImageDataReader(
        registrationName='pua_4.vti',
        FileName=['paraview/pua_4.vti'])
    pua_4vti.CellArrayStatus = ['poly', 'bin', 'eventcount', 'rate_learning', 'bval', 'lmmin', 'mmin', 'mmax', 'model', 'params', 'rates_bin', 'rate', 'mask']
    pua_4vti.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    pua_5vti = XMLImageDataReader(
        registrationName='pua_5.vti',
        FileName=['paraview/pua_5.vti'])
    pua_5vti.CellArrayStatus = ['poly', 'bin', 'eventcount', 'rate_learning', 'bval', 'lmmin', 'mmin', 'mmax', 'model', 'params', 'rates_bin', 'rate', 'mask']
    pua_5vti.TimeArray = 'None'

    # create a new 'XML MultiBlock Data Reader'
    regionvtm = XMLMultiBlockDataReader(
        registrationName='region.vtm',
        FileName=['paraview/region.vtm'])
    regionvtm.CellArrayStatus = ['id']
    regionvtm.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    basemap_2193vti = XMLImageDataReader(
        registrationName='basemap_2193.vti',
        FileName=['paraview/basemap_2193.vti'])
    basemap_2193vti.CellArrayStatus = ['basemap', 'mask']
    basemap_2193vti.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    hybridvti = XMLImageDataReader(
        registrationName='hybrid.vti',
        FileName=['paraview/hybrid.vti'])
    hybridvti.CellArrayStatus = ['poly', 'bin', 'eventcount', 'rate_learning', 'bval', 'lmmin', 'mmin', 'mmax', 'model', 'params', 'rates_bin', 'rate', 'mask']
    hybridvti.TimeArray = 'None'

    # create a new 'XML MultiBlock Data Reader'
    coastlinevtm = XMLMultiBlockDataReader(
        registrationName='coastline.vtm',
        FileName=['paraview/coastline.vtm'])
    coastlinevtm.CellArrayStatus = ['t50_fid', 'elevation']
    coastlinevtm.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    pua_3vti = XMLImageDataReader(
        registrationName='pua_3.vti',
        FileName=['paraview/pua_3.vti'])
    pua_3vti.CellArrayStatus = ['poly', 'bin', 'eventcount', 'rate_learning', 'bval', 'lmmin', 'mmin', 'mmax', 'model', 'params', 'rates_bin', 'rate', 'mask']
    pua_3vti.TimeArray = 'None'

    # ----------------------------------------------------------------
    # setup the visualization in view 'renderView1'
    # ----------------------------------------------------------------

    # show data from basemap_2193vti
    basemap_2193vtiDisplay = Show(basemap_2193vti, renderView1, 'UniformGridRepresentation')

    # get 2D transfer function for 'basemap'
    basemapTF2D = GetTransferFunction2D('basemap')
    basemapTF2D.ScalarRangeInitialized = 1
    basemapTF2D.Range = [0.0, 441.6729559300637, 0.0, 1.0]

    # get color transfer function/color map for 'basemap'
    basemapLUT = GetColorTransferFunction('basemap')
    basemapLUT.TransferFunction2D = basemapTF2D
    basemapLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 220.83647796503186, 0.865003, 0.865003, 0.865003, 441.6729559300637, 0.705882, 0.0156863, 0.14902]
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
    basemap_2193vtiDisplay.SliceFunction.Origin = [1575927.8735133181, 5400949.511356351, -10.0]

    # show data from coastlinevtm
    coastlinevtmDisplay = Show(coastlinevtm, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    coastlinevtmDisplay.Representation = 'Surface'
    coastlinevtmDisplay.AmbientColor = [0.0, 0.0, 0.0]
    coastlinevtmDisplay.ColorArrayName = ['POINTS', '']
    coastlinevtmDisplay.DiffuseColor = [0.0, 0.0, 0.0]
    coastlinevtmDisplay.SelectTCoordArray = 'None'
    coastlinevtmDisplay.SelectNormalArray = 'None'
    coastlinevtmDisplay.SelectTangentArray = 'None'
    coastlinevtmDisplay.LineWidth = size_factor
    coastlinevtmDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    coastlinevtmDisplay.SelectOrientationVectors = 'None'
    coastlinevtmDisplay.ScaleFactor = 144610.3415327726
    coastlinevtmDisplay.SelectScaleArray = 'None'
    coastlinevtmDisplay.GlyphType = 'Arrow'
    coastlinevtmDisplay.GlyphTableIndexArray = 'None'
    coastlinevtmDisplay.GaussianRadius = 7230.51707663863
    coastlinevtmDisplay.SetScaleArray = [None, '']
    coastlinevtmDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    coastlinevtmDisplay.OpacityArray = [None, '']
    coastlinevtmDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    coastlinevtmDisplay.DataAxesGrid = 'GridAxesRepresentation'
    coastlinevtmDisplay.PolarAxes = 'PolarAxesRepresentation'
    coastlinevtmDisplay.SelectInputVectors = [None, '']
    coastlinevtmDisplay.WriteLog = ''

    # show data from regionvtm
    regionvtmDisplay = Show(regionvtm, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    regionvtmDisplay.Representation = 'Surface'
    regionvtmDisplay.AmbientColor = [0.0, 0.0, 0.0]
    regionvtmDisplay.ColorArrayName = [None, '']
    regionvtmDisplay.DiffuseColor = [0.0, 0.0, 0.0]
    regionvtmDisplay.SelectTCoordArray = 'None'
    regionvtmDisplay.SelectNormalArray = 'None'
    regionvtmDisplay.SelectTangentArray = 'None'
    regionvtmDisplay.LineWidth = size_factor
    regionvtmDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    regionvtmDisplay.SelectOrientationVectors = 'None'
    regionvtmDisplay.ScaleFactor = 153998.45800642233
    regionvtmDisplay.SelectScaleArray = 'None'
    regionvtmDisplay.GlyphType = 'Arrow'
    regionvtmDisplay.GlyphTableIndexArray = 'None'
    regionvtmDisplay.GaussianRadius = 7699.922900321116
    regionvtmDisplay.SetScaleArray = [None, '']
    regionvtmDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    regionvtmDisplay.OpacityArray = [None, '']
    regionvtmDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    regionvtmDisplay.DataAxesGrid = 'GridAxesRepresentation'
    regionvtmDisplay.PolarAxes = 'PolarAxesRepresentation'
    regionvtmDisplay.SelectInputVectors = [None, '']
    regionvtmDisplay.WriteLog = ''

    # show data from hybridvti
    pua_3vti = Show(pua_3vti, renderView1, 'UniformGridRepresentation')

    # get 2D transfer function for 'rate'
    rateTF2D = GetTransferFunction2D('rate')
    rateTF2D.ScalarRangeInitialized = 1
    rateTF2D.Range = [4.9999999999999996e-06, 0.003000000000000001, 7.564979114249576e-07, 0.0007075337856316542]

    # get color transfer function/color map for 'rate'
    rateLUT = GetColorTransferFunction('rate')
    rateLUT.AutomaticRescaleRangeMode = 'Never'
    rateLUT.TransferFunction2D = rateTF2D
    rateLUT.RGBPoints = [4.9999999999999996e-06, 0.02, 0.3813, 0.9981, 5.822593385447654e-06, 0.02000006, 0.424267768, 0.96906969, 6.780518746451757e-06, 0.02, 0.467233763, 0.940033043, 7.896040720598767e-06, 0.02, 0.5102, 0.911, 9.195086894196724e-06, 0.02000006, 0.546401494, 0.872669438, 1.0707850425753277e-05, 0.02, 0.582600362, 0.83433295, 1.2469491812270778e-05, 0.02, 0.6188, 0.796, 1.4520956109204277e-05, 0.02000006, 0.652535156, 0.749802434, 1.690992459836571e-05, 0.02, 0.686267004, 0.703599538, 1.9691923022972557e-05, 0.02, 0.72, 0.6574, 2.293161214806088e-05, 0.02000006, 0.757035456, 0.603735359, 2.6704290642190078e-05, 0.02, 0.794067037, 0.55006613, 3.1097645211257534e-05, 0.02, 0.8311, 0.4964, 3.621378866201322e-05, 0.021354336738172372, 0.8645368555261631, 0.4285579460761159, 4.217163326508741e-05, 0.023312914349117714, 0.897999359924484, 0.36073871343115577, 4.910965458056455e-05, 0.015976108242848862, 0.9310479513349017, 0.2925631815088092, 5.718910998448275e-05, 0.27421074700988196, 0.952562960995083, 0.15356836602739213, 6.659778670305754e-05, 0.4933546281681699, 0.9619038625309482, 0.11119493614749336, 7.755436646853534e-05, 0.6439, 0.9773, 0.0469, 9.031350824245546e-05, 0.762401813, 0.984669591, 0.034600153, 0.0001051717671418187, 0.880901185, 0.992033407, 0.022299877, 0.00012247448713915892, 0.9995285432627147, 0.9995193706781492, 0.0134884641450013, 0.00014262382774051213, 0.999402998, 0.955036376, 0.079066628, 0.00016608811120182637, 0.9994, 0.910666223, 0.148134024, 0.0001934127075370494, 0.9994, 0.8663, 0.2172, 0.00022523271031334936, 0.999269665, 0.818035981, 0.217200652, 0.0002622876978513912, 0.999133332, 0.769766184, 0.2172, 0.00030543892291876073, 0.999, 0.7215, 0.2172, 0.0003556893304490061, 0.99913633, 0.673435546, 0.217200652, 0.0004142068685493377, 0.999266668, 0.625366186, 0.2172, 0.000482351634604472, 0.9994, 0.5773, 0.2172, 0.0005617074874215727, 0.999402998, 0.521068455, 0.217200652, 0.0006541188601634542, 0.9994, 0.464832771, 0.2172, 0.0007617336296968577, 0.9994, 0.4086, 0.2172, 0.0008870530387491915, 0.9947599917687346, 0.33177297300202935, 0.2112309638520206, 0.001032989831192457, 0.9867129505479589, 0.2595183410914934, 0.19012239549291934, 0.001202935951667177, 0.9912458875646419, 0.14799417507952672, 0.21078892136920357, 0.001400841383058897, 0.949903037, 0.116867171, 0.252900603, 0.0016313059542120158, 0.903199533, 0.078432949, 0.291800389, 0.001899686251727252, 0.8565, 0.04, 0.3307, 0.002212220120746587, 0.798902627, 0.04333345, 0.358434298, 0.0025761716484426584, 0.741299424, 0.0466667, 0.386166944, 0.003000000000000001, 0.6837, 0.05, 0.4139]
    rateLUT.UseLogScale = 1
    rateLUT.UseOpacityControlPointsFreehandDrawing = 1
    rateLUT.ColorSpace = 'RGB'
    rateLUT.NanColor = [1.0, 0.0, 0.0]
    rateLUT.NanOpacity = 0.0
    rateLUT.ScalarRangeInitialized = 1.0

    # get opacity transfer function/opacity map for 'rate'
    ratePWF = GetOpacityTransferFunction('rate')
    ratePWF.Points = [4.9999999999999996e-06, 0.0, 0.5, 0.0, 0.0004178252980862393, 1.0, 0.5, 0.0, 0.0020051646169275193, 1.0, 0.5, 0.0, 0.003000000000000001, 1.0, 0.5, 0.0]
    ratePWF.UseLogScale = 1
    ratePWF.ScalarRangeInitialized = 1

    # trace defaults for the display properties.
    pua_3vti.Representation = 'Slice'
    pua_3vti.ColorArrayName = ['CELLS', 'rate']
    pua_3vti.LookupTable = rateLUT
    pua_3vti.LineWidth = 2.0
    pua_3vti.SelectTCoordArray = 'None'
    pua_3vti.SelectNormalArray = 'None'
    pua_3vti.SelectTangentArray = 'None'
    pua_3vti.OSPRayScaleFunction = 'PiecewiseFunction'
    pua_3vti.SelectOrientationVectors = 'None'
    pua_3vti.ScaleFactor = 154350.0
    pua_3vti.SelectScaleArray = 'None'
    pua_3vti.GlyphType = 'Arrow'
    pua_3vti.GlyphTableIndexArray = 'None'
    pua_3vti.GaussianRadius = 7717.5
    pua_3vti.SetScaleArray = [None, '']
    pua_3vti.ScaleTransferFunction = 'PiecewiseFunction'
    pua_3vti.OpacityArray = [None, '']
    pua_3vti.OpacityTransferFunction = 'PiecewiseFunction'
    pua_3vti.DataAxesGrid = 'GridAxesRepresentation'
    pua_3vti.PolarAxes = 'PolarAxesRepresentation'
    pua_3vti.ScalarOpacityUnitDistance = 20818.636043244984
    pua_3vti.ScalarOpacityFunction = ratePWF
    pua_3vti.TransferFunction2D = rateTF2D
    pua_3vti.OpacityArrayName = ['CELLS', 'bin']
    pua_3vti.ColorArray2Name = ['CELLS', 'bin']
    pua_3vti.SliceFunction = 'Plane'
    pua_3vti.SelectInputVectors = [None, '']
    pua_3vti.WriteLog = ''

    # init the 'Plane' selected for 'SliceFunction'
    pua_3vti.SliceFunction.Origin = [1589946.015518637, 5470136.7749801995, 10.0]

    # setup the color legend parameters for each legend in this view

    # get color legend/bar for rateLUT in view renderView1
    rateLUTColorBar = GetScalarBar(rateLUT, renderView1)
    rateLUTColorBar.WindowLocation = 'Any Location'
    rateLUTColorBar.Position = [0.08282111399957778, 0.5012722646310435]
    rateLUTColorBar.Title = '$\\frac{\\mu}{N_{tot}}$'
    rateLUTColorBar.ComponentTitle = ''
    rateLUTColorBar.HorizontalTitle = 1
    rateLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    rateLUTColorBar.TitleFontSize = 40
    rateLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    rateLUTColorBar.LabelFontSize = 20
    rateLUTColorBar.ScalarBarLength = 0.32999999999999996
    rateLUTColorBar.AutomaticLabelFormat = 0
    rateLUTColorBar.LabelFormat = '%.0e'
    rateLUTColorBar.RangeLabelFormat = '%.0e'

    # set color bar visibility
    rateLUTColorBar.Visibility = 1

    # show color legend
    pua_3vti.SetScalarBarVisibility(renderView1, False)

    # ----------------------------------------------------------------
    # setup color maps and opacity mapes used in the visualization
    # note: the Get..() functions create a new object, if needed
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # restore active source
    SetActiveSource(pua_3vti)
    # ----------------------------------------------------------------

    SaveExtracts(ExtractsOutputDirectory='extracts')

    SaveScreenshot('figure_p3.png',
                   renderView1,
                   ImageResolution=[size_factor * i for i in renderView1.ViewSize])


def makefig_p4(size_factor=1):
    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # Create a new 'Render View'
    renderView1 = CreateView('RenderView')
    renderView1.ViewSize = [558, 786]
    renderView1.InteractionMode = '2D'
    renderView1.AxesGrid = 'GridAxes3DActor'
    renderView1.OrientationAxesVisibility = 0
    renderView1.CenterOfRotation = [1590196.015518637, 5465886.7749801995, 10.0]
    renderView1.StereoType = 'Crystal Eyes'
    renderView1.CameraPosition = [1592444.049301613, 5466129.775278663, 5175760.0]
    renderView1.CameraFocalPoint = [1592444.049301613, 5466129.775278663, 10.0]
    renderView1.CameraFocalDisk = 1.0
    renderView1.CameraParallelScale = 785071.8935111373
    renderView1.BackEnd = 'OSPRay raycaster'
    renderView1.OSPRayMaterialLibrary = materialLibrary1

    SetActiveView(None)

    # ----------------------------------------------------------------
    # setup view layouts
    # ----------------------------------------------------------------

    # create new layout object 'Layout #1'
    layout1 = CreateLayout(name='Layout #1')
    layout1.AssignView(0, renderView1)
    layout1.SetSize(558, 786)

    # ----------------------------------------------------------------
    # restore active view
    SetActiveView(renderView1)
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # setup the data processing pipelines
    # ----------------------------------------------------------------

    # create a new 'XML Image Data Reader'
    pua_4vti = XMLImageDataReader(
        registrationName='pua_4.vti',
        FileName=['paraview/pua_4.vti'])
    pua_4vti.CellArrayStatus = ['poly', 'bin', 'eventcount', 'rate_learning', 'bval', 'lmmin', 'mmin', 'mmax', 'model', 'params', 'rates_bin', 'rate', 'mask']
    pua_4vti.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    pua_5vti = XMLImageDataReader(
        registrationName='pua_5.vti',
        FileName=['paraview/pua_5.vti'])
    pua_5vti.CellArrayStatus = ['poly', 'bin', 'eventcount', 'rate_learning', 'bval', 'lmmin', 'mmin', 'mmax', 'model', 'params', 'rates_bin', 'rate', 'mask']
    pua_5vti.TimeArray = 'None'

    # create a new 'XML MultiBlock Data Reader'
    regionvtm = XMLMultiBlockDataReader(
        registrationName='region.vtm',
        FileName=['paraview/region.vtm'])
    regionvtm.CellArrayStatus = ['id']
    regionvtm.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    basemap_2193vti = XMLImageDataReader(
        registrationName='basemap_2193.vti',
        FileName=['paraview/basemap_2193.vti'])
    basemap_2193vti.CellArrayStatus = ['basemap', 'mask']
    basemap_2193vti.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    hybridvti = XMLImageDataReader(
        registrationName='hybrid.vti',
        FileName=['paraview/hybrid.vti'])
    hybridvti.CellArrayStatus = ['poly', 'bin', 'eventcount', 'rate_learning', 'bval', 'lmmin', 'mmin', 'mmax', 'model', 'params', 'rates_bin', 'rate', 'mask']
    hybridvti.TimeArray = 'None'

    # create a new 'XML MultiBlock Data Reader'
    coastlinevtm = XMLMultiBlockDataReader(
        registrationName='coastline.vtm',
        FileName=['paraview/coastline.vtm'])
    coastlinevtm.CellArrayStatus = ['t50_fid', 'elevation']
    coastlinevtm.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    pua_3vti = XMLImageDataReader(
        registrationName='pua_3.vti',
        FileName=['paraview/pua_3.vti'])
    pua_3vti.CellArrayStatus = ['poly', 'bin', 'eventcount', 'rate_learning', 'bval', 'lmmin', 'mmin', 'mmax', 'model', 'params', 'rates_bin', 'rate', 'mask']
    pua_3vti.TimeArray = 'None'

    # ----------------------------------------------------------------
    # setup the visualization in view 'renderView1'
    # ----------------------------------------------------------------

    # show data from basemap_2193vti
    basemap_2193vtiDisplay = Show(basemap_2193vti, renderView1, 'UniformGridRepresentation')

    # get 2D transfer function for 'basemap'
    basemapTF2D = GetTransferFunction2D('basemap')
    basemapTF2D.ScalarRangeInitialized = 1
    basemapTF2D.Range = [0.0, 441.6729559300637, 0.0, 1.0]

    # get color transfer function/color map for 'basemap'
    basemapLUT = GetColorTransferFunction('basemap')
    basemapLUT.TransferFunction2D = basemapTF2D
    basemapLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 220.83647796503186, 0.865003, 0.865003, 0.865003, 441.6729559300637, 0.705882, 0.0156863, 0.14902]
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
    basemap_2193vtiDisplay.SliceFunction.Origin = [1575927.8735133181, 5400949.511356351, -10.0]

    # show data from coastlinevtm
    coastlinevtmDisplay = Show(coastlinevtm, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    coastlinevtmDisplay.Representation = 'Surface'
    coastlinevtmDisplay.AmbientColor = [0.0, 0.0, 0.0]
    coastlinevtmDisplay.ColorArrayName = ['POINTS', '']
    coastlinevtmDisplay.DiffuseColor = [0.0, 0.0, 0.0]
    coastlinevtmDisplay.SelectTCoordArray = 'None'
    coastlinevtmDisplay.SelectNormalArray = 'None'
    coastlinevtmDisplay.SelectTangentArray = 'None'
    coastlinevtmDisplay.LineWidth = size_factor
    coastlinevtmDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    coastlinevtmDisplay.SelectOrientationVectors = 'None'
    coastlinevtmDisplay.ScaleFactor = 144610.3415327726
    coastlinevtmDisplay.SelectScaleArray = 'None'
    coastlinevtmDisplay.GlyphType = 'Arrow'
    coastlinevtmDisplay.GlyphTableIndexArray = 'None'
    coastlinevtmDisplay.GaussianRadius = 7230.51707663863
    coastlinevtmDisplay.SetScaleArray = [None, '']
    coastlinevtmDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    coastlinevtmDisplay.OpacityArray = [None, '']
    coastlinevtmDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    coastlinevtmDisplay.DataAxesGrid = 'GridAxesRepresentation'
    coastlinevtmDisplay.PolarAxes = 'PolarAxesRepresentation'
    coastlinevtmDisplay.SelectInputVectors = [None, '']
    coastlinevtmDisplay.WriteLog = ''

    # show data from regionvtm
    regionvtmDisplay = Show(regionvtm, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    regionvtmDisplay.Representation = 'Surface'
    regionvtmDisplay.AmbientColor = [0.0, 0.0, 0.0]
    regionvtmDisplay.ColorArrayName = [None, '']
    regionvtmDisplay.DiffuseColor = [0.0, 0.0, 0.0]
    regionvtmDisplay.SelectTCoordArray = 'None'
    regionvtmDisplay.SelectNormalArray = 'None'
    regionvtmDisplay.SelectTangentArray = 'None'
    regionvtmDisplay.LineWidth = size_factor
    regionvtmDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    regionvtmDisplay.SelectOrientationVectors = 'None'
    regionvtmDisplay.ScaleFactor = 153998.45800642233
    regionvtmDisplay.SelectScaleArray = 'None'
    regionvtmDisplay.GlyphType = 'Arrow'
    regionvtmDisplay.GlyphTableIndexArray = 'None'
    regionvtmDisplay.GaussianRadius = 7699.922900321116
    regionvtmDisplay.SetScaleArray = [None, '']
    regionvtmDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    regionvtmDisplay.OpacityArray = [None, '']
    regionvtmDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    regionvtmDisplay.DataAxesGrid = 'GridAxesRepresentation'
    regionvtmDisplay.PolarAxes = 'PolarAxesRepresentation'
    regionvtmDisplay.SelectInputVectors = [None, '']
    regionvtmDisplay.WriteLog = ''

    # show data from hybridvti
    pua_4vtiDisplay = Show(pua_4vti, renderView1, 'UniformGridRepresentation')

    # get 2D transfer function for 'rate'
    rateTF2D = GetTransferFunction2D('rate')
    rateTF2D.ScalarRangeInitialized = 1
    rateTF2D.Range = [4.9999999999999996e-06, 0.003000000000000001, 7.564979114249576e-07, 0.0007075337856316542]

    # get color transfer function/color map for 'rate'
    rateLUT = GetColorTransferFunction('rate')
    rateLUT.AutomaticRescaleRangeMode = 'Never'
    rateLUT.TransferFunction2D = rateTF2D
    rateLUT.RGBPoints = [4.9999999999999996e-06, 0.02, 0.3813, 0.9981, 5.822593385447654e-06, 0.02000006, 0.424267768, 0.96906969, 6.780518746451757e-06, 0.02, 0.467233763, 0.940033043, 7.896040720598767e-06, 0.02, 0.5102, 0.911, 9.195086894196724e-06, 0.02000006, 0.546401494, 0.872669438, 1.0707850425753277e-05, 0.02, 0.582600362, 0.83433295, 1.2469491812270778e-05, 0.02, 0.6188, 0.796, 1.4520956109204277e-05, 0.02000006, 0.652535156, 0.749802434, 1.690992459836571e-05, 0.02, 0.686267004, 0.703599538, 1.9691923022972557e-05, 0.02, 0.72, 0.6574, 2.293161214806088e-05, 0.02000006, 0.757035456, 0.603735359, 2.6704290642190078e-05, 0.02, 0.794067037, 0.55006613, 3.1097645211257534e-05, 0.02, 0.8311, 0.4964, 3.621378866201322e-05, 0.021354336738172372, 0.8645368555261631, 0.4285579460761159, 4.217163326508741e-05, 0.023312914349117714, 0.897999359924484, 0.36073871343115577, 4.910965458056455e-05, 0.015976108242848862, 0.9310479513349017, 0.2925631815088092, 5.718910998448275e-05, 0.27421074700988196, 0.952562960995083, 0.15356836602739213, 6.659778670305754e-05, 0.4933546281681699, 0.9619038625309482, 0.11119493614749336, 7.755436646853534e-05, 0.6439, 0.9773, 0.0469, 9.031350824245546e-05, 0.762401813, 0.984669591, 0.034600153, 0.0001051717671418187, 0.880901185, 0.992033407, 0.022299877, 0.00012247448713915892, 0.9995285432627147, 0.9995193706781492, 0.0134884641450013, 0.00014262382774051213, 0.999402998, 0.955036376, 0.079066628, 0.00016608811120182637, 0.9994, 0.910666223, 0.148134024, 0.0001934127075370494, 0.9994, 0.8663, 0.2172, 0.00022523271031334936, 0.999269665, 0.818035981, 0.217200652, 0.0002622876978513912, 0.999133332, 0.769766184, 0.2172, 0.00030543892291876073, 0.999, 0.7215, 0.2172, 0.0003556893304490061, 0.99913633, 0.673435546, 0.217200652, 0.0004142068685493377, 0.999266668, 0.625366186, 0.2172, 0.000482351634604472, 0.9994, 0.5773, 0.2172, 0.0005617074874215727, 0.999402998, 0.521068455, 0.217200652, 0.0006541188601634542, 0.9994, 0.464832771, 0.2172, 0.0007617336296968577, 0.9994, 0.4086, 0.2172, 0.0008870530387491915, 0.9947599917687346, 0.33177297300202935, 0.2112309638520206, 0.001032989831192457, 0.9867129505479589, 0.2595183410914934, 0.19012239549291934, 0.001202935951667177, 0.9912458875646419, 0.14799417507952672, 0.21078892136920357, 0.001400841383058897, 0.949903037, 0.116867171, 0.252900603, 0.0016313059542120158, 0.903199533, 0.078432949, 0.291800389, 0.001899686251727252, 0.8565, 0.04, 0.3307, 0.002212220120746587, 0.798902627, 0.04333345, 0.358434298, 0.0025761716484426584, 0.741299424, 0.0466667, 0.386166944, 0.003000000000000001, 0.6837, 0.05, 0.4139]
    rateLUT.UseLogScale = 1
    rateLUT.UseOpacityControlPointsFreehandDrawing = 1
    rateLUT.ColorSpace = 'RGB'
    rateLUT.NanColor = [1.0, 0.0, 0.0]
    rateLUT.NanOpacity = 0.0
    rateLUT.ScalarRangeInitialized = 1.0

    # get opacity transfer function/opacity map for 'rate'
    ratePWF = GetOpacityTransferFunction('rate')
    ratePWF.Points = [4.9999999999999996e-06, 0.0, 0.5, 0.0, 0.0004178252980862393, 1.0, 0.5, 0.0, 0.0020051646169275193, 1.0, 0.5, 0.0, 0.003000000000000001, 1.0, 0.5, 0.0]
    ratePWF.UseLogScale = 1
    ratePWF.ScalarRangeInitialized = 1

    # trace defaults for the display properties.
    pua_4vtiDisplay.Representation = 'Slice'
    pua_4vtiDisplay.ColorArrayName = ['CELLS', 'rate']
    pua_4vtiDisplay.LookupTable = rateLUT
    pua_4vtiDisplay.LineWidth = 2.0
    pua_4vtiDisplay.SelectTCoordArray = 'None'
    pua_4vtiDisplay.SelectNormalArray = 'None'
    pua_4vtiDisplay.SelectTangentArray = 'None'
    pua_4vtiDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    pua_4vtiDisplay.SelectOrientationVectors = 'None'
    pua_4vtiDisplay.ScaleFactor = 154350.0
    pua_4vtiDisplay.SelectScaleArray = 'None'
    pua_4vtiDisplay.GlyphType = 'Arrow'
    pua_4vtiDisplay.GlyphTableIndexArray = 'None'
    pua_4vtiDisplay.GaussianRadius = 7717.5
    pua_4vtiDisplay.SetScaleArray = [None, '']
    pua_4vtiDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    pua_4vtiDisplay.OpacityArray = [None, '']
    pua_4vtiDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    pua_4vtiDisplay.DataAxesGrid = 'GridAxesRepresentation'
    pua_4vtiDisplay.PolarAxes = 'PolarAxesRepresentation'
    pua_4vtiDisplay.ScalarOpacityUnitDistance = 20818.636043244984
    pua_4vtiDisplay.ScalarOpacityFunction = ratePWF
    pua_4vtiDisplay.TransferFunction2D = rateTF2D
    pua_4vtiDisplay.OpacityArrayName = ['CELLS', 'bin']
    pua_4vtiDisplay.ColorArray2Name = ['CELLS', 'bin']
    pua_4vtiDisplay.SliceFunction = 'Plane'
    pua_4vtiDisplay.SelectInputVectors = [None, '']
    pua_4vtiDisplay.WriteLog = ''

    # init the 'Plane' selected for 'SliceFunction'
    pua_4vtiDisplay.SliceFunction.Origin = [1589946.015518637, 5470136.7749801995, 10.0]

    # setup the color legend parameters for each legend in this view

    # get color legend/bar for rateLUT in view renderView1
    rateLUTColorBar = GetScalarBar(rateLUT, renderView1)
    rateLUTColorBar.WindowLocation = 'Any Location'
    rateLUTColorBar.Position = [0.08282111399957778, 0.5012722646310435]
    rateLUTColorBar.Title = '$\\frac{\\mu}{N_{tot}}$'
    rateLUTColorBar.ComponentTitle = ''
    rateLUTColorBar.HorizontalTitle = 1
    rateLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    rateLUTColorBar.TitleFontSize = 40
    rateLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    rateLUTColorBar.LabelFontSize = 20
    rateLUTColorBar.ScalarBarLength = 0.32999999999999996
    rateLUTColorBar.AutomaticLabelFormat = 0
    rateLUTColorBar.LabelFormat = '%.0e'
    rateLUTColorBar.RangeLabelFormat = '%.0e'

    # set color bar visibility
    rateLUTColorBar.Visibility = 1

    # show color legend
    pua_4vtiDisplay.SetScalarBarVisibility(renderView1, False)

    # ----------------------------------------------------------------
    # setup color maps and opacity mapes used in the visualization
    # note: the Get..() functions create a new object, if needed
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # restore active source
    SetActiveSource(pua_3vti)
    # ----------------------------------------------------------------

    SaveExtracts(ExtractsOutputDirectory='extracts')

    SaveScreenshot('figure_p4.png',
                   renderView1,
                   ImageResolution=[size_factor * i for i in renderView1.ViewSize])


def makefig_p5(size_factor=1):
    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # Create a new 'Render View'
    renderView1 = CreateView('RenderView')
    renderView1.ViewSize = [558, 786]
    renderView1.InteractionMode = '2D'
    renderView1.AxesGrid = 'GridAxes3DActor'
    renderView1.OrientationAxesVisibility = 0
    renderView1.CenterOfRotation = [1590196.015518637, 5465886.7749801995, 10.0]
    renderView1.StereoType = 'Crystal Eyes'
    renderView1.CameraPosition = [1592444.049301613, 5466129.775278663, 5175760.0]
    renderView1.CameraFocalPoint = [1592444.049301613, 5466129.775278663, 10.0]
    renderView1.CameraFocalDisk = 1.0
    renderView1.CameraParallelScale = 785071.8935111373
    renderView1.BackEnd = 'OSPRay raycaster'
    renderView1.OSPRayMaterialLibrary = materialLibrary1

    SetActiveView(None)

    # ----------------------------------------------------------------
    # setup view layouts
    # ----------------------------------------------------------------

    # create new layout object 'Layout #1'
    layout1 = CreateLayout(name='Layout #1')
    layout1.AssignView(0, renderView1)
    layout1.SetSize(558, 786)

    # ----------------------------------------------------------------
    # restore active view
    SetActiveView(renderView1)
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # setup the data processing pipelines
    # ----------------------------------------------------------------

    # create a new 'XML Image Data Reader'
    pua_4vti = XMLImageDataReader(
        registrationName='pua_4.vti',
        FileName=['paraview/pua_4.vti'])
    pua_4vti.CellArrayStatus = ['poly', 'bin', 'eventcount', 'rate_learning', 'bval', 'lmmin', 'mmin', 'mmax', 'model', 'params', 'rates_bin', 'rate', 'mask']
    pua_4vti.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    pua_5vti = XMLImageDataReader(
        registrationName='pua_5.vti',
        FileName=['paraview/pua_5.vti'])
    pua_5vti.CellArrayStatus = ['poly', 'bin', 'eventcount', 'rate_learning', 'bval', 'lmmin', 'mmin', 'mmax', 'model', 'params', 'rates_bin', 'rate', 'mask']
    pua_5vti.TimeArray = 'None'

    # create a new 'XML MultiBlock Data Reader'
    regionvtm = XMLMultiBlockDataReader(
        registrationName='region.vtm',
        FileName=['paraview/region.vtm'])
    regionvtm.CellArrayStatus = ['id']
    regionvtm.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    basemap_2193vti = XMLImageDataReader(
        registrationName='basemap_2193.vti',
        FileName=['paraview/basemap_2193.vti'])
    basemap_2193vti.CellArrayStatus = ['basemap', 'mask']
    basemap_2193vti.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    hybridvti = XMLImageDataReader(
        registrationName='hybrid.vti',
        FileName=['paraview/hybrid.vti'])
    hybridvti.CellArrayStatus = ['poly', 'bin', 'eventcount', 'rate_learning', 'bval', 'lmmin', 'mmin', 'mmax', 'model', 'params', 'rates_bin', 'rate', 'mask']
    hybridvti.TimeArray = 'None'

    # create a new 'XML MultiBlock Data Reader'
    coastlinevtm = XMLMultiBlockDataReader(
        registrationName='coastline.vtm',
        FileName=['paraview/coastline.vtm'])
    coastlinevtm.CellArrayStatus = ['t50_fid', 'elevation']
    coastlinevtm.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    pua_3vti = XMLImageDataReader(
        registrationName='pua_3.vti',
        FileName=['paraview/pua_3.vti'])
    pua_3vti.CellArrayStatus = ['poly', 'bin', 'eventcount', 'rate_learning', 'bval', 'lmmin', 'mmin', 'mmax', 'model', 'params', 'rates_bin', 'rate', 'mask']
    pua_3vti.TimeArray = 'None'

    # ----------------------------------------------------------------
    # setup the visualization in view 'renderView1'
    # ----------------------------------------------------------------

    # show data from basemap_2193vti
    basemap_2193vtiDisplay = Show(basemap_2193vti, renderView1, 'UniformGridRepresentation')

    # get 2D transfer function for 'basemap'
    basemapTF2D = GetTransferFunction2D('basemap')
    basemapTF2D.ScalarRangeInitialized = 1
    basemapTF2D.Range = [0.0, 441.6729559300637, 0.0, 1.0]

    # get color transfer function/color map for 'basemap'
    basemapLUT = GetColorTransferFunction('basemap')
    basemapLUT.TransferFunction2D = basemapTF2D
    basemapLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 220.83647796503186, 0.865003, 0.865003, 0.865003, 441.6729559300637, 0.705882, 0.0156863, 0.14902]
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
    basemap_2193vtiDisplay.SliceFunction.Origin = [1575927.8735133181, 5400949.511356351, -10.0]

    # show data from coastlinevtm
    coastlinevtmDisplay = Show(coastlinevtm, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    coastlinevtmDisplay.Representation = 'Surface'
    coastlinevtmDisplay.AmbientColor = [0.0, 0.0, 0.0]
    coastlinevtmDisplay.ColorArrayName = ['POINTS', '']
    coastlinevtmDisplay.DiffuseColor = [0.0, 0.0, 0.0]
    coastlinevtmDisplay.SelectTCoordArray = 'None'
    coastlinevtmDisplay.SelectNormalArray = 'None'
    coastlinevtmDisplay.SelectTangentArray = 'None'
    coastlinevtmDisplay.LineWidth = size_factor
    coastlinevtmDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    coastlinevtmDisplay.SelectOrientationVectors = 'None'
    coastlinevtmDisplay.ScaleFactor = 144610.3415327726
    coastlinevtmDisplay.SelectScaleArray = 'None'
    coastlinevtmDisplay.GlyphType = 'Arrow'
    coastlinevtmDisplay.GlyphTableIndexArray = 'None'
    coastlinevtmDisplay.GaussianRadius = 7230.51707663863
    coastlinevtmDisplay.SetScaleArray = [None, '']
    coastlinevtmDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    coastlinevtmDisplay.OpacityArray = [None, '']
    coastlinevtmDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    coastlinevtmDisplay.DataAxesGrid = 'GridAxesRepresentation'
    coastlinevtmDisplay.PolarAxes = 'PolarAxesRepresentation'
    coastlinevtmDisplay.SelectInputVectors = [None, '']
    coastlinevtmDisplay.WriteLog = ''

    # show data from regionvtm
    regionvtmDisplay = Show(regionvtm, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    regionvtmDisplay.Representation = 'Surface'
    regionvtmDisplay.AmbientColor = [0.0, 0.0, 0.0]
    regionvtmDisplay.ColorArrayName = [None, '']
    regionvtmDisplay.DiffuseColor = [0.0, 0.0, 0.0]
    regionvtmDisplay.SelectTCoordArray = 'None'
    regionvtmDisplay.SelectNormalArray = 'None'
    regionvtmDisplay.SelectTangentArray = 'None'
    regionvtmDisplay.LineWidth = size_factor
    regionvtmDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    regionvtmDisplay.SelectOrientationVectors = 'None'
    regionvtmDisplay.ScaleFactor = 153998.45800642233
    regionvtmDisplay.SelectScaleArray = 'None'
    regionvtmDisplay.GlyphType = 'Arrow'
    regionvtmDisplay.GlyphTableIndexArray = 'None'
    regionvtmDisplay.GaussianRadius = 7699.922900321116
    regionvtmDisplay.SetScaleArray = [None, '']
    regionvtmDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    regionvtmDisplay.OpacityArray = [None, '']
    regionvtmDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    regionvtmDisplay.DataAxesGrid = 'GridAxesRepresentation'
    regionvtmDisplay.PolarAxes = 'PolarAxesRepresentation'
    regionvtmDisplay.SelectInputVectors = [None, '']
    regionvtmDisplay.WriteLog = ''

    # show data from hybridvti
    pua_5vtiDisplay = Show(pua_5vti, renderView1, 'UniformGridRepresentation')

    # get 2D transfer function for 'rate'
    rateTF2D = GetTransferFunction2D('rate')
    rateTF2D.ScalarRangeInitialized = 1
    rateTF2D.Range = [4.9999999999999996e-06, 0.003000000000000001, 7.564979114249576e-07, 0.0007075337856316542]

    # get color transfer function/color map for 'rate'
    rateLUT = GetColorTransferFunction('rate')
    rateLUT.AutomaticRescaleRangeMode = 'Never'
    rateLUT.TransferFunction2D = rateTF2D
    rateLUT.RGBPoints = [4.9999999999999996e-06, 0.02, 0.3813, 0.9981, 5.822593385447654e-06, 0.02000006, 0.424267768, 0.96906969, 6.780518746451757e-06, 0.02, 0.467233763, 0.940033043, 7.896040720598767e-06, 0.02, 0.5102, 0.911, 9.195086894196724e-06, 0.02000006, 0.546401494, 0.872669438, 1.0707850425753277e-05, 0.02, 0.582600362, 0.83433295, 1.2469491812270778e-05, 0.02, 0.6188, 0.796, 1.4520956109204277e-05, 0.02000006, 0.652535156, 0.749802434, 1.690992459836571e-05, 0.02, 0.686267004, 0.703599538, 1.9691923022972557e-05, 0.02, 0.72, 0.6574, 2.293161214806088e-05, 0.02000006, 0.757035456, 0.603735359, 2.6704290642190078e-05, 0.02, 0.794067037, 0.55006613, 3.1097645211257534e-05, 0.02, 0.8311, 0.4964, 3.621378866201322e-05, 0.021354336738172372, 0.8645368555261631, 0.4285579460761159, 4.217163326508741e-05, 0.023312914349117714, 0.897999359924484, 0.36073871343115577, 4.910965458056455e-05, 0.015976108242848862, 0.9310479513349017, 0.2925631815088092, 5.718910998448275e-05, 0.27421074700988196, 0.952562960995083, 0.15356836602739213, 6.659778670305754e-05, 0.4933546281681699, 0.9619038625309482, 0.11119493614749336, 7.755436646853534e-05, 0.6439, 0.9773, 0.0469, 9.031350824245546e-05, 0.762401813, 0.984669591, 0.034600153, 0.0001051717671418187, 0.880901185, 0.992033407, 0.022299877, 0.00012247448713915892, 0.9995285432627147, 0.9995193706781492, 0.0134884641450013, 0.00014262382774051213, 0.999402998, 0.955036376, 0.079066628, 0.00016608811120182637, 0.9994, 0.910666223, 0.148134024, 0.0001934127075370494, 0.9994, 0.8663, 0.2172, 0.00022523271031334936, 0.999269665, 0.818035981, 0.217200652, 0.0002622876978513912, 0.999133332, 0.769766184, 0.2172, 0.00030543892291876073, 0.999, 0.7215, 0.2172, 0.0003556893304490061, 0.99913633, 0.673435546, 0.217200652, 0.0004142068685493377, 0.999266668, 0.625366186, 0.2172, 0.000482351634604472, 0.9994, 0.5773, 0.2172, 0.0005617074874215727, 0.999402998, 0.521068455, 0.217200652, 0.0006541188601634542, 0.9994, 0.464832771, 0.2172, 0.0007617336296968577, 0.9994, 0.4086, 0.2172, 0.0008870530387491915, 0.9947599917687346, 0.33177297300202935, 0.2112309638520206, 0.001032989831192457, 0.9867129505479589, 0.2595183410914934, 0.19012239549291934, 0.001202935951667177, 0.9912458875646419, 0.14799417507952672, 0.21078892136920357, 0.001400841383058897, 0.949903037, 0.116867171, 0.252900603, 0.0016313059542120158, 0.903199533, 0.078432949, 0.291800389, 0.001899686251727252, 0.8565, 0.04, 0.3307, 0.002212220120746587, 0.798902627, 0.04333345, 0.358434298, 0.0025761716484426584, 0.741299424, 0.0466667, 0.386166944, 0.003000000000000001, 0.6837, 0.05, 0.4139]
    rateLUT.UseLogScale = 1
    rateLUT.UseOpacityControlPointsFreehandDrawing = 1
    rateLUT.ColorSpace = 'RGB'
    rateLUT.NanColor = [1.0, 0.0, 0.0]
    rateLUT.NanOpacity = 0.0
    rateLUT.ScalarRangeInitialized = 1.0

    # get opacity transfer function/opacity map for 'rate'
    ratePWF = GetOpacityTransferFunction('rate')
    ratePWF.Points = [4.9999999999999996e-06, 0.0, 0.5, 0.0, 0.0004178252980862393, 1.0, 0.5, 0.0, 0.0020051646169275193, 1.0, 0.5, 0.0, 0.003000000000000001, 1.0, 0.5, 0.0]
    ratePWF.UseLogScale = 1
    ratePWF.ScalarRangeInitialized = 1

    # trace defaults for the display properties.
    pua_5vtiDisplay.Representation = 'Slice'
    pua_5vtiDisplay.ColorArrayName = ['CELLS', 'rate']
    pua_5vtiDisplay.LookupTable = rateLUT
    pua_5vtiDisplay.LineWidth = 2.0
    pua_5vtiDisplay.SelectTCoordArray = 'None'
    pua_5vtiDisplay.SelectNormalArray = 'None'
    pua_5vtiDisplay.SelectTangentArray = 'None'
    pua_5vtiDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    pua_5vtiDisplay.SelectOrientationVectors = 'None'
    pua_5vtiDisplay.ScaleFactor = 154350.0
    pua_5vtiDisplay.SelectScaleArray = 'None'
    pua_5vtiDisplay.GlyphType = 'Arrow'
    pua_5vtiDisplay.GlyphTableIndexArray = 'None'
    pua_5vtiDisplay.GaussianRadius = 7717.5
    pua_5vtiDisplay.SetScaleArray = [None, '']
    pua_5vtiDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    pua_5vtiDisplay.OpacityArray = [None, '']
    pua_5vtiDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    pua_5vtiDisplay.DataAxesGrid = 'GridAxesRepresentation'
    pua_5vtiDisplay.PolarAxes = 'PolarAxesRepresentation'
    pua_5vtiDisplay.ScalarOpacityUnitDistance = 20818.636043244984
    pua_5vtiDisplay.ScalarOpacityFunction = ratePWF
    pua_5vtiDisplay.TransferFunction2D = rateTF2D
    pua_5vtiDisplay.OpacityArrayName = ['CELLS', 'bin']
    pua_5vtiDisplay.ColorArray2Name = ['CELLS', 'bin']
    pua_5vtiDisplay.SliceFunction = 'Plane'
    pua_5vtiDisplay.SelectInputVectors = [None, '']
    pua_5vtiDisplay.WriteLog = ''

    # init the 'Plane' selected for 'SliceFunction'
    pua_5vtiDisplay.SliceFunction.Origin = [1589946.015518637, 5470136.7749801995, 10.0]

    # setup the color legend parameters for each legend in this view

    # get color legend/bar for rateLUT in view renderView1
    rateLUTColorBar = GetScalarBar(rateLUT, renderView1)
    rateLUTColorBar.WindowLocation = 'Any Location'
    rateLUTColorBar.Position = [0.08282111399957778, 0.5012722646310435]
    rateLUTColorBar.Title = '$\\frac{\\mu}{N_{tot}}$'
    rateLUTColorBar.ComponentTitle = ''
    rateLUTColorBar.HorizontalTitle = 1
    rateLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    rateLUTColorBar.TitleFontSize = 40
    rateLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    rateLUTColorBar.LabelFontSize = 20
    rateLUTColorBar.ScalarBarLength = 0.32999999999999996
    rateLUTColorBar.AutomaticLabelFormat = 0
    rateLUTColorBar.LabelFormat = '%.0e'
    rateLUTColorBar.RangeLabelFormat = '%.0e'

    # set color bar visibility
    rateLUTColorBar.Visibility = 1

    # show color legend
    pua_5vtiDisplay.SetScalarBarVisibility(renderView1, False)

    # ----------------------------------------------------------------
    # setup color maps and opacity mapes used in the visualization
    # note: the Get..() functions create a new object, if needed
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # restore active source
    SetActiveSource(pua_5vti)
    # ----------------------------------------------------------------

    SaveExtracts(ExtractsOutputDirectory='extracts')

    SaveScreenshot('figure_p5.png',
                   renderView1,
                   ImageResolution=[size_factor * i for i in renderView1.ViewSize])



def make_joint_figure():

    binmaps = [plt.imread(f"figure_{i}.png") for i in ['m', 'p3', 'p4', 'p5']]
    titles = ['Hybrid \n Multiplicative',
              'Poisson URZ \n $J_2$ 3-bins',
              'Poisson URZ \n $J_2$ 4-bins',
              'Poisson URZ \n $J_2$ 5-bins']
    fig, axs = plt.subplots(1, 4,
                            figsize=(12, 5),
                            gridspec_kw={'wspace': -0.0},
                            constrained_layout=False)

    for i, j in enumerate(binmaps):
        axs[i].imshow(j)
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)
        axs[i].set_title(titles[i], loc='left', fontsize=16,
                         **{'fontname': 'Ubuntu'})

    plt.savefig("forecasts_pua.jpg", dpi=400, bbox_inches='tight')
    plt.savefig("fig14.jpg", dpi=1200, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    makefig_m(5)
    makefig_p3(5)
    makefig_p4(5)
    makefig_p5(5)
    make_joint_figure()

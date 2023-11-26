# state file generated using paraview version 5.11.1
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()
import matplotlib.pyplot as plt

def make_gammamax():
    # ----------------------------------------------------------------
    # setup views used in the visualization
    # ----------------------------------------------------------------

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # Create a new 'Render View'
    renderView1 = CreateView('RenderView')
    renderView1.ViewSize = [618, 754]
    renderView1.InteractionMode = '2D'
    renderView1.AxesGrid = 'GridAxes3DActor'
    renderView1.OrientationAxesVisibility = 0
    renderView1.CenterOfRotation = [865382.5607010864, 4708139.662813947, 0.0]
    renderView1.StereoType = 'Crystal Eyes'
    renderView1.CameraPosition = [1574781.1721654106, 5463924.321078337, 5411.637910357848]
    renderView1.CameraFocalPoint = [1574781.1721654106, 5463924.321078337, 0.0]
    renderView1.CameraViewUp = [-0.002781279385139002, 0.9999961322350112, 0.0]
    renderView1.CameraFocalDisk = 1.0
    renderView1.CameraParallelScale = 735733.2358293127
    renderView1.BackEnd = 'OSPRay raycaster'
    renderView1.OSPRayMaterialLibrary = materialLibrary1

    SetActiveView(None)

    # ----------------------------------------------------------------
    # setup view layouts
    # ----------------------------------------------------------------

    # create new layout object 'Layout #1'
    layout1 = CreateLayout(name='Layout #1')
    layout1.AssignView(0, renderView1)
    layout1.SetSize(618, 754)

    # ----------------------------------------------------------------
    # restore active view
    SetActiveView(renderView1)
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # setup the data processing pipelines
    # ----------------------------------------------------------------

    # create a new 'XML Image Data Reader'
    basemap_2193vti = XMLImageDataReader(
        registrationName='basemap_2193.vti',
        FileName=['paraview/basemap_2193.vti'])
    basemap_2193vti.CellArrayStatus = ['basemap', 'mask']
    basemap_2193vti.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    strain_mapvti = XMLImageDataReader(
        registrationName='strain_map.vti',
        FileName=['paraview/strain_map.vti'])
    strain_mapvti.CellArrayStatus = ['tau_max', 'mask']
    strain_mapvti.PointArrayStatus = ['ImageScalars']
    strain_mapvti.TimeArray = 'None'

    # ----------------------------------------------------------------
    # setup the visualization in view 'renderView1'
    # ----------------------------------------------------------------

    # show data from strain_mapvti
    strain_mapvtiDisplay = Show(strain_mapvti, renderView1, 'UniformGridRepresentation')

    # get 2D transfer function for 'tau_max'
    tau_maxTF2D = GetTransferFunction2D('tau_max')
    tau_maxTF2D.ScalarRangeInitialized = 1
    tau_maxTF2D.Range = [0.0, 0.3, 0.0, 1.0]

    # get color transfer function/color map for 'tau_max'
    tau_maxLUT = GetColorTransferFunction('tau_max')
    tau_maxLUT.EnableOpacityMapping = 1
    tau_maxLUT.TransferFunction2D = tau_maxTF2D
    tau_maxLUT.RGBPoints = [0.0, 0.831373, 0.909804, 0.980392, 0.0037500000000000007, 0.74902, 0.862745, 0.960784, 0.0075000000000000015, 0.694118, 0.827451, 0.941176, 0.015000000000000003, 0.568627, 0.760784, 0.921569, 0.0225, 0.45098, 0.705882, 0.901961, 0.030000000000000006, 0.345098, 0.643137, 0.858824, 0.0375, 0.247059, 0.572549, 0.819608, 0.045, 0.180392, 0.521569, 0.780392, 0.048, 0.14902, 0.490196, 0.74902, 0.053999999999999986, 0.129412, 0.447059, 0.709804, 0.06000000000000001, 0.101961, 0.427451, 0.690196, 0.06300000000000001, 0.094118, 0.403922, 0.658824, 0.06600000000000002, 0.090196, 0.392157, 0.639216, 0.06900000000000002, 0.082353, 0.368627, 0.619608, 0.072, 0.070588, 0.352941, 0.6, 0.075, 0.066667, 0.329412, 0.568627, 0.078, 0.07451, 0.313725, 0.541176, 0.081, 0.086275, 0.305882, 0.509804, 0.084, 0.094118, 0.286275, 0.478431, 0.087, 0.101961, 0.278431, 0.45098, 0.09, 0.109804, 0.266667, 0.411765, 0.093, 0.113725, 0.258824, 0.380392, 0.096, 0.113725, 0.25098, 0.34902, 0.099, 0.109804, 0.266667, 0.321569, 0.10200000000000001, 0.105882, 0.301961, 0.262745, 0.10499999999999997, 0.094118, 0.309804, 0.243137, 0.10799999999999997, 0.082353, 0.321569, 0.227451, 0.11099999999999997, 0.07451, 0.341176, 0.219608, 0.11400000000000002, 0.070588, 0.360784, 0.211765, 0.11700000000000002, 0.066667, 0.380392, 0.215686, 0.12000000000000002, 0.062745, 0.4, 0.176471, 0.12749999999999997, 0.07451, 0.419608, 0.145098, 0.13499999999999995, 0.086275, 0.439216, 0.117647, 0.14250000000000002, 0.121569, 0.470588, 0.117647, 0.15, 0.184314, 0.501961, 0.14902, 0.1575, 0.254902, 0.541176, 0.188235, 0.165, 0.32549, 0.580392, 0.231373, 0.1725, 0.403922, 0.619608, 0.278431, 0.18, 0.501961, 0.670588, 0.333333, 0.189, 0.592157, 0.729412, 0.4, 0.195, 0.741176, 0.788235, 0.490196, 0.201, 0.858824, 0.858824, 0.603922, 0.20999999999999994, 0.921569, 0.835294, 0.580392, 0.22499999999999995, 0.901961, 0.729412, 0.494118, 0.24000000000000005, 0.858824, 0.584314, 0.388235, 0.25499999999999995, 0.8, 0.439216, 0.321569, 0.2699999999999999, 0.678431, 0.298039, 0.203922, 0.28500000000000003, 0.54902, 0.168627, 0.109804, 0.29250000000000004, 0.478431, 0.082353, 0.047059, 0.3, 0.45098, 0.007843, 0.0]
    tau_maxLUT.ColorSpace = 'RGB'
    tau_maxLUT.NanOpacity = 0.0
    tau_maxLUT.ScalarRangeInitialized = 1.0

    # get opacity transfer function/opacity map for 'tau_max'
    tau_maxPWF = GetOpacityTransferFunction('tau_max')
    tau_maxPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.009798994304359742, 0.10869565606117249, 0.5, 0.0, 0.035804018917288766, 0.5271739363670349, 0.5, 0.0, 0.11344220443365974, 0.8152174353599548, 0.5, 0.0, 0.3, 1.0, 0.5, 0.0]
    tau_maxPWF.ScalarRangeInitialized = 1

    # trace defaults for the display properties.
    strain_mapvtiDisplay.Representation = 'Slice'
    strain_mapvtiDisplay.ColorArrayName = ['CELLS', 'tau_max']
    strain_mapvtiDisplay.LookupTable = tau_maxLUT
    strain_mapvtiDisplay.SelectTCoordArray = 'None'
    strain_mapvtiDisplay.SelectNormalArray = 'None'
    strain_mapvtiDisplay.SelectTangentArray = 'None'
    strain_mapvtiDisplay.OSPRayScaleArray = 'ImageScalars'
    strain_mapvtiDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    strain_mapvtiDisplay.SelectOrientationVectors = 'None'
    strain_mapvtiDisplay.ScaleFactor = 147.4
    strain_mapvtiDisplay.SelectScaleArray = 'ImageScalars'
    strain_mapvtiDisplay.GlyphType = 'Arrow'
    strain_mapvtiDisplay.GlyphTableIndexArray = 'ImageScalars'
    strain_mapvtiDisplay.GaussianRadius = 7.37
    strain_mapvtiDisplay.SetScaleArray = ['POINTS', 'ImageScalars']
    strain_mapvtiDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    strain_mapvtiDisplay.OpacityArray = ['POINTS', 'ImageScalars']
    strain_mapvtiDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    strain_mapvtiDisplay.DataAxesGrid = 'GridAxesRepresentation'
    strain_mapvtiDisplay.PolarAxes = 'PolarAxesRepresentation'
    strain_mapvtiDisplay.ScalarOpacityUnitDistance = 15.86992564234172
    strain_mapvtiDisplay.ScalarOpacityFunction = tau_maxPWF
    strain_mapvtiDisplay.TransferFunction2D = tau_maxTF2D
    strain_mapvtiDisplay.OpacityArrayName = ['POINTS', 'ImageScalars']
    strain_mapvtiDisplay.ColorArray2Name = ['POINTS', 'ImageScalars']
    strain_mapvtiDisplay.IsosurfaceValues = [0.0]
    strain_mapvtiDisplay.SliceFunction = 'Plane'
    strain_mapvtiDisplay.SelectInputVectors = [None, '']
    strain_mapvtiDisplay.WriteLog = ''

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    strain_mapvtiDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    strain_mapvtiDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

    # init the 'Plane' selected for 'SliceFunction'
    strain_mapvtiDisplay.SliceFunction.Origin = [865382.5607010864, 4708139.662813947, 0.0]

    # show data from basemap_2193vti
    basemap_2193vtiDisplay = Show(basemap_2193vti, renderView1, 'UniformGridRepresentation')

    # get 2D transfer function for 'basemap'
    basemapTF2D = GetTransferFunction2D('basemap')
    basemapTF2D.ScalarRangeInitialized = 1
    basemapTF2D.Range = [0.0, 441.6729559300637, 0.0, 1.0]

    # get color transfer function/color map for 'basemap'
    basemapLUT = GetColorTransferFunction('basemap')
    basemapLUT.TransferFunction2D = basemapTF2D
    basemapLUT.RGBPoints = [0.0, 0.278431372549, 0.278431372549, 0.858823529412, 63.15923269799911, 0.0, 0.0, 0.360784313725, 125.87679244006814, 0.0, 1.0, 1.0, 189.47769809399733, 0.0, 0.501960784314, 0.0, 252.19525783606636, 1.0, 1.0, 0.0, 315.35449053406546, 1.0, 0.380392156863, 0.0, 378.5137232320646, 0.419607843137, 0.0, 0.0, 441.6729559300637, 0.878431372549, 0.301960784314, 0.301960784314]
    basemapLUT.ColorSpace = 'RGB'
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

    # setup the color legend parameters for each legend in this view

    # get color legend/bar for tau_maxLUT in view renderView1
    tau_maxLUTColorBar = GetScalarBar(tau_maxLUT, renderView1)
    tau_maxLUTColorBar.WindowLocation = 'Any Location'
    tau_maxLUTColorBar.Position = [0.7929936305732485, 0.0755]
    tau_maxLUTColorBar.Title = '$\\gamma_{max}$'
    tau_maxLUTColorBar.ComponentTitle = ''
    tau_maxLUTColorBar.HorizontalTitle = 1
    tau_maxLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    tau_maxLUTColorBar.TitleFontSize = 32
    tau_maxLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    tau_maxLUTColorBar.AutomaticLabelFormat = 0
    tau_maxLUTColorBar.LabelFormat = '%.2f'
    tau_maxLUTColorBar.RangeLabelFormat = '%.2f'

    # set color bar visibility
    tau_maxLUTColorBar.Visibility = 1

    # show color legend
    strain_mapvtiDisplay.SetScalarBarVisibility(renderView1, True)

    # ----------------------------------------------------------------
    # setup color maps and opacity mapes used in the visualization
    # note: the Get..() functions create a new object, if needed
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # restore active source
    SetActiveSource(strain_mapvti)
    # ----------------------------------------------------------------
    SaveExtracts(ExtractsOutputDirectory='extracts')
    res = [1236, 1508]
    SaveScreenshot('figure_gammamax.png',
                   renderView1, ImageResolution=[2*i for i in res])


def make_j2():
    materialLibrary1 = GetMaterialLibrary()

    # Create a new 'Render View'
    renderView1 = CreateView('RenderView')
    renderView1.ViewSize = [618, 754]
    renderView1.InteractionMode = '2D'
    renderView1.AxesGrid = 'GridAxes3DActor'
    renderView1.OrientationAxesVisibility = 0
    renderView1.CenterOfRotation = [865382.5607010864, 4708139.662813947, 0.0]
    renderView1.StereoType = 'Crystal Eyes'
    renderView1.CameraPosition = [1574781.1721654106, 5463924.321078337, 5411.637910357848]
    renderView1.CameraFocalPoint = [1574781.1721654106, 5463924.321078337, 0.0]
    renderView1.CameraViewUp = [-0.002781279385139002, 0.9999961322350112, 0.0]
    renderView1.CameraFocalDisk = 1.0
    renderView1.CameraParallelScale = 735733.2358293127
    renderView1.BackEnd = 'OSPRay raycaster'
    renderView1.OSPRayMaterialLibrary = materialLibrary1

    SetActiveView(None)

    # ----------------------------------------------------------------
    # setup view layouts
    # ----------------------------------------------------------------

    # create new layout object 'Layout #1'
    layout1 = CreateLayout(name='Layout #1')
    layout1.AssignView(0, renderView1)
    layout1.SetSize(618, 754)

    # ----------------------------------------------------------------
    # restore active view
    SetActiveView(renderView1)
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # setup the data processing pipelines
    # ----------------------------------------------------------------

    # create a new 'XML Image Data Reader'
    strain_mapvti = XMLImageDataReader(
        registrationName='strain_map.vti',
        FileName=['paraview/strain_map.vti'])
    strain_mapvti.CellArrayStatus = ['j2', 'mask']
    strain_mapvti.PointArrayStatus = ['ImageScalars']
    strain_mapvti.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    basemap_2193vti = XMLImageDataReader(
        registrationName='basemap_2193.vti',
        FileName=['paraview/basemap_2193.vti'])
    basemap_2193vti.CellArrayStatus = ['basemap', 'mask']
    basemap_2193vti.TimeArray = 'None'

    # ----------------------------------------------------------------
    # setup the visualization in view 'renderView1'
    # ----------------------------------------------------------------

    # show data from strain_mapvti
    strain_mapvtiDisplay = Show(strain_mapvti, renderView1, 'UniformGridRepresentation')

    # get 2D transfer function for 'j2'
    j2TF2D = GetTransferFunction2D('j2')
    j2TF2D.ScalarRangeInitialized = 1
    j2TF2D.Range = [0.0, 0.5, 0.0, 1.0]

    # get color transfer function/color map for 'j2'
    j2LUT = GetColorTransferFunction('j2')
    j2LUT.EnableOpacityMapping = 1
    j2LUT.TransferFunction2D = j2TF2D
    j2LUT.RGBPoints = [0.0, 0.831373, 0.909804, 0.980392, 0.00625, 0.74902, 0.862745, 0.960784, 0.0125, 0.694118, 0.827451, 0.941176, 0.025, 0.568627, 0.760784, 0.921569, 0.0375, 0.45098, 0.705882, 0.901961, 0.05, 0.345098, 0.643137, 0.858824, 0.0625, 0.247059, 0.572549, 0.819608, 0.075, 0.180392, 0.521569, 0.780392, 0.08, 0.14902, 0.490196, 0.74902, 0.09, 0.129412, 0.447059, 0.709804, 0.1, 0.101961, 0.427451, 0.690196, 0.105, 0.094118, 0.403922, 0.658824, 0.11, 0.090196, 0.392157, 0.639216, 0.115, 0.082353, 0.368627, 0.619608, 0.12, 0.070588, 0.352941, 0.6, 0.125, 0.066667, 0.329412, 0.568627, 0.13, 0.07451, 0.313725, 0.541176, 0.135, 0.086275, 0.305882, 0.509804, 0.14, 0.094118, 0.286275, 0.478431, 0.145, 0.101961, 0.278431, 0.45098, 0.15, 0.109804, 0.266667, 0.411765, 0.155, 0.113725, 0.258824, 0.380392, 0.16, 0.113725, 0.25098, 0.34902, 0.165, 0.109804, 0.266667, 0.321569, 0.17, 0.105882, 0.301961, 0.262745, 0.175, 0.094118, 0.309804, 0.243137, 0.18, 0.082353, 0.321569, 0.227451, 0.185, 0.07451, 0.341176, 0.219608, 0.19, 0.070588, 0.360784, 0.211765, 0.195, 0.066667, 0.380392, 0.215686, 0.2, 0.062745, 0.4, 0.176471, 0.2125, 0.07451, 0.419608, 0.145098, 0.22499999999999998, 0.086275, 0.439216, 0.117647, 0.23750000000000002, 0.121569, 0.470588, 0.117647, 0.25, 0.184314, 0.501961, 0.14902, 0.2625, 0.254902, 0.541176, 0.188235, 0.275, 0.32549, 0.580392, 0.231373, 0.2875, 0.403922, 0.619608, 0.278431, 0.3, 0.501961, 0.670588, 0.333333, 0.315, 0.592157, 0.729412, 0.4, 0.325, 0.741176, 0.788235, 0.490196, 0.335, 0.858824, 0.858824, 0.603922, 0.35, 0.921569, 0.835294, 0.580392, 0.375, 0.901961, 0.729412, 0.494118, 0.4, 0.858824, 0.584314, 0.388235, 0.425, 0.8, 0.439216, 0.321569, 0.44999999999999996, 0.678431, 0.298039, 0.203922, 0.47500000000000003, 0.54902, 0.168627, 0.109804, 0.48750000000000004, 0.478431, 0.082353, 0.047059, 0.5, 0.45098, 0.007843, 0.0]
    j2LUT.ColorSpace = 'RGB'
    j2LUT.NanColor = [0.25, 0.0, 0.0]
    j2LUT.NanOpacity = 0.0
    j2LUT.ScalarRangeInitialized = 1.0

    # get opacity transfer function/opacity map for 'j2'
    j2PWF = GetOpacityTransferFunction('j2')
    j2PWF.Points = [0.0, 0.0, 0.5, 0.0, 0.016331657173932906, 0.10869565606117249, 0.5, 0.0, 0.05967336486214795, 0.5271739363670349, 0.5, 0.0, 0.18907034072276624, 0.8152174353599548, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0]
    j2PWF.ScalarRangeInitialized = 1

    # trace defaults for the display properties.
    strain_mapvtiDisplay.Representation = 'Slice'
    strain_mapvtiDisplay.ColorArrayName = ['CELLS', 'j2']
    strain_mapvtiDisplay.LookupTable = j2LUT
    strain_mapvtiDisplay.SelectTCoordArray = 'None'
    strain_mapvtiDisplay.SelectNormalArray = 'None'
    strain_mapvtiDisplay.SelectTangentArray = 'None'
    strain_mapvtiDisplay.OSPRayScaleArray = 'ImageScalars'
    strain_mapvtiDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    strain_mapvtiDisplay.SelectOrientationVectors = 'None'
    strain_mapvtiDisplay.ScaleFactor = 147.4
    strain_mapvtiDisplay.SelectScaleArray = 'ImageScalars'
    strain_mapvtiDisplay.GlyphType = 'Arrow'
    strain_mapvtiDisplay.GlyphTableIndexArray = 'ImageScalars'
    strain_mapvtiDisplay.GaussianRadius = 7.37
    strain_mapvtiDisplay.SetScaleArray = ['POINTS', 'ImageScalars']
    strain_mapvtiDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    strain_mapvtiDisplay.OpacityArray = ['POINTS', 'ImageScalars']
    strain_mapvtiDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    strain_mapvtiDisplay.DataAxesGrid = 'GridAxesRepresentation'
    strain_mapvtiDisplay.PolarAxes = 'PolarAxesRepresentation'
    strain_mapvtiDisplay.ScalarOpacityUnitDistance = 15.86992564234172
    strain_mapvtiDisplay.ScalarOpacityFunction = j2PWF
    strain_mapvtiDisplay.TransferFunction2D = j2TF2D
    strain_mapvtiDisplay.OpacityArrayName = ['POINTS', 'ImageScalars']
    strain_mapvtiDisplay.ColorArray2Name = ['POINTS', 'ImageScalars']
    strain_mapvtiDisplay.IsosurfaceValues = [0.0]
    strain_mapvtiDisplay.SliceFunction = 'Plane'
    strain_mapvtiDisplay.SelectInputVectors = [None, '']
    strain_mapvtiDisplay.WriteLog = ''

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    strain_mapvtiDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    strain_mapvtiDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

    # init the 'Plane' selected for 'SliceFunction'
    strain_mapvtiDisplay.SliceFunction.Origin = [865382.5607010864, 4708139.662813947, 0.0]

    # show data from basemap_2193vti
    basemap_2193vtiDisplay = Show(basemap_2193vti, renderView1, 'UniformGridRepresentation')

    # get 2D transfer function for 'basemap'
    basemapTF2D = GetTransferFunction2D('basemap')
    basemapTF2D.ScalarRangeInitialized = 1
    basemapTF2D.Range = [0.0, 441.6729559300637, 0.0, 1.0]

    # get color transfer function/color map for 'basemap'
    basemapLUT = GetColorTransferFunction('basemap')
    basemapLUT.TransferFunction2D = basemapTF2D
    basemapLUT.RGBPoints = [0.0, 0.278431372549, 0.278431372549, 0.858823529412, 63.15923269799911, 0.0, 0.0, 0.360784313725, 125.87679244006814, 0.0, 1.0, 1.0, 189.47769809399733, 0.0, 0.501960784314, 0.0, 252.19525783606636, 1.0, 1.0, 0.0, 315.35449053406546, 1.0, 0.380392156863, 0.0, 378.5137232320646, 0.419607843137, 0.0, 0.0, 441.6729559300637, 0.878431372549, 0.301960784314, 0.301960784314]
    basemapLUT.ColorSpace = 'RGB'
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

    # setup the color legend parameters for each legend in this view

    # get color legend/bar for j2LUT in view renderView1
    j2LUTColorBar = GetScalarBar(j2LUT, renderView1)
    j2LUTColorBar.WindowLocation = 'Any Location'
    j2LUTColorBar.Position = [0.7929936305732485, 0.07559681697612736]
    j2LUTColorBar.Title = '$J_2$'
    j2LUTColorBar.ComponentTitle = ''
    j2LUTColorBar.HorizontalTitle = 1
    j2LUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    j2LUTColorBar.TitleFontSize = 32
    j2LUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    j2LUTColorBar.AutomaticLabelFormat = 0
    j2LUTColorBar.LabelFormat = '%.1f'
    j2LUTColorBar.RangeLabelFormat = '%.1f'

    # set color bar visibility
    j2LUTColorBar.Visibility = 1

    # show color legend
    strain_mapvtiDisplay.SetScalarBarVisibility(renderView1, True)

    # ----------------------------------------------------------------
    # setup color maps and opacity mapes used in the visualization
    # note: the Get..() functions create a new object, if needed
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # restore active source
    SetActiveSource(strain_mapvti)
    # ----------------------------------------------------------------
    SaveExtracts(ExtractsOutputDirectory='extracts')
    res = [1236, 1508]
    SaveScreenshot('figure_j2.png',
                   renderView1, ImageResolution=[2*i for i in res])


def make_ss():

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # Create a new 'Render View'
    renderView1 = CreateView('RenderView')
    renderView1.ViewSize = [618, 754]
    renderView1.InteractionMode = '2D'
    renderView1.AxesGrid = 'GridAxes3DActor'
    renderView1.OrientationAxesVisibility = 0
    renderView1.CenterOfRotation = [865382.5607010864, 4708139.662813947, 0.0]
    renderView1.StereoType = 'Crystal Eyes'
    renderView1.CameraPosition = [1574781.1721654106, 5463924.321078337, 5411.637910357848]
    renderView1.CameraFocalPoint = [1574781.1721654106, 5463924.321078337, 0.0]
    renderView1.CameraViewUp = [-0.002781279385139002, 0.9999961322350112, 0.0]
    renderView1.CameraFocalDisk = 1.0
    renderView1.CameraParallelScale = 735733.2358293127
    renderView1.BackEnd = 'OSPRay raycaster'
    renderView1.OSPRayMaterialLibrary = materialLibrary1

    SetActiveView(None)

    # ----------------------------------------------------------------
    # setup view layouts
    # ----------------------------------------------------------------

    # create new layout object 'Layout #1'
    layout1 = CreateLayout(name='Layout #1')
    layout1.AssignView(0, renderView1)
    layout1.SetSize(618, 754)

    # ----------------------------------------------------------------
    # restore active view
    SetActiveView(renderView1)
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # setup the data processing pipelines
    # ----------------------------------------------------------------

    # create a new 'XML Image Data Reader'
    basemap_2193vti = XMLImageDataReader(
        registrationName='basemap_2193.vti',
        FileName=['paraview/basemap_2193.vti'])
    basemap_2193vti.CellArrayStatus = ['basemap', 'mask']
    basemap_2193vti.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    strain_mapvti = XMLImageDataReader(
        registrationName='strain_map.vti',
        FileName=['paraview/strain_map.vti'])
    strain_mapvti.CellArrayStatus = ['ss', 'mask']
    strain_mapvti.PointArrayStatus = ['ImageScalars']
    strain_mapvti.TimeArray = 'None'

    # ----------------------------------------------------------------
    # setup the visualization in view 'renderView1'
    # ----------------------------------------------------------------

    # show data from strain_mapvti
    strain_mapvtiDisplay = Show(strain_mapvti, renderView1, 'UniformGridRepresentation')

    # get 2D transfer function for 'ss'
    ssTF2D = GetTransferFunction2D('ss')
    ssTF2D.ScalarRangeInitialized = 1
    ssTF2D.Range = [4.8039237518504585e-12, 0.3569337725639343, 0.0, 1.0]

    # get color transfer function/color map for 'ss'
    ssLUT = GetColorTransferFunction('ss')
    ssLUT.EnableOpacityMapping = 1
    ssLUT.TransferFunction2D = ssTF2D
    ssLUT.RGBPoints = [4.8039237518504585e-12, 0.831373, 0.909804, 0.980392, 0.004461672161793054, 0.74902, 0.862745, 0.960784, 0.008923344318782184, 0.694118, 0.827451, 0.941176, 0.017846688632760444, 0.568627, 0.760784, 0.921569, 0.026770032946738704, 0.45098, 0.705882, 0.901961, 0.03569337726071696, 0.345098, 0.643137, 0.858824, 0.04461672157469522, 0.247059, 0.572549, 0.819608, 0.05354006588867348, 0.180392, 0.521569, 0.780392, 0.05710940361426479, 0.14902, 0.490196, 0.74902, 0.06424807906544738, 0.129412, 0.447059, 0.709804, 0.07138675451663, 0.101961, 0.427451, 0.690196, 0.0749560922422213, 0.094118, 0.403922, 0.658824, 0.0785254299678126, 0.090196, 0.392157, 0.639216, 0.08209476769340392, 0.082353, 0.368627, 0.619608, 0.08566410541899522, 0.070588, 0.352941, 0.6, 0.08923344314458652, 0.066667, 0.329412, 0.568627, 0.09280278087017782, 0.07451, 0.313725, 0.541176, 0.09637211859576914, 0.086275, 0.305882, 0.509804, 0.09994145632136044, 0.094118, 0.286275, 0.478431, 0.10351079404695172, 0.101961, 0.278431, 0.45098, 0.10708013177254304, 0.109804, 0.266667, 0.411765, 0.11064946949813434, 0.113725, 0.258824, 0.380392, 0.11421880722372565, 0.113725, 0.25098, 0.34902, 0.11778814494931696, 0.109804, 0.266667, 0.321569, 0.12135748267490826, 0.105882, 0.301961, 0.262745, 0.12492682040049954, 0.094118, 0.309804, 0.243137, 0.12849615812609086, 0.082353, 0.321569, 0.227451, 0.13206549585168217, 0.07451, 0.341176, 0.219608, 0.1356348335772735, 0.070588, 0.360784, 0.211765, 0.1392041713028648, 0.066667, 0.380392, 0.215686, 0.1427735090284561, 0.062745, 0.4, 0.176471, 0.15169685334243435, 0.07451, 0.419608, 0.145098, 0.16062019765641258, 0.086275, 0.439216, 0.117647, 0.16954354197039087, 0.121569, 0.470588, 0.117647, 0.17846688628436913, 0.184314, 0.501961, 0.14902, 0.1873902305983474, 0.254902, 0.541176, 0.188235, 0.19631357491232568, 0.32549, 0.580392, 0.231373, 0.20523691922630388, 0.403922, 0.619608, 0.278431, 0.21416026354028217, 0.501961, 0.670588, 0.333333, 0.2248682767170561, 0.592157, 0.729412, 0.4, 0.2320069521682387, 0.741176, 0.788235, 0.490196, 0.23914562761942132, 0.858824, 0.858824, 0.603922, 0.24985364079619518, 0.921569, 0.835294, 0.580392, 0.26770032942415173, 0.901961, 0.729412, 0.494118, 0.28554701805210825, 0.858824, 0.584314, 0.388235, 0.30339370668006477, 0.8, 0.439216, 0.321569, 0.32124039530802123, 0.678431, 0.298039, 0.203922, 0.3390870839359778, 0.54902, 0.168627, 0.109804, 0.3480104282499561, 0.478431, 0.082353, 0.047059, 0.3569337725639343, 0.45098, 0.007843, 0.0]
    ssLUT.ColorSpace = 'RGB'
    ssLUT.NanOpacity = 0.0
    ssLUT.ScalarRangeInitialized = 1.0

    # get opacity transfer function/opacity map for 'ss'
    ssPWF = GetOpacityTransferFunction('ss')
    ssPWF.Points = [4.8039237518504585e-12, 0.0, 0.5, 0.0, 0.01165864001927244, 0.10869565606117249, 0.5, 0.0, 0.04259887848789176, 0.5271739363670349, 0.5, 0.0, 0.1349711799912382, 0.8152174353599548, 0.5, 0.0, 0.3569337725639343, 1.0, 0.5, 0.0]
    ssPWF.ScalarRangeInitialized = 1

    # trace defaults for the display properties.
    strain_mapvtiDisplay.Representation = 'Slice'
    strain_mapvtiDisplay.ColorArrayName = ['CELLS', 'ss']
    strain_mapvtiDisplay.LookupTable = ssLUT
    strain_mapvtiDisplay.SelectTCoordArray = 'None'
    strain_mapvtiDisplay.SelectNormalArray = 'None'
    strain_mapvtiDisplay.SelectTangentArray = 'None'
    strain_mapvtiDisplay.OSPRayScaleArray = 'ImageScalars'
    strain_mapvtiDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    strain_mapvtiDisplay.SelectOrientationVectors = 'None'
    strain_mapvtiDisplay.ScaleFactor = 147.4
    strain_mapvtiDisplay.SelectScaleArray = 'ImageScalars'
    strain_mapvtiDisplay.GlyphType = 'Arrow'
    strain_mapvtiDisplay.GlyphTableIndexArray = 'ImageScalars'
    strain_mapvtiDisplay.GaussianRadius = 7.37
    strain_mapvtiDisplay.SetScaleArray = ['POINTS', 'ImageScalars']
    strain_mapvtiDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    strain_mapvtiDisplay.OpacityArray = ['POINTS', 'ImageScalars']
    strain_mapvtiDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    strain_mapvtiDisplay.DataAxesGrid = 'GridAxesRepresentation'
    strain_mapvtiDisplay.PolarAxes = 'PolarAxesRepresentation'
    strain_mapvtiDisplay.ScalarOpacityUnitDistance = 15.86992564234172
    strain_mapvtiDisplay.ScalarOpacityFunction = ssPWF
    strain_mapvtiDisplay.TransferFunction2D = ssTF2D
    strain_mapvtiDisplay.OpacityArrayName = ['POINTS', 'ImageScalars']
    strain_mapvtiDisplay.ColorArray2Name = ['POINTS', 'ImageScalars']
    strain_mapvtiDisplay.IsosurfaceValues = [0.0]
    strain_mapvtiDisplay.SliceFunction = 'Plane'
    strain_mapvtiDisplay.SelectInputVectors = [None, '']
    strain_mapvtiDisplay.WriteLog = ''

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    strain_mapvtiDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    strain_mapvtiDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

    # init the 'Plane' selected for 'SliceFunction'
    strain_mapvtiDisplay.SliceFunction.Origin = [865382.5607010864, 4708139.662813947, 0.0]

    # show data from basemap_2193vti
    basemap_2193vtiDisplay = Show(basemap_2193vti, renderView1, 'UniformGridRepresentation')

    # get 2D transfer function for 'basemap'
    basemapTF2D = GetTransferFunction2D('basemap')
    basemapTF2D.ScalarRangeInitialized = 1
    basemapTF2D.Range = [0.0, 441.6729559300637, 0.0, 1.0]

    # get color transfer function/color map for 'basemap'
    basemapLUT = GetColorTransferFunction('basemap')
    basemapLUT.TransferFunction2D = basemapTF2D
    basemapLUT.RGBPoints = [0.0, 0.278431372549, 0.278431372549, 0.858823529412, 63.15923269799911, 0.0, 0.0, 0.360784313725, 125.87679244006814, 0.0, 1.0, 1.0, 189.47769809399733, 0.0, 0.501960784314, 0.0, 252.19525783606636, 1.0, 1.0, 0.0, 315.35449053406546, 1.0, 0.380392156863, 0.0, 378.5137232320646, 0.419607843137, 0.0, 0.0, 441.6729559300637, 0.878431372549, 0.301960784314, 0.301960784314]
    basemapLUT.ColorSpace = 'RGB'
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

    # setup the color legend parameters for each legend in this view

    # get color legend/bar for ssLUT in view renderView1
    ssLUTColorBar = GetScalarBar(ssLUT, renderView1)
    ssLUTColorBar.WindowLocation = 'Any Location'
    ssLUTColorBar.Position = [0.7929936305732485, 0.0755]
    ssLUTColorBar.Title = 'SS'
    ssLUTColorBar.ComponentTitle = ''
    ssLUTColorBar.HorizontalTitle = 1
    ssLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    ssLUTColorBar.TitleFontSize = 32
    ssLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    ssLUTColorBar.ScalarBarLength = 0.32999999999999996
    ssLUTColorBar.AutomaticLabelFormat = 0
    ssLUTColorBar.LabelFormat = '%.2f'
    ssLUTColorBar.RangeLabelFormat = '%.2f'

    # set color bar visibility
    ssLUTColorBar.Visibility = 1

    # show color legend
    strain_mapvtiDisplay.SetScalarBarVisibility(renderView1, True)

    # ----------------------------------------------------------------
    # setup color maps and opacity mapes used in the visualization
    # note: the Get..() functions create a new object, if needed
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # restore active source
    SetActiveSource(strain_mapvti)
    # ----------------------------------------------------------------
    SaveExtracts(ExtractsOutputDirectory='extracts')
    res = [1236, 1508]
    SaveScreenshot('figure_ss.png',
                   renderView1, ImageResolution=[2*i for i in res])


def make_joint_figure():

    j2_img = plt.imread("figure_j2.png")
    ss_img = plt.imread("figure_ss.png")
    gammamax_img = plt.imread("figure_gammamax.png")

    figsize_ratio = (gammamax_img.shape[1] / gammamax_img.shape[0])

    fig, axs = plt.subplots(1, 3,
                            figsize=(12, 4/figsize_ratio),
                            gridspec_kw={'wspace': -60},
                            constrained_layout=True)

    axs[0].imshow(j2_img)
    axs[0].get_xaxis().set_visible(False)
    axs[0].get_yaxis().set_visible(False)
    axs[0].text(100, 220, 'a)', fontsize=20, **{'fontname': 'Ubuntu'})

    axs[1].imshow(ss_img)
    axs[1].get_xaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)
    axs[1].text(100, 220, 'b)', fontsize=20, **{'fontname': 'Ubuntu'})

    axs[2].imshow(gammamax_img)
    axs[2].get_xaxis().set_visible(False)
    axs[2].get_yaxis().set_visible(False)
    axs[2].text(100, 220, 'c)', fontsize=20, **{'fontname': 'Ubuntu'})

    plt.savefig("strain_maps.jpg", dpi=600)
    plt.savefig("fig11.jpg", dpi=1200)
    plt.show()


if __name__ == '__main__':

    make_j2()
    make_gammamax()
    make_ss()
    make_joint_figure()


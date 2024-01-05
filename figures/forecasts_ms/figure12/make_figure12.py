# state file generated using paraview version 5.11.1
import paraview

paraview.compatibility.major = 5
paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()
import matplotlib.pyplot as plt


def makefig_3bins():
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

    # create a new 'XML Image Data Reader'
    basemap_2193vti = XMLImageDataReader(
        registrationName='basemap_2193.vti',
        FileName=['paraview/basemap_2193.vti'])
    basemap_2193vti.CellArrayStatus = ['basemap', 'mask']
    basemap_2193vti.TimeArray = 'None'

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

    # get 2D transfer function for 'j2_disc'
    j2_discTF2D = GetTransferFunction2D('j2_disc')
    j2_discTF2D.ScalarRangeInitialized = 1
    j2_discTF2D.Range = [-0.5, 2.5, 0.0, 1.0]

    # get color transfer function/color map for 'j2_disc'
    j2_discLUT = GetColorTransferFunction('j2_disc')
    j2_discLUT.AutomaticRescaleRangeMode = 'Never'
    j2_discLUT.AnnotationsInitialized = 1
    j2_discLUT.TransferFunction2D = j2_discTF2D
    j2_discLUT.RGBPoints = [-0.5, 0.054902, 0.109804, 0.121569, -0.35, 0.07451,
                            0.172549, 0.180392, -0.19999999999999996, 0.086275,
                            0.231373, 0.219608, -0.050000000000000044,
                            0.094118, 0.278431, 0.25098, 0.10000000000000009,
                            0.109804, 0.34902, 0.278431, 0.25, 0.113725, 0.4,
                            0.278431, 0.3999999999999999, 0.117647, 0.45098,
                            0.270588, 0.5499999999999998, 0.117647, 0.490196,
                            0.243137, 0.7000000000000002, 0.113725, 0.521569,
                            0.203922, 0.8500000000000001, 0.109804, 0.54902,
                            0.152941, 1.0, 0.082353, 0.588235, 0.082353,
                            1.1500000000000001, 0.109804, 0.631373, 0.05098,
                            1.2999999999999998, 0.211765, 0.678431, 0.082353,
                            1.4500000000000002, 0.317647, 0.721569, 0.113725,
                            1.5999999999999996, 0.431373, 0.760784, 0.160784,
                            1.75, 0.556863, 0.8, 0.239216, 1.9000000000000004,
                            0.666667, 0.839216, 0.294118, 2.05, 0.784314,
                            0.878431, 0.396078, 2.2, 0.886275, 0.921569,
                            0.533333, 2.3499999999999996, 0.960784, 0.94902,
                            0.670588, 2.5, 1.0, 0.984314, 0.901961]
    j2_discLUT.ColorSpace = 'Lab'
    j2_discLUT.NanColor = [0.1725490196078431, 0.6352941176470588,
                           0.3725490196078431]
    j2_discLUT.NanOpacity = 0.0
    j2_discLUT.NumberOfTableValues = 3
    j2_discLUT.ScalarRangeInitialized = 1.0
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
                         -0.06718064844608307, 0.8478261232376099, 0.5, 0.0,
                         0.6960351765155792, 0.875, 0.5, 0.0, 2.5, 1.0, 0.5,
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
    j2_discLUTColorBar.Title = 'Bin'
    j2_discLUTColorBar.ComponentTitle = ''
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
    j2_discLUTColorBar.CustomLabels = [0.0, 1.0, 2.0]
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
    # ----------------------------------------------------------------
    res = [1290, 1572]
    SaveScreenshot('figure3bins.png',
                   renderView1, ImageResolution=[2 * i for i in res])


def makefig_4bins():
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

    # create a new 'XML Image Data Reader'
    basemap_2193vti = XMLImageDataReader(
        registrationName='basemap_2193.vti',
        FileName=['paraview/basemap_2193.vti'])
    basemap_2193vti.CellArrayStatus = ['basemap', 'mask']
    basemap_2193vti.TimeArray = 'None'

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

    # get 2D transfer function for 'j2_disc'
    j2_discTF2D = GetTransferFunction2D('j2_disc')
    j2_discTF2D.ScalarRangeInitialized = 1
    j2_discTF2D.Range = [-0.5, 3.5, 0.0, 1.0]

    # get color transfer function/color map for 'j2_disc'
    j2_discLUT = GetColorTransferFunction('j2_disc')
    j2_discLUT.AutomaticRescaleRangeMode = 'Never'
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
    j2_discLUTColorBar.Title = 'Bin'
    j2_discLUTColorBar.ComponentTitle = ''
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
    # ----------------------------------------------------------------
    SaveExtracts(ExtractsOutputDirectory='extracts')
    # ----------------------------------------------------------------
    res = [1290, 1572]
    SaveScreenshot('figure4bins.png',
                   renderView1, ImageResolution=[2 * i for i in res])


def makefig_5bins():
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
    j2_discTF2D.Range = [-0.5, 4.5, 0.0, 1.0]

    # get color transfer function/color map for 'j2_disc'
    j2_discLUT = GetColorTransferFunction('j2_disc')
    j2_discLUT.AutomaticRescaleRangeMode = 'Never'
    j2_discLUT.AnnotationsInitialized = 1
    j2_discLUT.TransferFunction2D = j2_discTF2D
    j2_discLUT.RGBPoints = [-0.5, 0.054902, 0.109804, 0.121569,
                            -0.2499999999999999, 0.07451, 0.172549, 0.180392,
                            1.1102230246251565e-16, 0.086275, 0.231373,
                            0.219608, 0.25, 0.094118, 0.278431, 0.25098,
                            0.5000000000000002, 0.109804, 0.34902, 0.278431,
                            0.75, 0.113725, 0.4, 0.278431, 1.0, 0.117647,
                            0.45098, 0.270588, 1.2499999999999996, 0.117647,
                            0.490196, 0.243137, 1.5000000000000004, 0.113725,
                            0.521569, 0.203922, 1.75, 0.109804, 0.54902,
                            0.152941, 2.0, 0.082353, 0.588235, 0.082353, 2.25,
                            0.109804, 0.631373, 0.05098, 2.5, 0.211765,
                            0.678431, 0.082353, 2.75, 0.317647, 0.721569,
                            0.113725, 2.999999999999999, 0.431373, 0.760784,
                            0.160784, 3.25, 0.556863, 0.8, 0.239216,
                            3.500000000000001, 0.666667, 0.839216, 0.294118,
                            3.75, 0.784314, 0.878431, 0.396078, 4.0, 0.886275,
                            0.921569, 0.533333, 4.249999999999999, 0.960784,
                            0.94902, 0.670588, 4.5, 1.0, 0.984314, 0.901961]
    j2_discLUT.ColorSpace = 'Lab'
    j2_discLUT.NanColor = [0.1725490196078431, 0.6352941176470588,
                           0.3725490196078431]
    j2_discLUT.NanOpacity = 0.0
    j2_discLUT.NumberOfTableValues = 5
    j2_discLUT.ScalarRangeInitialized = 1.0
    j2_discLUT.VectorComponent = 2
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
                         0.22136558592319489, 0.8478261232376099, 0.5, 0.0,
                         1.4933919608592987, 0.875, 0.5, 0.0, 4.5, 1.0, 0.5,
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
    j2_discLUTColorBar.Title = 'Bin'
    j2_discLUTColorBar.ComponentTitle = ''
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
    j2_discLUTColorBar.CustomLabels = [0.0, 1.0, 2.0, 3.0, 4.0]
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
    # ----------------------------------------------------------------
    SaveExtracts(ExtractsOutputDirectory='extracts')
    # ----------------------------------------------------------------
    res = [1290, 1572]
    SaveScreenshot('figure5bins.png',
                   renderView1, ImageResolution=[2 * i for i in res])


def makefig_6bins():
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

    # create a new 'XML Image Data Reader'
    basemap_2193vti = XMLImageDataReader(
        registrationName='basemap_2193.vti',
        FileName=['paraview/basemap_2193.vti'])
    basemap_2193vti.CellArrayStatus = ['basemap', 'mask']
    basemap_2193vti.TimeArray = 'None'

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

    # get 2D transfer function for 'j2_disc'
    j2_discTF2D = GetTransferFunction2D('j2_disc')
    j2_discTF2D.ScalarRangeInitialized = 1
    j2_discTF2D.Range = [-0.5, 5.5, 0.0, 1.0]

    # get color transfer function/color map for 'j2_disc'
    j2_discLUT = GetColorTransferFunction('j2_disc')
    j2_discLUT.AutomaticRescaleRangeMode = 'Never'
    j2_discLUT.AnnotationsInitialized = 1
    j2_discLUT.TransferFunction2D = j2_discTF2D
    j2_discLUT.RGBPoints = [-0.5, 0.054902, 0.109804, 0.121569,
                            -0.19999999999999984, 0.07451, 0.172549, 0.180392,
                            0.10000000000000009, 0.086275, 0.231373, 0.219608,
                            0.3999999999999999, 0.094118, 0.278431, 0.25098,
                            0.7000000000000002, 0.109804, 0.34902, 0.278431,
                            1.0, 0.113725, 0.4, 0.278431, 1.2999999999999998,
                            0.117647, 0.45098, 0.270588, 1.5999999999999996,
                            0.117647, 0.490196, 0.243137, 1.9000000000000004,
                            0.113725, 0.521569, 0.203922, 2.2, 0.109804,
                            0.54902, 0.152941, 2.5, 0.082353, 0.588235,
                            0.082353, 2.8000000000000003, 0.109804, 0.631373,
                            0.05098, 3.0999999999999996, 0.211765, 0.678431,
                            0.082353, 3.4000000000000004, 0.317647, 0.721569,
                            0.113725, 3.6999999999999993, 0.431373, 0.760784,
                            0.160784, 4.0, 0.556863, 0.8, 0.239216,
                            4.300000000000001, 0.666667, 0.839216, 0.294118,
                            4.6, 0.784314, 0.878431, 0.396078, 4.9, 0.886275,
                            0.921569, 0.533333, 5.199999999999999, 0.960784,
                            0.94902, 0.670588, 5.5, 1.0, 0.984314, 0.901961]
    j2_discLUT.ColorSpace = 'Lab'
    j2_discLUT.NanColor = [0.1725490196078431, 0.6352941176470588,
                           0.3725490196078431]
    j2_discLUT.NanOpacity = 0.0
    j2_discLUT.NumberOfTableValues = 6
    j2_discLUT.ScalarRangeInitialized = 1.0
    j2_discLUT.VectorComponent = 3
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
                         0.36563870310783386, 0.8478261232376099, 0.5, 0.0,
                         1.8920703530311584, 0.875, 0.5, 0.0, 5.5, 1.0, 0.5,
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
    j2_discLUTColorBar.Title = 'Bin'
    j2_discLUTColorBar.ComponentTitle = ''
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
    j2_discLUTColorBar.CustomLabels = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
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

    # ----------------------------------------------------------------
    SaveExtracts(ExtractsOutputDirectory='extracts')
    # ----------------------------------------------------------------
    res = [1290, 1572]
    SaveScreenshot('figure6bins.png',
                   renderView1, ImageResolution=[2 * i for i in res])


def make_joint_figure():
    binmaps = [plt.imread(f"figure{i}bins.png") for i in range(3, 7)]
    binhists = [plt.imread(f"hist_maxent_{i}.png") for i in range(1, 5)]
    letters = ['a)', 'b)', 'c)', 'd)']
    fig, axs = plt.subplots(2, 4,
                            figsize=(12, 7),
                            gridspec_kw={'wspace': -0.01,
                                         'hspace': -0.1},
                            constrained_layout=False)

    for i, j in enumerate(binmaps):
        axs[0, i].imshow(j)
        axs[0, i].get_xaxis().set_visible(False)
        axs[0, i].get_yaxis().set_visible(False)
        axs[0, i].text(100, 250, letters[i], fontsize=16,
                       **{'fontname': 'Ubuntu'})

    for i, j in enumerate(binhists):
        axs[1, i].imshow(j)
        axs[1, i].axis('off')

    plt.savefig("strain_bins.jpg", dpi=600, bbox_inches='tight')
    plt.savefig("fig12.jpg", dpi=1200, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    makefig_3bins()
    makefig_4bins()
    makefig_5bins()
    makefig_6bins()

    make_joint_figure()

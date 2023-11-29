# state file generated using paraview version 5.11.1
import paraview
import numpy as np
import matplotlib.pyplot as plt
from nonpoisson.geo import reproject_coordinates
from nonpoisson import paths
paraview.compatibility.major = 5
paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()


def makefig_fe_low(size_factor):

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
    renderView1.Background2 = [0.11764705882352941, 0.11764705882352941, 0.11764705882352941]
    renderView1.Background = [0.7568627450980392, 0.7568627450980392, 0.7568627450980392]
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
    basemap_2193vti = XMLImageDataReader(registrationName='basemap_2193.vti', FileName=['paraview/basemap_2193.vti'])
    basemap_2193vti.CellArrayStatus = ['basemap', 'mask']
    basemap_2193vti.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    fevti = XMLImageDataReader(registrationName='fe.vti', FileName=['paraview/fe.vti'])
    fevti.CellArrayStatus = ['poly', 'bin', 'eventcount', 'rate_learning', 'bval', 'lmmin', 'mmin', 'mmax', 'model', 'params', 'rates_bin', 'rate', 'mask']
    fevti.TimeArray = 'None'

    # create a new 'XML MultiBlock Data Reader'
    regionvtm = XMLMultiBlockDataReader(registrationName='region.vtm', FileName=['paraview/region.vtm'])
    regionvtm.CellArrayStatus = ['id']
    regionvtm.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    hybridvti = XMLImageDataReader(registrationName='hybrid.vti', FileName=['paraview/hybrid.vti'])
    hybridvti.CellArrayStatus = ['poly', 'bin', 'eventcount', 'rate_learning', 'bval', 'lmmin', 'mmin', 'mmax', 'model', 'params', 'rates_bin', 'rate', 'mask']
    hybridvti.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    fe_lowvti = XMLImageDataReader(registrationName='fe_low.vti', FileName=['paraview/fe_low.vti'])
    fe_lowvti.CellArrayStatus = ['poly', 'bin', 'eventcount', 'rate_learning', 'bval', 'lmmin', 'mmin', 'mmax', 'model', 'params', 'rates_bin', 'rate', 'mask']
    fe_lowvti.TimeArray = 'None'

    # create a new 'XML MultiBlock Data Reader'
    coastlinevtm = XMLMultiBlockDataReader(registrationName='coastline.vtm', FileName=['paraview/coastline.vtm'])
    coastlinevtm.CellArrayStatus = ['t50_fid', 'elevation']
    coastlinevtm.TimeArray = 'None'

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
    basemapLUT.AutomaticRescaleRangeMode = 'Never'
    basemapLUT.EnableOpacityMapping = 1
    basemapLUT.TransferFunction2D = basemapTF2D
    basemapLUT.RGBPoints = [0.0, 0.8901960784313725, 0.9568627450980393, 0.984313725490196, 7.3264608544966245, 0.807843137254902, 0.9019607843137255, 0.9607843137254902, 14.652921708993299, 0.7607843137254902, 0.8588235294117647, 0.9294117647058824, 29.305843417986598, 0.6862745098039216, 0.7843137254901961, 0.8705882352941177, 43.95876512697984, 0.6, 0.6980392156862745, 0.8, 58.611686835973195, 0.5176470588235295, 0.615686274509804, 0.7254901960784313, 73.26460854496644, 0.44313725490196076, 0.5333333333333333, 0.6431372549019608, 87.91753025395974, 0.37254901960784315, 0.4588235294117647, 0.5686274509803921, 102.57045196295304, 0.3058823529411765, 0.38823529411764707, 0.49411764705882355, 117.22337367194633, 0.24705882352941178, 0.3176470588235294, 0.41568627450980394, 131.87629538093964, 0.2, 0.2549019607843137, 0.34509803921568627, 146.52921708993287, 0.14902, 0.196078, 0.278431, 154.05125062177993, 0.2, 0.1450980392156863, 0.13725490196078433, 161.56886742406763, 0.23137254901960785, 0.17254901960784313, 0.16470588235294117, 169.08648422635534, 0.2549019607843137, 0.2, 0.1843137254901961, 176.60410102864307, 0.28627450980392155, 0.23137254901960785, 0.21176470588235294, 184.12171783093078, 0.3176470588235294, 0.2627450980392157, 0.23921568627450981, 191.63933463321848, 0.34509803921568627, 0.29411764705882354, 0.26666666666666666, 199.1569514355062, 0.37254901960784315, 0.3215686274509804, 0.29411764705882354, 206.67456823779392, 0.403921568627451, 0.3568627450980392, 0.3176470588235294, 214.19218504008163, 0.43137254901960786, 0.38823529411764707, 0.34509803921568627, 221.70980184236933, 0.4627450980392157, 0.4196078431372549, 0.37254901960784315, 229.22741864465706, 0.4980392156862745, 0.4588235294117647, 0.40784313725490196, 236.74503544694477, 0.5372549019607843, 0.5058823529411764, 0.44313725490196076, 244.26265224923253, 0.5803921568627451, 0.5568627450980392, 0.48627450980392156, 251.78026905152018, 0.6313725490196078, 0.6078431372549019, 0.5333333333333333, 259.29788585380794, 0.6784313725490196, 0.6627450980392157, 0.5803921568627451, 266.8155026560956, 0.7254901960784313, 0.7137254901960784, 0.6274509803921569, 274.33311945838335, 0.7764705882352941, 0.7647058823529411, 0.6784313725490196, 281.85073626067106, 0.8274509803921568, 0.8235294117647058, 0.7372549019607844, 289.3683530629588, 0.8901960784313725, 0.8901960784313725, 0.807843137254902, 296.88596986524647, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 296.88596986524647, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 296.8903865948058, 0.9882352941176471, 0.9882352941176471, 0.8588235294117647, 297.17995173347634, 0.984313725490196, 0.9882352941176471, 0.8549019607843137, 304.12951506156867, 0.9882352941176471, 0.9882352941176471, 0.7568627450980392, 311.3686435283316, 0.9803921568627451, 0.9725490196078431, 0.6705882352941176, 318.6077719950945, 0.9803921568627451, 0.9490196078431372, 0.596078431372549, 325.8469004618574, 0.9725490196078431, 0.9176470588235294, 0.5137254901960784, 333.0860289286203, 0.9686274509803922, 0.8784313725490196, 0.4235294117647059, 340.32515739538314, 0.9647058823529412, 0.8274509803921568, 0.3254901960784314, 347.56428586214605, 0.9529411764705882, 0.7686274509803922, 0.24705882352941178, 354.80341432890896, 0.9450980392156862, 0.7098039215686275, 0.1843137254901961, 362.04254279567186, 0.9372549019607843, 0.6431372549019608, 0.12156862745098039, 369.28167126243477, 0.9254901960784314, 0.5803921568627451, 0.0784313725490196, 376.5207997291976, 0.9176470588235294, 0.5215686274509804, 0.047058823529411764, 383.7599281959606, 0.9019607843137255, 0.4588235294117647, 0.027450980392156862, 390.99905666272343, 0.8627450980392157, 0.3803921568627451, 0.011764705882352941, 398.23818512948634, 0.788235294117647, 0.2901960784313726, 0.0, 405.47731359624925, 0.7411764705882353, 0.22745098039215686, 0.00392156862745098, 412.7164420630121, 0.6627450980392157, 0.16862745098039217, 0.01568627450980392, 419.955570529775, 0.5882352941176471, 0.11372549019607843, 0.023529411764705882, 427.19469899653797, 0.49411764705882355, 0.054901960784313725, 0.03529411764705882, 434.4338274633009, 0.396078431372549, 0.0392156862745098, 0.058823529411764705, 441.6729559300637, 0.301961, 0.047059, 0.090196]
    basemapLUT.ColorSpace = 'Lab'
    basemapLUT.NanColor = [0.25, 0.0, 0.0]
    basemapLUT.NanOpacity = 0.0
    basemapLUT.NumberOfTableValues = 10
    basemapLUT.ScalarRangeInitialized = 1.0

    # get opacity transfer function/opacity map for 'basemap'
    basemapPWF = GetOpacityTransferFunction('basemap')
    basemapPWF.Points = [0.0, 0.0, 0.5, 0.0, 6.713812823552561, 0.5955882668495178, 0.5, 0.0, 40.76243438313056, 0.625, 0.5, 0.0, 44.598900499092004, 0.8897058963775635, 0.5, 0.0, 99.74807952635801, 0.9705882668495178, 0.5, 0.0, 441.6729559300637, 1.0, 0.5, 0.0]
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

    # init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
    basemap_2193vtiDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    basemap_2193vtiDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    basemap_2193vtiDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

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
    coastlinevtmDisplay.Position = [0.0, 0.0, 200.0]
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

    # init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
    coastlinevtmDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    coastlinevtmDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    coastlinevtmDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PolarAxesRepresentation' selected for 'PolarAxes'
    coastlinevtmDisplay.PolarAxes.Translation = [0.0, 0.0, 200.0]

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
    regionvtmDisplay.Position = [0.0, 0.0, 200.0]
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

    # init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
    regionvtmDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    regionvtmDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    regionvtmDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PolarAxesRepresentation' selected for 'PolarAxes'
    regionvtmDisplay.PolarAxes.Translation = [0.0, 0.0, 200.0]

    # show data from fe_lowvti
    fe_lowvtiDisplay = Show(fe_lowvti, renderView1, 'UniformGridRepresentation')

    # get 2D transfer function for 'rate'
    rateTF2D = GetTransferFunction2D('rate')
    rateTF2D.ScalarRangeInitialized = 1
    rateTF2D.Range = [5e-06, 0.001, 0.0, 1.0]

    # get color transfer function/color map for 'rate'
    rateLUT = GetColorTransferFunction('rate')
    rateLUT.AutomaticRescaleRangeMode = 'Never'
    rateLUT.TransferFunction2D = rateTF2D
    rateLUT.RGBPoints = [4.9999999999999996e-06, 0.02, 0.3813, 0.9981, 5.67226396226752e-06, 0.02000006, 0.424267768, 0.96906969, 6.434915691527779e-06, 0.02, 0.467233763, 0.940033043, 7.3001080754565605e-06, 0.02, 0.5102, 0.911, 8.281627991414054e-06, 0.02000006, 0.546401494, 0.872669438, 9.395116000920818e-06, 0.02, 0.582600362, 0.83433295, 1.065831558266922e-05, 0.02, 0.6188, 0.796, 1.2091355875609768e-05, 0.02000006, 0.652535156, 0.749802434, 1.3717072437634615e-05, 0.02, 0.686267004, 0.703599538, 1.5561371131161585e-05, 0.02, 0.72, 0.6574, 1.7653640934151603e-05, 0.02000006, 0.757035456, 0.603735359, 2.0027222254719816e-05, 0.02, 0.794067037, 0.55006613, 2.2719938211953857e-05, 0.02, 0.8311, 0.4964, 2.577469734892213e-05, 0.021354336738172372, 0.8645368555261631, 0.4285579460761159, 2.9240177382128637e-05, 0.023312914349117714, 0.897999359924484, 0.36073871343115577, 3.317160088299169e-05, 0.015976108242848862, 0.9310479513349017, 0.2925631815088092, 3.763161525186305e-05, 0.27421074700988196, 0.952562960995083, 0.15356836602739213, 4.269129100701191e-05, 0.4933546281681699, 0.9619038625309482, 0.11119493614749336, 4.843125429634984e-05, 0.6439, 0.9773, 0.0469, 5.494297167851996e-05, 0.762401813, 0.984669591, 0.034600153, 6.233020764639076e-05, 0.880901185, 0.992033407, 0.022299877, 7.071067811865475e-05, 0.9995285432627147, 0.9995193706781492, 0.0134884641450013, 8.021792624798877e-05, 0.999402998, 0.955036376, 0.079066628, 9.10034504368601e-05, 0.9994, 0.910666223, 0.148134024, 0.00010323911847100001, 0.9994, 0.8663, 0.2172, 0.00011711990623986435, 0.999269665, 0.818035981, 0.217200652, 0.00013286700468570668, 0.999133332, 0.769766184, 0.2172, 0.00015073134449063294, 0.999, 0.7215, 0.2172, 0.00017099759466766945, 0.99913633, 0.673435546, 0.217200652, 0.00019398869877357026, 0.999266668, 0.625366186, 0.2172, 0.00022007102102809866, 0.9994, 0.5773, 0.2172, 0.0002496601843434203, 0.999402998, 0.521068455, 0.217200652, 0.0002832276932928504, 0.9994, 0.464832771, 0.2172, 0.0003213084475562387, 0.9994, 0.4086, 0.2172, 0.0003645092655690757, 0.9947599917687346, 0.33177297300202935, 0.2112309638520206, 0.0004135185542000143, 0.9867129505479589, 0.2595183410914934, 0.19012239549291934, 0.00046911727854354095, 0.9912458875646419, 0.14799417507952672, 0.21078892136920357, 0.0005321914066319095, 0.949903037, 0.116867171, 0.252900603, 0.000603746027373328, 0.903199533, 0.078432949, 0.291800389, 0.0006849213666863824, 0.8565, 0.04, 0.3307, 0.000777010957048437, 0.798902627, 0.04333345, 0.358434298, 0.0008814822499905691, 0.741299424, 0.0466667, 0.386166944, 0.001, 0.6837, 0.05, 0.4139]

    rateLUT.UseLogScale = 1
    rateLUT.ColorSpace = 'RGB'
    rateLUT.NanColor = [1.0, 0.0, 0.0]
    rateLUT.NanOpacity = 0.0
    rateLUT.NumberOfTableValues = 269
    rateLUT.ScalarRangeInitialized = 1.0

    # get opacity transfer function/opacity map for 'rate'
    ratePWF = GetOpacityTransferFunction('rate')
    ratePWF.Points = [5e-06, 0.0, 0.5, 0.0, 5.052660319488496e-05, 0.5955882668495178, 0.5, 0.0, 0.0002814115152135492, 0.625, 0.5, 0.0, 0.0003074267281964422, 0.8897058963775635, 0.5, 0.0, 0.0006813952697813511, 0.9705882668495178, 0.5, 0.0, 0.003, 1.0, 0.5, 0.0]
    ratePWF.ScalarRangeInitialized = 1

    # trace defaults for the display properties.
    fe_lowvtiDisplay.Representation = 'Slice'
    fe_lowvtiDisplay.ColorArrayName = ['CELLS', 'rate']
    fe_lowvtiDisplay.LookupTable = rateLUT
    fe_lowvtiDisplay.SelectTCoordArray = 'None'
    fe_lowvtiDisplay.SelectNormalArray = 'None'
    fe_lowvtiDisplay.SelectTangentArray = 'None'
    fe_lowvtiDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    fe_lowvtiDisplay.SelectOrientationVectors = 'None'
    fe_lowvtiDisplay.ScaleFactor = 154350.0
    fe_lowvtiDisplay.SelectScaleArray = 'None'
    fe_lowvtiDisplay.GlyphType = 'Arrow'
    fe_lowvtiDisplay.GlyphTableIndexArray = 'None'
    fe_lowvtiDisplay.GaussianRadius = 7717.5
    fe_lowvtiDisplay.SetScaleArray = [None, '']
    fe_lowvtiDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    fe_lowvtiDisplay.OpacityArray = [None, '']
    fe_lowvtiDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    fe_lowvtiDisplay.DataAxesGrid = 'GridAxesRepresentation'
    fe_lowvtiDisplay.PolarAxes = 'PolarAxesRepresentation'
    fe_lowvtiDisplay.ScalarOpacityUnitDistance = 20818.636043244984
    fe_lowvtiDisplay.ScalarOpacityFunction = ratePWF
    fe_lowvtiDisplay.TransferFunction2D = rateTF2D
    fe_lowvtiDisplay.OpacityArrayName = ['CELLS', 'bin']
    fe_lowvtiDisplay.ColorArray2Name = ['CELLS', 'bin']
    fe_lowvtiDisplay.SliceFunction = 'Plane'
    fe_lowvtiDisplay.SelectInputVectors = [None, '']
    fe_lowvtiDisplay.WriteLog = ''

    # init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
    fe_lowvtiDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    fe_lowvtiDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    fe_lowvtiDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'Plane' selected for 'SliceFunction'
    fe_lowvtiDisplay.SliceFunction.Origin = [1589946.015518637, 5470136.7749801995, 10.0]

    # show data from fevti
    fevtiDisplay = Show(fevti, renderView1, 'UniformGridRepresentation')

    # trace defaults for the display properties.
    fevtiDisplay.Representation = 'Slice'
    fevtiDisplay.ColorArrayName = ['CELLS', 'rate']
    fevtiDisplay.LookupTable = rateLUT
    fevtiDisplay.SelectTCoordArray = 'None'
    fevtiDisplay.SelectNormalArray = 'None'
    fevtiDisplay.SelectTangentArray = 'None'
    fevtiDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    fevtiDisplay.SelectOrientationVectors = 'None'
    fevtiDisplay.ScaleFactor = 154350.0
    fevtiDisplay.SelectScaleArray = 'None'
    fevtiDisplay.GlyphType = 'Arrow'
    fevtiDisplay.GlyphTableIndexArray = 'None'
    fevtiDisplay.GaussianRadius = 7717.5
    fevtiDisplay.SetScaleArray = [None, '']
    fevtiDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    fevtiDisplay.OpacityArray = [None, '']
    fevtiDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    fevtiDisplay.DataAxesGrid = 'GridAxesRepresentation'
    fevtiDisplay.PolarAxes = 'PolarAxesRepresentation'
    fevtiDisplay.ScalarOpacityUnitDistance = 20818.636043244984
    fevtiDisplay.ScalarOpacityFunction = ratePWF
    fevtiDisplay.TransferFunction2D = rateTF2D
    fevtiDisplay.OpacityArrayName = ['CELLS', 'bin']
    fevtiDisplay.ColorArray2Name = ['CELLS', 'bin']
    fevtiDisplay.SliceFunction = 'Plane'
    fevtiDisplay.SelectInputVectors = [None, '']
    fevtiDisplay.WriteLog = ''

    # init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
    fevtiDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    fevtiDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    fevtiDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'Plane' selected for 'SliceFunction'
    fevtiDisplay.SliceFunction.Origin = [1589946.015518637, 5470136.7749801995, 10.0]

    # show data from hybridvti
    hybridvtiDisplay = Show(hybridvti, renderView1, 'UniformGridRepresentation')

    # trace defaults for the display properties.
    hybridvtiDisplay.Representation = 'Slice'
    hybridvtiDisplay.ColorArrayName = ['CELLS', 'rate']
    hybridvtiDisplay.LookupTable = rateLUT
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

    # init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
    hybridvtiDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    hybridvtiDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    hybridvtiDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'Plane' selected for 'SliceFunction'
    hybridvtiDisplay.SliceFunction.Origin = [1589946.015518637, 5470136.7749801995, 10.0]

    # setup the color legend parameters for each legend in this view

    # get 2D transfer function for 'mask'
    maskTF2D = GetTransferFunction2D('mask')

    # get color transfer function/color map for 'mask'
    maskLUT = GetColorTransferFunction('mask')
    maskLUT.AutomaticRescaleRangeMode = 'Never'
    maskLUT.EnableOpacityMapping = 1
    maskLUT.TransferFunction2D = maskTF2D
    maskLUT.RGBPoints = [0.0, 0.8901960784313725, 0.9568627450980393, 0.984313725490196, 0.016587977045297575, 0.807843137254902, 0.9019607843137255, 0.9607843137254902, 0.03317595409059526, 0.7607843137254902, 0.8588235294117647, 0.9294117647058824, 0.06635190818119052, 0.6862745098039216, 0.7843137254901961, 0.8705882352941177, 0.09952786227178567, 0.6, 0.6980392156862745, 0.8, 0.13270381636238104, 0.5176470588235295, 0.615686274509804, 0.7254901960784313, 0.1658797704529762, 0.44313725490196076, 0.5333333333333333, 0.6431372549019608, 0.19905572454357146, 0.37254901960784315, 0.4588235294117647, 0.5686274509803921, 0.23223167863416672, 0.3058823529411765, 0.38823529411764707, 0.49411764705882355, 0.265407632724762, 0.24705882352941178, 0.3176470588235294, 0.41568627450980394, 0.29858358681535724, 0.2, 0.2549019607843137, 0.34509803921568627, 0.3317595409059524, 0.14902, 0.196078, 0.278431, 0.348790317707777, 0.2, 0.1450980392156863, 0.13725490196078433, 0.3658110945096016, 0.23137254901960785, 0.17254901960784313, 0.16470588235294117, 0.38283187131142615, 0.2549019607843137, 0.2, 0.1843137254901961, 0.3998526481132507, 0.28627450980392155, 0.23137254901960785, 0.21176470588235294, 0.4168734249150753, 0.3176470588235294, 0.2627450980392157, 0.23921568627450981, 0.43389420171689985, 0.34509803921568627, 0.29411764705882354, 0.26666666666666666, 0.4509149785187244, 0.37254901960784315, 0.3215686274509804, 0.29411764705882354, 0.467935755320549, 0.403921568627451, 0.3568627450980392, 0.3176470588235294, 0.48495653212237355, 0.43137254901960786, 0.38823529411764707, 0.34509803921568627, 0.5019773089241981, 0.4627450980392157, 0.4196078431372549, 0.37254901960784315, 0.5189980857260227, 0.4980392156862745, 0.4588235294117647, 0.40784313725490196, 0.5360188625278473, 0.5372549019607843, 0.5058823529411764, 0.44313725490196076, 0.5530396393296719, 0.5803921568627451, 0.5568627450980392, 0.48627450980392156, 0.5700604161314964, 0.6313725490196078, 0.6078431372549019, 0.5333333333333333, 0.5870811929333211, 0.6784313725490196, 0.6627450980392157, 0.5803921568627451, 0.6041019697351455, 0.7254901960784313, 0.7137254901960784, 0.6274509803921569, 0.6211227465369702, 0.7764705882352941, 0.7647058823529411, 0.6784313725490196, 0.6381435233387946, 0.8274509803921568, 0.8235294117647058, 0.7372549019607844, 0.6551643001406193, 0.8901960784313725, 0.8901960784313725, 0.807843137254902, 0.6721850769424438, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 0.6721850769424438, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 0.6721950769424438, 0.9882352941176471, 0.9882352941176471, 0.8588235294117647, 0.672850686788559, 0.984313725490196, 0.9882352941176471, 0.8549019607843137, 0.6885853230953216, 0.9882352941176471, 0.9882352941176471, 0.7568627450980392, 0.7049755692481995, 0.9803921568627451, 0.9725490196078431, 0.6705882352941176, 0.7213658154010772, 0.9803921568627451, 0.9490196078431372, 0.596078431372549, 0.7377560615539551, 0.9725490196078431, 0.9176470588235294, 0.5137254901960784, 0.7541463077068329, 0.9686274509803922, 0.8784313725490196, 0.4235294117647059, 0.7705365538597106, 0.9647058823529412, 0.8274509803921568, 0.3254901960784314, 0.7869268000125885, 0.9529411764705882, 0.7686274509803922, 0.24705882352941178, 0.8033170461654663, 0.9450980392156862, 0.7098039215686275, 0.1843137254901961, 0.8197072923183442, 0.9372549019607843, 0.6431372549019608, 0.12156862745098039, 0.8360975384712219, 0.9254901960784314, 0.5803921568627451, 0.0784313725490196, 0.8524877846240997, 0.9176470588235294, 0.5215686274509804, 0.047058823529411764, 0.8688780307769776, 0.9019607843137255, 0.4588235294117647, 0.027450980392156862, 0.8852682769298553, 0.8627450980392157, 0.3803921568627451, 0.011764705882352941, 0.9016585230827332, 0.788235294117647, 0.2901960784313726, 0.0, 0.918048769235611, 0.7411764705882353, 0.22745098039215686, 0.00392156862745098, 0.9344390153884887, 0.6627450980392157, 0.16862745098039217, 0.01568627450980392, 0.9508292615413665, 0.5882352941176471, 0.11372549019607843, 0.023529411764705882, 0.9672195076942445, 0.49411764705882355, 0.054901960784313725, 0.03529411764705882, 0.9836097538471222, 0.396078431372549, 0.0392156862745098, 0.058823529411764705, 1.0, 0.301961, 0.047059, 0.090196]
    maskLUT.ColorSpace = 'Lab'
    maskLUT.NanColor = [0.25, 0.0, 0.0]
    maskLUT.NanOpacity = 0.0
    maskLUT.NumberOfTableValues = 10
    maskLUT.ScalarRangeInitialized = 1.0

    # get color legend/bar for maskLUT in view renderView1
    maskLUTColorBar = GetScalarBar(maskLUT, renderView1)
    maskLUTColorBar.WindowLocation = 'Any Location'
    maskLUTColorBar.Title = 'mask'
    maskLUTColorBar.ComponentTitle = ''
    maskLUTColorBar.HorizontalTitle = 1
    maskLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    maskLUTColorBar.TitleFontSize = 24
    maskLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    maskLUTColorBar.LabelFormat = '%.4f'
    maskLUTColorBar.RangeLabelFormat = '%.3f'

    # set color bar visibility
    maskLUTColorBar.Visibility = 0

    # get 2D transfer function for 'vtkBlockColors'
    vtkBlockColorsTF2D = GetTransferFunction2D('vtkBlockColors')

    # get color transfer function/color map for 'vtkBlockColors'
    vtkBlockColorsLUT = GetColorTransferFunction('vtkBlockColors')
    vtkBlockColorsLUT.AutomaticRescaleRangeMode = 'Never'
    vtkBlockColorsLUT.InterpretValuesAsCategories = 1
    vtkBlockColorsLUT.AnnotationsInitialized = 1
    vtkBlockColorsLUT.EnableOpacityMapping = 1
    vtkBlockColorsLUT.TransferFunction2D = vtkBlockColorsTF2D
    vtkBlockColorsLUT.RGBPoints = [0.0, 0.8901960784313725, 0.9568627450980393, 0.984313725490196, 0.016587977045297575, 0.807843137254902, 0.9019607843137255, 0.9607843137254902, 0.03317595409059526, 0.7607843137254902, 0.8588235294117647, 0.9294117647058824, 0.06635190818119052, 0.6862745098039216, 0.7843137254901961, 0.8705882352941177, 0.09952786227178567, 0.6, 0.6980392156862745, 0.8, 0.13270381636238104, 0.5176470588235295, 0.615686274509804, 0.7254901960784313, 0.1658797704529762, 0.44313725490196076, 0.5333333333333333, 0.6431372549019608, 0.19905572454357146, 0.37254901960784315, 0.4588235294117647, 0.5686274509803921, 0.23223167863416672, 0.3058823529411765, 0.38823529411764707, 0.49411764705882355, 0.265407632724762, 0.24705882352941178, 0.3176470588235294, 0.41568627450980394, 0.29858358681535724, 0.2, 0.2549019607843137, 0.34509803921568627, 0.3317595409059524, 0.14902, 0.196078, 0.278431, 0.348790317707777, 0.2, 0.1450980392156863, 0.13725490196078433, 0.3658110945096016, 0.23137254901960785, 0.17254901960784313, 0.16470588235294117, 0.38283187131142615, 0.2549019607843137, 0.2, 0.1843137254901961, 0.3998526481132507, 0.28627450980392155, 0.23137254901960785, 0.21176470588235294, 0.4168734249150753, 0.3176470588235294, 0.2627450980392157, 0.23921568627450981, 0.43389420171689985, 0.34509803921568627, 0.29411764705882354, 0.26666666666666666, 0.4509149785187244, 0.37254901960784315, 0.3215686274509804, 0.29411764705882354, 0.467935755320549, 0.403921568627451, 0.3568627450980392, 0.3176470588235294, 0.48495653212237355, 0.43137254901960786, 0.38823529411764707, 0.34509803921568627, 0.5019773089241981, 0.4627450980392157, 0.4196078431372549, 0.37254901960784315, 0.5189980857260227, 0.4980392156862745, 0.4588235294117647, 0.40784313725490196, 0.5360188625278473, 0.5372549019607843, 0.5058823529411764, 0.44313725490196076, 0.5530396393296719, 0.5803921568627451, 0.5568627450980392, 0.48627450980392156, 0.5700604161314964, 0.6313725490196078, 0.6078431372549019, 0.5333333333333333, 0.5870811929333211, 0.6784313725490196, 0.6627450980392157, 0.5803921568627451, 0.6041019697351455, 0.7254901960784313, 0.7137254901960784, 0.6274509803921569, 0.6211227465369702, 0.7764705882352941, 0.7647058823529411, 0.6784313725490196, 0.6381435233387946, 0.8274509803921568, 0.8235294117647058, 0.7372549019607844, 0.6551643001406193, 0.8901960784313725, 0.8901960784313725, 0.807843137254902, 0.6721850769424438, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 0.6721850769424438, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 0.6721950769424438, 0.9882352941176471, 0.9882352941176471, 0.8588235294117647, 0.672850686788559, 0.984313725490196, 0.9882352941176471, 0.8549019607843137, 0.6885853230953216, 0.9882352941176471, 0.9882352941176471, 0.7568627450980392, 0.7049755692481995, 0.9803921568627451, 0.9725490196078431, 0.6705882352941176, 0.7213658154010772, 0.9803921568627451, 0.9490196078431372, 0.596078431372549, 0.7377560615539551, 0.9725490196078431, 0.9176470588235294, 0.5137254901960784, 0.7541463077068329, 0.9686274509803922, 0.8784313725490196, 0.4235294117647059, 0.7705365538597106, 0.9647058823529412, 0.8274509803921568, 0.3254901960784314, 0.7869268000125885, 0.9529411764705882, 0.7686274509803922, 0.24705882352941178, 0.8033170461654663, 0.9450980392156862, 0.7098039215686275, 0.1843137254901961, 0.8197072923183442, 0.9372549019607843, 0.6431372549019608, 0.12156862745098039, 0.8360975384712219, 0.9254901960784314, 0.5803921568627451, 0.0784313725490196, 0.8524877846240997, 0.9176470588235294, 0.5215686274509804, 0.047058823529411764, 0.8688780307769776, 0.9019607843137255, 0.4588235294117647, 0.027450980392156862, 0.8852682769298553, 0.8627450980392157, 0.3803921568627451, 0.011764705882352941, 0.9016585230827332, 0.788235294117647, 0.2901960784313726, 0.0, 0.918048769235611, 0.7411764705882353, 0.22745098039215686, 0.00392156862745098, 0.9344390153884887, 0.6627450980392157, 0.16862745098039217, 0.01568627450980392, 0.9508292615413665, 0.5882352941176471, 0.11372549019607843, 0.023529411764705882, 0.9672195076942445, 0.49411764705882355, 0.054901960784313725, 0.03529411764705882, 0.9836097538471222, 0.396078431372549, 0.0392156862745098, 0.058823529411764705, 1.0, 0.301961, 0.047059, 0.090196]
    vtkBlockColorsLUT.ColorSpace = 'Lab'
    vtkBlockColorsLUT.NanColor = [0.25, 0.0, 0.0]
    vtkBlockColorsLUT.NanOpacity = 0.0
    vtkBlockColorsLUT.NumberOfTableValues = 10
    vtkBlockColorsLUT.Annotations = ['0', '0', '1', '1', '2', '2', '3', '3', '4', '4', '5', '5', '6', '6', '7', '7', '8', '8', '9', '9', '10', '10', '11', '11']
    vtkBlockColorsLUT.IndexedColors = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.63, 0.63, 1.0, 0.67, 0.5, 0.33, 1.0, 0.5, 0.75, 0.53, 0.35, 0.7, 1.0, 0.75, 0.5]

    # get color legend/bar for vtkBlockColorsLUT in view renderView1
    vtkBlockColorsLUTColorBar = GetScalarBar(vtkBlockColorsLUT, renderView1)
    vtkBlockColorsLUTColorBar.WindowLocation = 'Any Location'
    vtkBlockColorsLUTColorBar.Title = 'vtkBlockColors'
    vtkBlockColorsLUTColorBar.ComponentTitle = ''
    vtkBlockColorsLUTColorBar.HorizontalTitle = 1
    vtkBlockColorsLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    vtkBlockColorsLUTColorBar.TitleFontSize = 24
    vtkBlockColorsLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    vtkBlockColorsLUTColorBar.LabelFormat = '%.4f'
    vtkBlockColorsLUTColorBar.RangeLabelFormat = '%.3f'

    # set color bar visibility
    vtkBlockColorsLUTColorBar.Visibility = 0

    # get 2D transfer function for 'bin'
    binTF2D = GetTransferFunction2D('bin')
    binTF2D.ScalarRangeInitialized = 1
    binTF2D.Range = [1.0, 1.000244140625, 0.0, 1.0]

    # get color transfer function/color map for 'bin'
    binLUT = GetColorTransferFunction('bin')
    binLUT.AutomaticRescaleRangeMode = 'Never'
    binLUT.EnableOpacityMapping = 1
    binLUT.TransferFunction2D = binTF2D
    binLUT.RGBPoints = [1.0, 0.8901960784313725, 0.9568627450980393, 0.984313725490196, 1.0000040497990834, 0.807843137254902, 0.9019607843137255, 0.9607843137254902, 1.0000080995981666, 0.7607843137254902, 0.8588235294117647, 0.9294117647058824, 1.0000161991963332, 0.6862745098039216, 0.7843137254901961, 0.8705882352941177, 1.0000242987945, 0.6, 0.6980392156862745, 0.8, 1.0000323983926667, 0.5176470588235295, 0.615686274509804, 0.7254901960784313, 1.0000404979908333, 0.44313725490196076, 0.5333333333333333, 0.6431372549019608, 1.0000485975889999, 0.37254901960784315, 0.4588235294117647, 0.5686274509803921, 1.0000566971871665, 0.3058823529411765, 0.38823529411764707, 0.49411764705882355, 1.000064796785333, 0.24705882352941178, 0.3176470588235294, 0.41568627450980394, 1.0000728963835, 0.2, 0.2549019607843137, 0.34509803921568627, 1.0000809959816666, 0.14902, 0.196078, 0.278431, 1.000085153886159, 0.2, 0.1450980392156863, 0.13725490196078433, 1.0000893093492456, 0.23137254901960785, 0.17254901960784313, 0.16470588235294117, 1.0000934648123319, 0.2549019607843137, 0.2, 0.1843137254901961, 1.0000976202754184, 0.28627450980392155, 0.23137254901960785, 0.21176470588235294, 1.0001017757385047, 0.3176470588235294, 0.2627450980392157, 0.23921568627450981, 1.000105931201591, 0.34509803921568627, 0.29411764705882354, 0.26666666666666666, 1.0001100866646775, 0.37254901960784315, 0.3215686274509804, 0.29411764705882354, 1.0001142421277638, 0.403921568627451, 0.3568627450980392, 0.3176470588235294, 1.00011839759085, 0.43137254901960786, 0.38823529411764707, 0.34509803921568627, 1.0001225530539366, 0.4627450980392157, 0.4196078431372549, 0.37254901960784315, 1.000126708517023, 0.4980392156862745, 0.4588235294117647, 0.40784313725490196, 1.0001308639801094, 0.5372549019607843, 0.5058823529411764, 0.44313725490196076, 1.0001350194431957, 0.5803921568627451, 0.5568627450980392, 0.48627450980392156, 1.000139174906282, 0.6313725490196078, 0.6078431372549019, 0.5333333333333333, 1.0001433303693685, 0.6784313725490196, 0.6627450980392157, 0.5803921568627451, 1.0001474858324548, 0.7254901960784313, 0.7137254901960784, 0.6274509803921569, 1.0001516412955413, 0.7764705882352941, 0.7647058823529411, 0.6784313725490196, 1.0001557967586276, 0.8274509803921568, 0.8235294117647058, 0.7372549019607844, 1.000159952221714, 0.8901960784313725, 0.8901960784313725, 0.807843137254902, 1.0001641076848005, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 1.0001641076848005, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 1.0001641101262067, 0.9882352941176471, 0.9882352941176471, 0.8588235294117647, 1.0001642701872042, 0.984313725490196, 0.9882352941176471, 0.8549019607843137, 1.0001681116511463, 0.9882352941176471, 0.9882352941176471, 0.7568627450980392, 1.000172113176086, 0.9803921568627451, 0.9725490196078431, 0.6705882352941176, 1.0001761147010257, 0.9803921568627451, 0.9490196078431372, 0.596078431372549, 1.0001801162259654, 0.9725490196078431, 0.9176470588235294, 0.5137254901960784, 1.000184117750905, 0.9686274509803922, 0.8784313725490196, 0.4235294117647059, 1.0001881192758446, 0.9647058823529412, 0.8274509803921568, 0.3254901960784314, 1.0001921208007842, 0.9529411764705882, 0.7686274509803922, 0.24705882352941178, 1.000196122325724, 0.9450980392156862, 0.7098039215686275, 0.1843137254901961, 1.0002001238506637, 0.9372549019607843, 0.6431372549019608, 0.12156862745098039, 1.0002041253756033, 0.9254901960784314, 0.5803921568627451, 0.0784313725490196, 1.000208126900543, 0.9176470588235294, 0.5215686274509804, 0.047058823529411764, 1.0002121284254826, 0.9019607843137255, 0.4588235294117647, 0.027450980392156862, 1.0002161299504224, 0.8627450980392157, 0.3803921568627451, 0.011764705882352941, 1.000220131475362, 0.788235294117647, 0.2901960784313726, 0.0, 1.0002241330003017, 0.7411764705882353, 0.22745098039215686, 0.00392156862745098, 1.0002281345252413, 0.6627450980392157, 0.16862745098039217, 0.01568627450980392, 1.000232136050181, 0.5882352941176471, 0.11372549019607843, 0.023529411764705882, 1.0002361375751208, 0.49411764705882355, 0.054901960784313725, 0.03529411764705882, 1.0002401391000604, 0.396078431372549, 0.0392156862745098, 0.058823529411764705, 1.000244140625, 0.301961, 0.047059, 0.090196]
    binLUT.ColorSpace = 'Lab'
    binLUT.NanColor = [0.25, 0.0, 0.0]
    binLUT.NanOpacity = 0.0
    binLUT.NumberOfTableValues = 10
    binLUT.ScalarRangeInitialized = 1.0

    # get color legend/bar for binLUT in view renderView1
    binLUTColorBar = GetScalarBar(binLUT, renderView1)
    binLUTColorBar.WindowLocation = 'Any Location'
    binLUTColorBar.Title = 'bin'
    binLUTColorBar.ComponentTitle = ''
    binLUTColorBar.HorizontalTitle = 1
    binLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    binLUTColorBar.TitleFontSize = 24
    binLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    binLUTColorBar.LabelFormat = '%.4f'
    binLUTColorBar.RangeLabelFormat = '%.3f'

    # set color bar visibility
    binLUTColorBar.Visibility = 0

    # get color legend/bar for basemapLUT in view renderView1
    basemapLUTColorBar = GetScalarBar(basemapLUT, renderView1)
    basemapLUTColorBar.WindowLocation = 'Any Location'
    basemapLUTColorBar.Title = 'basemap'
    basemapLUTColorBar.ComponentTitle = 'Magnitude'
    basemapLUTColorBar.HorizontalTitle = 1
    basemapLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    basemapLUTColorBar.TitleFontSize = 24
    basemapLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    basemapLUTColorBar.LabelFormat = '%.4f'
    basemapLUTColorBar.RangeLabelFormat = '%.3f'

    # set color bar visibility
    basemapLUTColorBar.Visibility = 0

    # get color legend/bar for rateLUT in view renderView1
    rateLUTColorBar = GetScalarBar(rateLUT, renderView1)
    rateLUTColorBar.WindowLocation = 'Any Location'
    rateLUTColorBar.Position = [0.7806810035842294, 0.06071246819338426]
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
    fe_lowvtiDisplay.SetScalarBarVisibility(renderView1, False)

    # show color legend
    fevtiDisplay.SetScalarBarVisibility(renderView1, False)

    # hide data in view
    Hide(fevti, renderView1)

    # show color legend
    hybridvtiDisplay.SetScalarBarVisibility(renderView1, False)

    # hide data in view
    Hide(hybridvti, renderView1)

    # ----------------------------------------------------------------
    # setup color maps and opacity mapes used in the visualization
    # note: the Get..() functions create a new object, if needed
    # ----------------------------------------------------------------

    # get opacity transfer function/opacity map for 'bin'
    binPWF = GetOpacityTransferFunction('bin')
    binPWF.Points = [1.0, 0.0, 0.5, 0.0, 1.0000037111497022, 0.5955882668495178, 0.5, 0.0, 1.0000225319799938, 0.625, 0.5, 0.0, 1.000024652637876, 0.8897058963775635, 0.5, 0.0, 1.000055137083109, 0.9705882668495178, 0.5, 0.0, 1.000244140625, 1.0, 0.5, 0.0]
    binPWF.ScalarRangeInitialized = 1

    # get opacity transfer function/opacity map for 'vtkBlockColors'
    vtkBlockColorsPWF = GetOpacityTransferFunction('vtkBlockColors')
    vtkBlockColorsPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # get opacity transfer function/opacity map for 'mask'
    maskPWF = GetOpacityTransferFunction('mask')
    maskPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    maskPWF.ScalarRangeInitialized = 1
    coastlinevtmDisplay.LineWidth = size_factor
    regionvtmDisplay.LineWidth = size_factor
    # ----------------------------------------------------------------
    # restore active source
    SetActiveSource(fe_lowvti)
    # ----------------------------------------------------------------

    SaveExtracts(ExtractsOutputDirectory='extracts')

    SaveScreenshot('figure_fe_low.png',
               renderView1,
               ImageResolution=[size_factor * i for i in renderView1.ViewSize])


def makefig_npfe_low(size_factor):
    paraview.simple._DisableFirstRenderCameraReset()

    # ----------------------------------------------------------------
    # setup views used in the visualization
    # ----------------------------------------------------------------

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
    renderView1.Background2 = [0.11764705882352941, 0.11764705882352941, 0.11764705882352941]
    renderView1.Background = [0.7568627450980392, 0.7568627450980392, 0.7568627450980392]
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

    # create a new 'XML MultiBlock Data Reader'
    coastlinevtm = XMLMultiBlockDataReader(registrationName='coastline.vtm', FileName=['paraview/coastline.vtm'])
    coastlinevtm.CellArrayStatus = ['t50_fid', 'elevation']
    coastlinevtm.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    fevti = XMLImageDataReader(registrationName='fe.vti', FileName=['paraview/fe.vti'])
    fevti.CellArrayStatus = ['poly', 'bin', 'eventcount', 'rate_learning', 'bval', 'lmmin', 'mmin', 'mmax', 'model', 'params', 'rates_bin', 'rate', 'mask']
    fevti.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    basemap_2193vti = XMLImageDataReader(registrationName='basemap_2193.vti', FileName=['paraview/basemap_2193.vti'])
    basemap_2193vti.CellArrayStatus = ['basemap', 'mask']
    basemap_2193vti.TimeArray = 'None'

    # create a new 'XML MultiBlock Data Reader'
    regionvtm = XMLMultiBlockDataReader(registrationName='region.vtm', FileName=['paraview/region.vtm'])
    regionvtm.CellArrayStatus = ['id']
    regionvtm.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    hybridvti = XMLImageDataReader(registrationName='hybrid.vti', FileName=['paraview/hybrid.vti'])
    hybridvti.CellArrayStatus = ['poly', 'bin', 'eventcount', 'rate_learning', 'bval', 'lmmin', 'mmin', 'mmax', 'model', 'params', 'rates_bin', 'rate', 'mask']
    hybridvti.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    npfe_lowvti = XMLImageDataReader(registrationName='npfe_low.vti', FileName=['paraview/npfe_low.vti'])
    npfe_lowvti.CellArrayStatus = ['poly', 'bin', 'eventcount', 'rate_learning', 'bval', 'lmmin', 'mmin', 'mmax', 'model', 'params', 'rates_bin', 'rate', 'mask']
    npfe_lowvti.TimeArray = 'None'

    # create a new 'XML Image Data Reader'
    fe_lowvti = XMLImageDataReader(registrationName='fe_low.vti', FileName=['paraview/fe_low.vti'])
    fe_lowvti.CellArrayStatus = ['poly', 'bin', 'eventcount', 'rate_learning', 'bval', 'lmmin', 'mmin', 'mmax', 'model', 'params', 'rates_bin', 'rate', 'mask']
    fe_lowvti.TimeArray = 'None'

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
    basemapLUT.AutomaticRescaleRangeMode = 'Never'
    basemapLUT.EnableOpacityMapping = 1
    basemapLUT.TransferFunction2D = basemapTF2D
    basemapLUT.RGBPoints = [0.0, 0.8901960784313725, 0.9568627450980393, 0.984313725490196, 7.3264608544966245, 0.807843137254902, 0.9019607843137255, 0.9607843137254902, 14.652921708993299, 0.7607843137254902, 0.8588235294117647, 0.9294117647058824, 29.305843417986598, 0.6862745098039216, 0.7843137254901961, 0.8705882352941177, 43.95876512697984, 0.6, 0.6980392156862745, 0.8, 58.611686835973195, 0.5176470588235295, 0.615686274509804, 0.7254901960784313, 73.26460854496644, 0.44313725490196076, 0.5333333333333333, 0.6431372549019608, 87.91753025395974, 0.37254901960784315, 0.4588235294117647, 0.5686274509803921, 102.57045196295304, 0.3058823529411765, 0.38823529411764707, 0.49411764705882355, 117.22337367194633, 0.24705882352941178, 0.3176470588235294, 0.41568627450980394, 131.87629538093964, 0.2, 0.2549019607843137, 0.34509803921568627, 146.52921708993287, 0.14902, 0.196078, 0.278431, 154.05125062177993, 0.2, 0.1450980392156863, 0.13725490196078433, 161.56886742406763, 0.23137254901960785, 0.17254901960784313, 0.16470588235294117, 169.08648422635534, 0.2549019607843137, 0.2, 0.1843137254901961, 176.60410102864307, 0.28627450980392155, 0.23137254901960785, 0.21176470588235294, 184.12171783093078, 0.3176470588235294, 0.2627450980392157, 0.23921568627450981, 191.63933463321848, 0.34509803921568627, 0.29411764705882354, 0.26666666666666666, 199.1569514355062, 0.37254901960784315, 0.3215686274509804, 0.29411764705882354, 206.67456823779392, 0.403921568627451, 0.3568627450980392, 0.3176470588235294, 214.19218504008163, 0.43137254901960786, 0.38823529411764707, 0.34509803921568627, 221.70980184236933, 0.4627450980392157, 0.4196078431372549, 0.37254901960784315, 229.22741864465706, 0.4980392156862745, 0.4588235294117647, 0.40784313725490196, 236.74503544694477, 0.5372549019607843, 0.5058823529411764, 0.44313725490196076, 244.26265224923253, 0.5803921568627451, 0.5568627450980392, 0.48627450980392156, 251.78026905152018, 0.6313725490196078, 0.6078431372549019, 0.5333333333333333, 259.29788585380794, 0.6784313725490196, 0.6627450980392157, 0.5803921568627451, 266.8155026560956, 0.7254901960784313, 0.7137254901960784, 0.6274509803921569, 274.33311945838335, 0.7764705882352941, 0.7647058823529411, 0.6784313725490196, 281.85073626067106, 0.8274509803921568, 0.8235294117647058, 0.7372549019607844, 289.3683530629588, 0.8901960784313725, 0.8901960784313725, 0.807843137254902, 296.88596986524647, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 296.88596986524647, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 296.8903865948058, 0.9882352941176471, 0.9882352941176471, 0.8588235294117647, 297.17995173347634, 0.984313725490196, 0.9882352941176471, 0.8549019607843137, 304.12951506156867, 0.9882352941176471, 0.9882352941176471, 0.7568627450980392, 311.3686435283316, 0.9803921568627451, 0.9725490196078431, 0.6705882352941176, 318.6077719950945, 0.9803921568627451, 0.9490196078431372, 0.596078431372549, 325.8469004618574, 0.9725490196078431, 0.9176470588235294, 0.5137254901960784, 333.0860289286203, 0.9686274509803922, 0.8784313725490196, 0.4235294117647059, 340.32515739538314, 0.9647058823529412, 0.8274509803921568, 0.3254901960784314, 347.56428586214605, 0.9529411764705882, 0.7686274509803922, 0.24705882352941178, 354.80341432890896, 0.9450980392156862, 0.7098039215686275, 0.1843137254901961, 362.04254279567186, 0.9372549019607843, 0.6431372549019608, 0.12156862745098039, 369.28167126243477, 0.9254901960784314, 0.5803921568627451, 0.0784313725490196, 376.5207997291976, 0.9176470588235294, 0.5215686274509804, 0.047058823529411764, 383.7599281959606, 0.9019607843137255, 0.4588235294117647, 0.027450980392156862, 390.99905666272343, 0.8627450980392157, 0.3803921568627451, 0.011764705882352941, 398.23818512948634, 0.788235294117647, 0.2901960784313726, 0.0, 405.47731359624925, 0.7411764705882353, 0.22745098039215686, 0.00392156862745098, 412.7164420630121, 0.6627450980392157, 0.16862745098039217, 0.01568627450980392, 419.955570529775, 0.5882352941176471, 0.11372549019607843, 0.023529411764705882, 427.19469899653797, 0.49411764705882355, 0.054901960784313725, 0.03529411764705882, 434.4338274633009, 0.396078431372549, 0.0392156862745098, 0.058823529411764705, 441.6729559300637, 0.301961, 0.047059, 0.090196]
    basemapLUT.ColorSpace = 'Lab'
    basemapLUT.NanColor = [0.25, 0.0, 0.0]
    basemapLUT.NanOpacity = 0.0
    basemapLUT.NumberOfTableValues = 10
    basemapLUT.ScalarRangeInitialized = 1.0

    # get opacity transfer function/opacity map for 'basemap'
    basemapPWF = GetOpacityTransferFunction('basemap')
    basemapPWF.Points = [0.0, 0.0, 0.5, 0.0, 6.713812823552561, 0.5955882668495178, 0.5, 0.0, 40.76243438313056, 0.625, 0.5, 0.0, 44.598900499092004, 0.8897058963775635, 0.5, 0.0, 99.74807952635801, 0.9705882668495178, 0.5, 0.0, 441.6729559300637, 1.0, 0.5, 0.0]
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

    # init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
    basemap_2193vtiDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    basemap_2193vtiDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    basemap_2193vtiDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

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
    coastlinevtmDisplay.Position = [0.0, 0.0, 200.0]
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

    # init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
    coastlinevtmDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    coastlinevtmDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    coastlinevtmDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PolarAxesRepresentation' selected for 'PolarAxes'
    coastlinevtmDisplay.PolarAxes.Translation = [0.0, 0.0, 200.0]

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
    regionvtmDisplay.Position = [0.0, 0.0, 200.0]
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

    # init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
    regionvtmDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    regionvtmDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    regionvtmDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PolarAxesRepresentation' selected for 'PolarAxes'
    regionvtmDisplay.PolarAxes.Translation = [0.0, 0.0, 200.0]

    # show data from fe_lowvti
    fe_lowvtiDisplay = Show(fe_lowvti, renderView1, 'UniformGridRepresentation')

    # get 2D transfer function for 'rate'
    rateTF2D = GetTransferFunction2D('rate')
    rateTF2D.ScalarRangeInitialized = 1
    rateTF2D.Range = [4.9999999999999996e-06, 0.001, 0.0, 1.0]

    # get color transfer function/color map for 'rate'
    rateLUT = GetColorTransferFunction('rate')
    rateLUT.AutomaticRescaleRangeMode = 'Never'
    rateLUT.TransferFunction2D = rateTF2D
    rateLUT.RGBPoints = [4.9999999999999996e-06, 0.02, 0.3813, 0.9981, 5.822593385447654e-06, 0.02000006, 0.424267768, 0.96906969, 6.780518746451757e-06, 0.02, 0.467233763, 0.940033043, 7.896040720598767e-06, 0.02, 0.5102, 0.911, 9.195086894196724e-06, 0.02000006, 0.546401494, 0.872669438, 1.0707850425753277e-05, 0.02, 0.582600362, 0.83433295, 1.2469491812270778e-05, 0.02, 0.6188, 0.796, 1.4520956109204277e-05, 0.02000006, 0.652535156, 0.749802434, 1.690992459836571e-05, 0.02, 0.686267004, 0.703599538, 1.9691923022972557e-05, 0.02, 0.72, 0.6574, 2.293161214806088e-05, 0.02000006, 0.757035456, 0.603735359, 2.6704290642190078e-05, 0.02, 0.794067037, 0.55006613, 3.1097645211257534e-05, 0.02, 0.8311, 0.4964, 3.621378866201322e-05, 0.021354336738172372, 0.8645368555261631, 0.4285579460761159, 4.217163326508741e-05, 0.023312914349117714, 0.897999359924484, 0.36073871343115577, 4.910965458056455e-05, 0.015976108242848862, 0.9310479513349017, 0.2925631815088092, 5.718910998448275e-05, 0.27421074700988196, 0.952562960995083, 0.15356836602739213, 6.659778670305754e-05, 0.4933546281681699, 0.9619038625309482, 0.11119493614749336, 7.755436646853534e-05, 0.6439, 0.9773, 0.0469, 9.031350824245546e-05, 0.762401813, 0.984669591, 0.034600153, 0.0001051717671418187, 0.880901185, 0.992033407, 0.022299877, 0.00012247448713915892, 0.9995285432627147, 0.9995193706781492, 0.0134884641450013, 0.00014262382774051213, 0.999402998, 0.955036376, 0.079066628, 0.00016608811120182637, 0.9994, 0.910666223, 0.148134024, 0.0001934127075370494, 0.9994, 0.8663, 0.2172, 0.00022523271031334936, 0.999269665, 0.818035981, 0.217200652, 0.0002622876978513912, 0.999133332, 0.769766184, 0.2172, 0.00030543892291876073, 0.999, 0.7215, 0.2172, 0.0003556893304490061, 0.99913633, 0.673435546, 0.217200652, 0.0004142068685493377, 0.999266668, 0.625366186, 0.2172, 0.000482351634604472, 0.9994, 0.5773, 0.2172, 0.0005617074874215727, 0.999402998, 0.521068455, 0.217200652, 0.0006541188601634542, 0.9994, 0.464832771, 0.2172, 0.0007617336296968577, 0.9994, 0.4086, 0.2172, 0.0008870530387491915, 0.9947599917687346, 0.33177297300202935, 0.2112309638520206, 0.001032989831192457, 0.9867129505479589, 0.2595183410914934, 0.19012239549291934, 0.0012029359516671756, 0.9912458875646419, 0.14799417507952672, 0.21078892136920357, 0.001400841383058897, 0.949903037, 0.116867171, 0.252900603, 0.0016313059542120158, 0.903199533, 0.078432949, 0.291800389, 0.001899686251727252, 0.8565, 0.04, 0.3307, 0.002212220120746587, 0.798902627, 0.04333345, 0.358434298, 0.0025761716484426584, 0.741299424, 0.0466667, 0.386166944, 0.003000000000000001, 0.6837, 0.05, 0.4139]
    rateLUT.UseLogScale = 1
    rateLUT.ColorSpace = 'RGB'
    rateLUT.NanColor = [1.0, 0.0, 0.0]
    rateLUT.NanOpacity = 0.0
    rateLUT.NumberOfTableValues = 269
    rateLUT.ScalarRangeInitialized = 1.0

    # get opacity transfer function/opacity map for 'rate'
    ratePWF = GetOpacityTransferFunction('rate')
    ratePWF.Points = [4.9999999999999996e-06, 0.0, 0.5, 0.0, 5.052660319488496e-05, 0.5955882668495178, 0.5, 0.0, 0.0002814115152135492, 0.625, 0.5, 0.0, 0.0003074267281964422, 0.8897058963775635, 0.5, 0.0, 0.0006813952697813511, 0.9705882668495178, 0.5, 0.0, 0.003, 1.0, 0.5, 0.0]
    ratePWF.ScalarRangeInitialized = 1

    # trace defaults for the display properties.
    fe_lowvtiDisplay.Representation = 'Slice'
    fe_lowvtiDisplay.ColorArrayName = ['CELLS', 'rate']
    fe_lowvtiDisplay.LookupTable = rateLUT
    fe_lowvtiDisplay.SelectTCoordArray = 'None'
    fe_lowvtiDisplay.SelectNormalArray = 'None'
    fe_lowvtiDisplay.SelectTangentArray = 'None'
    fe_lowvtiDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    fe_lowvtiDisplay.SelectOrientationVectors = 'None'
    fe_lowvtiDisplay.ScaleFactor = 154350.0
    fe_lowvtiDisplay.SelectScaleArray = 'None'
    fe_lowvtiDisplay.GlyphType = 'Arrow'
    fe_lowvtiDisplay.GlyphTableIndexArray = 'None'
    fe_lowvtiDisplay.GaussianRadius = 7717.5
    fe_lowvtiDisplay.SetScaleArray = [None, '']
    fe_lowvtiDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    fe_lowvtiDisplay.OpacityArray = [None, '']
    fe_lowvtiDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    fe_lowvtiDisplay.DataAxesGrid = 'GridAxesRepresentation'
    fe_lowvtiDisplay.PolarAxes = 'PolarAxesRepresentation'
    fe_lowvtiDisplay.ScalarOpacityUnitDistance = 20818.636043244984
    fe_lowvtiDisplay.ScalarOpacityFunction = ratePWF
    fe_lowvtiDisplay.TransferFunction2D = rateTF2D
    fe_lowvtiDisplay.OpacityArrayName = ['CELLS', 'bin']
    fe_lowvtiDisplay.ColorArray2Name = ['CELLS', 'bin']
    fe_lowvtiDisplay.SliceFunction = 'Plane'
    fe_lowvtiDisplay.SelectInputVectors = [None, '']
    fe_lowvtiDisplay.WriteLog = ''

    # init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
    fe_lowvtiDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    fe_lowvtiDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    fe_lowvtiDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'Plane' selected for 'SliceFunction'
    fe_lowvtiDisplay.SliceFunction.Origin = [1589946.015518637, 5470136.7749801995, 10.0]

    # show data from fevti
    fevtiDisplay = Show(fevti, renderView1, 'UniformGridRepresentation')

    # trace defaults for the display properties.
    fevtiDisplay.Representation = 'Slice'
    fevtiDisplay.ColorArrayName = ['CELLS', 'rate']
    fevtiDisplay.LookupTable = rateLUT
    fevtiDisplay.SelectTCoordArray = 'None'
    fevtiDisplay.SelectNormalArray = 'None'
    fevtiDisplay.SelectTangentArray = 'None'
    fevtiDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    fevtiDisplay.SelectOrientationVectors = 'None'
    fevtiDisplay.ScaleFactor = 154350.0
    fevtiDisplay.SelectScaleArray = 'None'
    fevtiDisplay.GlyphType = 'Arrow'
    fevtiDisplay.GlyphTableIndexArray = 'None'
    fevtiDisplay.GaussianRadius = 7717.5
    fevtiDisplay.SetScaleArray = [None, '']
    fevtiDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    fevtiDisplay.OpacityArray = [None, '']
    fevtiDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    fevtiDisplay.DataAxesGrid = 'GridAxesRepresentation'
    fevtiDisplay.PolarAxes = 'PolarAxesRepresentation'
    fevtiDisplay.ScalarOpacityUnitDistance = 20818.636043244984
    fevtiDisplay.ScalarOpacityFunction = ratePWF
    fevtiDisplay.TransferFunction2D = rateTF2D
    fevtiDisplay.OpacityArrayName = ['CELLS', 'bin']
    fevtiDisplay.ColorArray2Name = ['CELLS', 'bin']
    fevtiDisplay.SliceFunction = 'Plane'
    fevtiDisplay.SelectInputVectors = [None, '']
    fevtiDisplay.WriteLog = ''

    # init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
    fevtiDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    fevtiDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    fevtiDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'Plane' selected for 'SliceFunction'
    fevtiDisplay.SliceFunction.Origin = [1589946.015518637, 5470136.7749801995, 10.0]

    # show data from hybridvti
    hybridvtiDisplay = Show(hybridvti, renderView1, 'UniformGridRepresentation')

    # trace defaults for the display properties.
    hybridvtiDisplay.Representation = 'Slice'
    hybridvtiDisplay.ColorArrayName = ['CELLS', 'rate']
    hybridvtiDisplay.LookupTable = rateLUT
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

    # init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
    hybridvtiDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    hybridvtiDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    hybridvtiDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'Plane' selected for 'SliceFunction'
    hybridvtiDisplay.SliceFunction.Origin = [1589946.015518637, 5470136.7749801995, 10.0]

    # show data from npfe_lowvti
    npfe_lowvtiDisplay = Show(npfe_lowvti, renderView1, 'UniformGridRepresentation')

    # get 2D transfer function for 'params'
    paramsTF2D = GetTransferFunction2D('params')
    paramsTF2D.ScalarRangeInitialized = 1
    paramsTF2D.Range = [5e-06, 0.001, 0.0, 1.0]

    # get color transfer function/color map for 'params'
    paramsLUT = GetColorTransferFunction('params')
    paramsLUT.AutomaticRescaleRangeMode = 'Never'
    paramsLUT.TransferFunction2D = paramsTF2D
    paramsLUT.RGBPoints = [4.9999999999999996e-06, 0.02, 0.3813, 0.9981, 5.67226396226752e-06, 0.02000006, 0.424267768, 0.96906969, 6.434915691527779e-06, 0.02, 0.467233763, 0.940033043, 7.3001080754565605e-06, 0.02, 0.5102, 0.911, 8.281627991414054e-06, 0.02000006, 0.546401494, 0.872669438, 9.395116000920818e-06, 0.02, 0.582600362, 0.83433295, 1.065831558266922e-05, 0.02, 0.6188, 0.796, 1.2091355875609768e-05, 0.02000006, 0.652535156, 0.749802434, 1.3717072437634615e-05, 0.02, 0.686267004, 0.703599538, 1.5561371131161585e-05, 0.02, 0.72, 0.6574, 1.7653640934151603e-05, 0.02000006, 0.757035456, 0.603735359, 2.0027222254719816e-05, 0.02, 0.794067037, 0.55006613, 2.2719938211953857e-05, 0.02, 0.8311, 0.4964, 2.577469734892213e-05, 0.021354336738172372, 0.8645368555261631, 0.4285579460761159, 2.9240177382128637e-05, 0.023312914349117714, 0.897999359924484, 0.36073871343115577, 3.317160088299169e-05, 0.015976108242848862, 0.9310479513349017, 0.2925631815088092, 3.763161525186305e-05, 0.27421074700988196, 0.952562960995083, 0.15356836602739213, 4.269129100701191e-05, 0.4933546281681699, 0.9619038625309482, 0.11119493614749336, 4.843125429634984e-05, 0.6439, 0.9773, 0.0469, 5.494297167851996e-05, 0.762401813, 0.984669591, 0.034600153, 6.233020764639076e-05, 0.880901185, 0.992033407, 0.022299877, 7.071067811865475e-05, 0.9995285432627147, 0.9995193706781492, 0.0134884641450013, 8.021792624798877e-05, 0.999402998, 0.955036376, 0.079066628, 9.10034504368601e-05, 0.9994, 0.910666223, 0.148134024, 0.00010323911847100001, 0.9994, 0.8663, 0.2172, 0.00011711990623986435, 0.999269665, 0.818035981, 0.217200652, 0.00013286700468570668, 0.999133332, 0.769766184, 0.2172, 0.00015073134449063294, 0.999, 0.7215, 0.2172, 0.00017099759466766945, 0.99913633, 0.673435546, 0.217200652, 0.00019398869877357026, 0.999266668, 0.625366186, 0.2172, 0.00022007102102809866, 0.9994, 0.5773, 0.2172, 0.0002496601843434203, 0.999402998, 0.521068455, 0.217200652, 0.0002832276932928504, 0.9994, 0.464832771, 0.2172, 0.0003213084475562387, 0.9994, 0.4086, 0.2172, 0.0003645092655690757, 0.9947599917687346, 0.33177297300202935, 0.2112309638520206, 0.0004135185542000143, 0.9867129505479589, 0.2595183410914934, 0.19012239549291934, 0.00046911727854354095, 0.9912458875646419, 0.14799417507952672, 0.21078892136920357, 0.0005321914066319095, 0.949903037, 0.116867171, 0.252900603, 0.000603746027373328, 0.903199533, 0.078432949, 0.291800389, 0.0006849213666863824, 0.8565, 0.04, 0.3307, 0.000777010957048437, 0.798902627, 0.04333345, 0.358434298, 0.0008814822499905691, 0.741299424, 0.0466667, 0.386166944, 0.001, 0.6837, 0.05, 0.4139]
    paramsLUT.UseLogScale = 1
    paramsLUT.ColorSpace = 'RGB'
    paramsLUT.NanColor = [1.0, 0.0, 0.0]
    paramsLUT.NanOpacity = 0.0
    paramsLUT.NumberOfTableValues = 216
    paramsLUT.ScalarRangeInitialized = 1.0
    paramsLUT.VectorMode = 'Component'

    # get opacity transfer function/opacity map for 'params'
    paramsPWF = GetOpacityTransferFunction('params')
    paramsPWF.Points = [5e-06, 0.0, 0.5, 0.0, 5.052660319488497e-05, 0.5955882668495178, 0.5, 0.0, 0.0002814115152135492, 0.625, 0.5, 0.0, 0.00030742672819644213, 0.8897058963775635, 0.5, 0.0, 0.0006813952697813511, 0.9705882668495178, 0.5, 0.0, 0.003, 1.0, 0.5, 0.0]
    paramsPWF.ScalarRangeInitialized = 1

    # trace defaults for the display properties.
    npfe_lowvtiDisplay.Representation = 'Slice'
    npfe_lowvtiDisplay.ColorArrayName = ['CELLS', 'params']
    npfe_lowvtiDisplay.LookupTable = paramsLUT
    npfe_lowvtiDisplay.SelectTCoordArray = 'None'
    npfe_lowvtiDisplay.SelectNormalArray = 'None'
    npfe_lowvtiDisplay.SelectTangentArray = 'None'
    npfe_lowvtiDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    npfe_lowvtiDisplay.SelectOrientationVectors = 'None'
    npfe_lowvtiDisplay.ScaleFactor = 154400.0
    npfe_lowvtiDisplay.SelectScaleArray = 'None'
    npfe_lowvtiDisplay.GlyphType = 'Arrow'
    npfe_lowvtiDisplay.GlyphTableIndexArray = 'None'
    npfe_lowvtiDisplay.GaussianRadius = 7720.0
    npfe_lowvtiDisplay.SetScaleArray = [None, '']
    npfe_lowvtiDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    npfe_lowvtiDisplay.OpacityArray = [None, '']
    npfe_lowvtiDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    npfe_lowvtiDisplay.DataAxesGrid = 'GridAxesRepresentation'
    npfe_lowvtiDisplay.PolarAxes = 'PolarAxesRepresentation'
    npfe_lowvtiDisplay.ScalarOpacityUnitDistance = 15889.26963058965
    npfe_lowvtiDisplay.ScalarOpacityFunction = paramsPWF
    npfe_lowvtiDisplay.TransferFunction2D = paramsTF2D
    npfe_lowvtiDisplay.OpacityArrayName = ['CELLS', 'bin']
    npfe_lowvtiDisplay.ColorArray2Name = ['CELLS', 'bin']
    npfe_lowvtiDisplay.SliceFunction = 'Plane'
    npfe_lowvtiDisplay.SelectInputVectors = [None, '']
    npfe_lowvtiDisplay.WriteLog = ''

    # init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
    npfe_lowvtiDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    npfe_lowvtiDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    npfe_lowvtiDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # init the 'Plane' selected for 'SliceFunction'
    npfe_lowvtiDisplay.SliceFunction.Origin = [1589696.015518637, 5470386.7749801995, 10.0]

    # setup the color legend parameters for each legend in this view

    # get 2D transfer function for 'mask'
    maskTF2D = GetTransferFunction2D('mask')

    # get color transfer function/color map for 'mask'
    maskLUT = GetColorTransferFunction('mask')
    maskLUT.AutomaticRescaleRangeMode = 'Never'
    maskLUT.EnableOpacityMapping = 1
    maskLUT.TransferFunction2D = maskTF2D
    maskLUT.RGBPoints = [0.0, 0.8901960784313725, 0.9568627450980393, 0.984313725490196, 0.016587977045297575, 0.807843137254902, 0.9019607843137255, 0.9607843137254902, 0.03317595409059526, 0.7607843137254902, 0.8588235294117647, 0.9294117647058824, 0.06635190818119052, 0.6862745098039216, 0.7843137254901961, 0.8705882352941177, 0.09952786227178567, 0.6, 0.6980392156862745, 0.8, 0.13270381636238104, 0.5176470588235295, 0.615686274509804, 0.7254901960784313, 0.1658797704529762, 0.44313725490196076, 0.5333333333333333, 0.6431372549019608, 0.19905572454357146, 0.37254901960784315, 0.4588235294117647, 0.5686274509803921, 0.23223167863416672, 0.3058823529411765, 0.38823529411764707, 0.49411764705882355, 0.265407632724762, 0.24705882352941178, 0.3176470588235294, 0.41568627450980394, 0.29858358681535724, 0.2, 0.2549019607843137, 0.34509803921568627, 0.3317595409059524, 0.14902, 0.196078, 0.278431, 0.348790317707777, 0.2, 0.1450980392156863, 0.13725490196078433, 0.3658110945096016, 0.23137254901960785, 0.17254901960784313, 0.16470588235294117, 0.38283187131142615, 0.2549019607843137, 0.2, 0.1843137254901961, 0.3998526481132507, 0.28627450980392155, 0.23137254901960785, 0.21176470588235294, 0.4168734249150753, 0.3176470588235294, 0.2627450980392157, 0.23921568627450981, 0.43389420171689985, 0.34509803921568627, 0.29411764705882354, 0.26666666666666666, 0.4509149785187244, 0.37254901960784315, 0.3215686274509804, 0.29411764705882354, 0.467935755320549, 0.403921568627451, 0.3568627450980392, 0.3176470588235294, 0.48495653212237355, 0.43137254901960786, 0.38823529411764707, 0.34509803921568627, 0.5019773089241981, 0.4627450980392157, 0.4196078431372549, 0.37254901960784315, 0.5189980857260227, 0.4980392156862745, 0.4588235294117647, 0.40784313725490196, 0.5360188625278473, 0.5372549019607843, 0.5058823529411764, 0.44313725490196076, 0.5530396393296719, 0.5803921568627451, 0.5568627450980392, 0.48627450980392156, 0.5700604161314964, 0.6313725490196078, 0.6078431372549019, 0.5333333333333333, 0.5870811929333211, 0.6784313725490196, 0.6627450980392157, 0.5803921568627451, 0.6041019697351455, 0.7254901960784313, 0.7137254901960784, 0.6274509803921569, 0.6211227465369702, 0.7764705882352941, 0.7647058823529411, 0.6784313725490196, 0.6381435233387946, 0.8274509803921568, 0.8235294117647058, 0.7372549019607844, 0.6551643001406193, 0.8901960784313725, 0.8901960784313725, 0.807843137254902, 0.6721850769424438, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 0.6721850769424438, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 0.6721950769424438, 0.9882352941176471, 0.9882352941176471, 0.8588235294117647, 0.672850686788559, 0.984313725490196, 0.9882352941176471, 0.8549019607843137, 0.6885853230953216, 0.9882352941176471, 0.9882352941176471, 0.7568627450980392, 0.7049755692481995, 0.9803921568627451, 0.9725490196078431, 0.6705882352941176, 0.7213658154010772, 0.9803921568627451, 0.9490196078431372, 0.596078431372549, 0.7377560615539551, 0.9725490196078431, 0.9176470588235294, 0.5137254901960784, 0.7541463077068329, 0.9686274509803922, 0.8784313725490196, 0.4235294117647059, 0.7705365538597106, 0.9647058823529412, 0.8274509803921568, 0.3254901960784314, 0.7869268000125885, 0.9529411764705882, 0.7686274509803922, 0.24705882352941178, 0.8033170461654663, 0.9450980392156862, 0.7098039215686275, 0.1843137254901961, 0.8197072923183442, 0.9372549019607843, 0.6431372549019608, 0.12156862745098039, 0.8360975384712219, 0.9254901960784314, 0.5803921568627451, 0.0784313725490196, 0.8524877846240997, 0.9176470588235294, 0.5215686274509804, 0.047058823529411764, 0.8688780307769776, 0.9019607843137255, 0.4588235294117647, 0.027450980392156862, 0.8852682769298553, 0.8627450980392157, 0.3803921568627451, 0.011764705882352941, 0.9016585230827332, 0.788235294117647, 0.2901960784313726, 0.0, 0.918048769235611, 0.7411764705882353, 0.22745098039215686, 0.00392156862745098, 0.9344390153884887, 0.6627450980392157, 0.16862745098039217, 0.01568627450980392, 0.9508292615413665, 0.5882352941176471, 0.11372549019607843, 0.023529411764705882, 0.9672195076942445, 0.49411764705882355, 0.054901960784313725, 0.03529411764705882, 0.9836097538471222, 0.396078431372549, 0.0392156862745098, 0.058823529411764705, 1.0, 0.301961, 0.047059, 0.090196]
    maskLUT.ColorSpace = 'Lab'
    maskLUT.NanColor = [0.25, 0.0, 0.0]
    maskLUT.NanOpacity = 0.0
    maskLUT.NumberOfTableValues = 10
    maskLUT.ScalarRangeInitialized = 1.0

    # get color legend/bar for maskLUT in view renderView1
    maskLUTColorBar = GetScalarBar(maskLUT, renderView1)
    maskLUTColorBar.WindowLocation = 'Any Location'
    maskLUTColorBar.Title = 'mask'
    maskLUTColorBar.ComponentTitle = ''
    maskLUTColorBar.HorizontalTitle = 1
    maskLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    maskLUTColorBar.TitleFontSize = 24
    maskLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    maskLUTColorBar.LabelFormat = '%.4f'
    maskLUTColorBar.RangeLabelFormat = '%.3f'

    # set color bar visibility
    maskLUTColorBar.Visibility = 0

    # get 2D transfer function for 'vtkBlockColors'
    vtkBlockColorsTF2D = GetTransferFunction2D('vtkBlockColors')

    # get color transfer function/color map for 'vtkBlockColors'
    vtkBlockColorsLUT = GetColorTransferFunction('vtkBlockColors')
    vtkBlockColorsLUT.AutomaticRescaleRangeMode = 'Never'
    vtkBlockColorsLUT.InterpretValuesAsCategories = 1
    vtkBlockColorsLUT.AnnotationsInitialized = 1
    vtkBlockColorsLUT.EnableOpacityMapping = 1
    vtkBlockColorsLUT.TransferFunction2D = vtkBlockColorsTF2D
    vtkBlockColorsLUT.RGBPoints = [0.0, 0.8901960784313725, 0.9568627450980393, 0.984313725490196, 0.016587977045297575, 0.807843137254902, 0.9019607843137255, 0.9607843137254902, 0.03317595409059526, 0.7607843137254902, 0.8588235294117647, 0.9294117647058824, 0.06635190818119052, 0.6862745098039216, 0.7843137254901961, 0.8705882352941177, 0.09952786227178567, 0.6, 0.6980392156862745, 0.8, 0.13270381636238104, 0.5176470588235295, 0.615686274509804, 0.7254901960784313, 0.1658797704529762, 0.44313725490196076, 0.5333333333333333, 0.6431372549019608, 0.19905572454357146, 0.37254901960784315, 0.4588235294117647, 0.5686274509803921, 0.23223167863416672, 0.3058823529411765, 0.38823529411764707, 0.49411764705882355, 0.265407632724762, 0.24705882352941178, 0.3176470588235294, 0.41568627450980394, 0.29858358681535724, 0.2, 0.2549019607843137, 0.34509803921568627, 0.3317595409059524, 0.14902, 0.196078, 0.278431, 0.348790317707777, 0.2, 0.1450980392156863, 0.13725490196078433, 0.3658110945096016, 0.23137254901960785, 0.17254901960784313, 0.16470588235294117, 0.38283187131142615, 0.2549019607843137, 0.2, 0.1843137254901961, 0.3998526481132507, 0.28627450980392155, 0.23137254901960785, 0.21176470588235294, 0.4168734249150753, 0.3176470588235294, 0.2627450980392157, 0.23921568627450981, 0.43389420171689985, 0.34509803921568627, 0.29411764705882354, 0.26666666666666666, 0.4509149785187244, 0.37254901960784315, 0.3215686274509804, 0.29411764705882354, 0.467935755320549, 0.403921568627451, 0.3568627450980392, 0.3176470588235294, 0.48495653212237355, 0.43137254901960786, 0.38823529411764707, 0.34509803921568627, 0.5019773089241981, 0.4627450980392157, 0.4196078431372549, 0.37254901960784315, 0.5189980857260227, 0.4980392156862745, 0.4588235294117647, 0.40784313725490196, 0.5360188625278473, 0.5372549019607843, 0.5058823529411764, 0.44313725490196076, 0.5530396393296719, 0.5803921568627451, 0.5568627450980392, 0.48627450980392156, 0.5700604161314964, 0.6313725490196078, 0.6078431372549019, 0.5333333333333333, 0.5870811929333211, 0.6784313725490196, 0.6627450980392157, 0.5803921568627451, 0.6041019697351455, 0.7254901960784313, 0.7137254901960784, 0.6274509803921569, 0.6211227465369702, 0.7764705882352941, 0.7647058823529411, 0.6784313725490196, 0.6381435233387946, 0.8274509803921568, 0.8235294117647058, 0.7372549019607844, 0.6551643001406193, 0.8901960784313725, 0.8901960784313725, 0.807843137254902, 0.6721850769424438, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 0.6721850769424438, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 0.6721950769424438, 0.9882352941176471, 0.9882352941176471, 0.8588235294117647, 0.672850686788559, 0.984313725490196, 0.9882352941176471, 0.8549019607843137, 0.6885853230953216, 0.9882352941176471, 0.9882352941176471, 0.7568627450980392, 0.7049755692481995, 0.9803921568627451, 0.9725490196078431, 0.6705882352941176, 0.7213658154010772, 0.9803921568627451, 0.9490196078431372, 0.596078431372549, 0.7377560615539551, 0.9725490196078431, 0.9176470588235294, 0.5137254901960784, 0.7541463077068329, 0.9686274509803922, 0.8784313725490196, 0.4235294117647059, 0.7705365538597106, 0.9647058823529412, 0.8274509803921568, 0.3254901960784314, 0.7869268000125885, 0.9529411764705882, 0.7686274509803922, 0.24705882352941178, 0.8033170461654663, 0.9450980392156862, 0.7098039215686275, 0.1843137254901961, 0.8197072923183442, 0.9372549019607843, 0.6431372549019608, 0.12156862745098039, 0.8360975384712219, 0.9254901960784314, 0.5803921568627451, 0.0784313725490196, 0.8524877846240997, 0.9176470588235294, 0.5215686274509804, 0.047058823529411764, 0.8688780307769776, 0.9019607843137255, 0.4588235294117647, 0.027450980392156862, 0.8852682769298553, 0.8627450980392157, 0.3803921568627451, 0.011764705882352941, 0.9016585230827332, 0.788235294117647, 0.2901960784313726, 0.0, 0.918048769235611, 0.7411764705882353, 0.22745098039215686, 0.00392156862745098, 0.9344390153884887, 0.6627450980392157, 0.16862745098039217, 0.01568627450980392, 0.9508292615413665, 0.5882352941176471, 0.11372549019607843, 0.023529411764705882, 0.9672195076942445, 0.49411764705882355, 0.054901960784313725, 0.03529411764705882, 0.9836097538471222, 0.396078431372549, 0.0392156862745098, 0.058823529411764705, 1.0, 0.301961, 0.047059, 0.090196]
    vtkBlockColorsLUT.ColorSpace = 'Lab'
    vtkBlockColorsLUT.NanColor = [0.25, 0.0, 0.0]
    vtkBlockColorsLUT.NanOpacity = 0.0
    vtkBlockColorsLUT.NumberOfTableValues = 10
    vtkBlockColorsLUT.Annotations = ['0', '0', '1', '1', '2', '2', '3', '3', '4', '4', '5', '5', '6', '6', '7', '7', '8', '8', '9', '9', '10', '10', '11', '11']
    vtkBlockColorsLUT.IndexedColors = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.63, 0.63, 1.0, 0.67, 0.5, 0.33, 1.0, 0.5, 0.75, 0.53, 0.35, 0.7, 1.0, 0.75, 0.5]

    # get color legend/bar for vtkBlockColorsLUT in view renderView1
    vtkBlockColorsLUTColorBar = GetScalarBar(vtkBlockColorsLUT, renderView1)
    vtkBlockColorsLUTColorBar.WindowLocation = 'Any Location'
    vtkBlockColorsLUTColorBar.Title = 'vtkBlockColors'
    vtkBlockColorsLUTColorBar.ComponentTitle = ''
    vtkBlockColorsLUTColorBar.HorizontalTitle = 1
    vtkBlockColorsLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    vtkBlockColorsLUTColorBar.TitleFontSize = 24
    vtkBlockColorsLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    vtkBlockColorsLUTColorBar.LabelFormat = '%.4f'
    vtkBlockColorsLUTColorBar.RangeLabelFormat = '%.3f'

    # set color bar visibility
    vtkBlockColorsLUTColorBar.Visibility = 0

    # get 2D transfer function for 'bin'
    binTF2D = GetTransferFunction2D('bin')
    binTF2D.ScalarRangeInitialized = 1
    binTF2D.Range = [0.0, 3.0, 0.0, 1.0]

    # get color transfer function/color map for 'bin'
    binLUT = GetColorTransferFunction('bin')
    binLUT.AutomaticRescaleRangeMode = 'Never'
    binLUT.EnableOpacityMapping = 1
    binLUT.TransferFunction2D = binTF2D
    binLUT.RGBPoints = [0.0, 0.8901960784313725, 0.9568627450980393, 0.984313725490196, 0.049763931137022155, 0.807843137254902, 0.9019607843137255, 0.9607843137254902, 0.09952786227131583, 0.7607843137254902, 0.8588235294117647, 0.9294117647058824, 0.19905572454263165, 0.6862745098039216, 0.7843137254901961, 0.8705882352941177, 0.29858358681667596, 0.6, 0.6980392156862745, 0.8, 0.3981114490879918, 0.5176470588235295, 0.615686274509804, 0.7254901960784313, 0.4976393113593076, 0.44313725490196076, 0.5333333333333333, 0.6431372549019608, 0.5971671736306234, 0.37254901960784315, 0.4588235294117647, 0.5686274509803921, 0.6966950359019393, 0.3058823529411765, 0.38823529411764707, 0.49411764705882355, 0.7962228981732551, 0.24705882352941178, 0.3176470588235294, 0.41568627450980394, 0.8957507604472994, 0.2, 0.2549019607843137, 0.34509803921568627, 0.9952786227186152, 0.14902, 0.196078, 0.278431, 1.0463709531222776, 0.2, 0.1450980392156863, 0.13725490196078433, 1.0974332835294263, 0.23137254901960785, 0.17254901960784313, 0.16470588235294117, 1.1484956139338465, 0.2549019607843137, 0.2, 0.1843137254901961, 1.1995579443409952, 0.28627450980392155, 0.23137254901960785, 0.21176470588235294, 1.2506202747454154, 0.3176470588235294, 0.2627450980392157, 0.23921568627450981, 1.3016826051498356, 0.34509803921568627, 0.29411764705882354, 0.26666666666666666, 1.3527449355569843, 0.37254901960784315, 0.3215686274509804, 0.29411764705882354, 1.4038072659614045, 0.403921568627451, 0.3568627450980392, 0.3176470588235294, 1.4548695963658247, 0.43137254901960786, 0.38823529411764707, 0.34509803921568627, 1.5059319267729734, 0.4627450980392157, 0.4196078431372549, 0.37254901960784315, 1.5569942571773936, 0.4980392156862745, 0.4588235294117647, 0.40784313725490196, 1.6080565875845423, 0.5372549019607843, 0.5058823529411764, 0.44313725490196076, 1.6591189179889625, 0.5803921568627451, 0.5568627450980392, 0.48627450980392156, 1.7101812483933827, 0.6313725490196078, 0.6078431372549019, 0.5333333333333333, 1.7612435788005314, 0.6784313725490196, 0.6627450980392157, 0.5803921568627451, 1.8123059092049516, 0.7254901960784313, 0.7137254901960784, 0.6274509803921569, 1.8633682396121003, 0.7764705882352941, 0.7647058823529411, 0.6784313725490196, 1.9144305700165205, 0.8274509803921568, 0.8235294117647058, 0.7372549019607844, 1.9654929004209407, 0.8901960784313725, 0.8901960784313725, 0.807843137254902, 2.0165552308280894, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 2.0165552308280894, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 2.0165852308273315, 0.9882352941176471, 0.9882352941176471, 0.8588235294117647, 2.0185520603654368, 0.984313725490196, 0.9882352941176471, 0.8549019607843137, 2.0657559692854193, 0.9882352941176471, 0.9882352941176471, 0.7568627450980392, 2.114926707743507, 0.9803921568627451, 0.9725490196078431, 0.6705882352941176, 2.164097446204323, 0.9803921568627451, 0.9490196078431372, 0.596078431372549, 2.213268184662411, 0.9725490196078431, 0.9176470588235294, 0.5137254901960784, 2.2624389231204987, 0.9686274509803922, 0.8784313725490196, 0.4235294117647059, 2.3116096615785864, 0.9647058823529412, 0.8274509803921568, 0.3254901960784314, 2.360780400036674, 0.9529411764705882, 0.7686274509803922, 0.24705882352941178, 2.4099511384974903, 0.9450980392156862, 0.7098039215686275, 0.1843137254901961, 2.459121876955578, 0.9372549019607843, 0.6431372549019608, 0.12156862745098039, 2.5082926154136658, 0.9254901960784314, 0.5803921568627451, 0.0784313725490196, 2.5574633538717535, 0.9176470588235294, 0.5215686274509804, 0.047058823529411764, 2.6066340923298412, 0.9019607843137255, 0.4588235294117647, 0.027450980392156862, 2.6558048307906574, 0.8627450980392157, 0.3803921568627451, 0.011764705882352941, 2.704975569248745, 0.788235294117647, 0.2901960784313726, 0.0, 2.754146307706833, 0.7411764705882353, 0.22745098039215686, 0.00392156862745098, 2.8033170461649206, 0.6627450980392157, 0.16862745098039217, 0.01568627450980392, 2.8524877846230083, 0.5882352941176471, 0.11372549019607843, 0.023529411764705882, 2.9016585230838245, 0.49411764705882355, 0.054901960784313725, 0.03529411764705882, 2.9508292615419123, 0.396078431372549, 0.0392156862745098, 0.058823529411764705, 3.0, 0.301961, 0.047059, 0.090196]
    binLUT.ColorSpace = 'Lab'
    binLUT.NanColor = [0.25, 0.0, 0.0]
    binLUT.NanOpacity = 0.0
    binLUT.NumberOfTableValues = 10
    binLUT.ScalarRangeInitialized = 1.0

    # get color legend/bar for binLUT in view renderView1
    binLUTColorBar = GetScalarBar(binLUT, renderView1)
    binLUTColorBar.WindowLocation = 'Any Location'
    binLUTColorBar.Title = 'bin'
    binLUTColorBar.ComponentTitle = ''
    binLUTColorBar.HorizontalTitle = 1
    binLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    binLUTColorBar.TitleFontSize = 24
    binLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    binLUTColorBar.LabelFormat = '%.4f'
    binLUTColorBar.RangeLabelFormat = '%.3f'

    # set color bar visibility
    binLUTColorBar.Visibility = 0

    # get color legend/bar for basemapLUT in view renderView1
    basemapLUTColorBar = GetScalarBar(basemapLUT, renderView1)
    basemapLUTColorBar.WindowLocation = 'Any Location'
    basemapLUTColorBar.Title = 'basemap'
    basemapLUTColorBar.ComponentTitle = 'Magnitude'
    basemapLUTColorBar.HorizontalTitle = 1
    basemapLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    basemapLUTColorBar.TitleFontSize = 24
    basemapLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    basemapLUTColorBar.LabelFormat = '%.4f'
    basemapLUTColorBar.RangeLabelFormat = '%.3f'

    # set color bar visibility
    basemapLUTColorBar.Visibility = 0

    # get color legend/bar for rateLUT in view renderView1
    rateLUTColorBar = GetScalarBar(rateLUT, renderView1)
    rateLUTColorBar.WindowLocation = 'Any Location'
    rateLUTColorBar.Position = [0.7806810035842294, 0.06071246819338426]
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
    rateLUTColorBar.Visibility = 0

    # get color legend/bar for paramsLUT in view renderView1
    paramsLUTColorBar = GetScalarBar(paramsLUT, renderView1)
    paramsLUTColorBar.WindowLocation = 'Any Location'
    paramsLUTColorBar.Position = [0.7430465949820788, 0.058167938931297694]
    paramsLUTColorBar.Title = 'params'
    paramsLUTColorBar.ComponentTitle = 'X'
    paramsLUTColorBar.HorizontalTitle = 1
    paramsLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    paramsLUTColorBar.TitleFontSize = 24
    paramsLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    paramsLUTColorBar.LabelFormat = '%.4f'
    paramsLUTColorBar.RangeLabelFormat = '%.3f'

    # set color bar visibility
    paramsLUTColorBar.Visibility = 0

    # hide data in view
    Hide(fe_lowvti, renderView1)

    # hide data in view
    Hide(fevti, renderView1)

    # hide data in view
    Hide(hybridvti, renderView1)

    # ----------------------------------------------------------------
    # setup color maps and opacity mapes used in the visualization
    # note: the Get..() functions create a new object, if needed
    # ----------------------------------------------------------------

    # get opacity transfer function/opacity map for 'mask'
    maskPWF = GetOpacityTransferFunction('mask')
    maskPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    maskPWF.ScalarRangeInitialized = 1

    # get opacity transfer function/opacity map for 'bin'
    binPWF = GetOpacityTransferFunction('bin')
    binPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.045602607540786266, 0.5955882668495178, 0.5, 0.0, 0.2768729701638222, 0.625, 0.5, 0.0, 0.30293161422014236, 0.8897058963775635, 0.5, 0.0, 0.6775244772434235, 0.9705882668495178, 0.5, 0.0, 3.0, 1.0, 0.5, 0.0]
    binPWF.ScalarRangeInitialized = 1

    # get opacity transfer function/opacity map for 'vtkBlockColors'
    vtkBlockColorsPWF = GetOpacityTransferFunction('vtkBlockColors')
    vtkBlockColorsPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    coastlinevtmDisplay.LineWidth = size_factor
    regionvtmDisplay.LineWidth = size_factor
    # ----------------------------------------------------------------
    # restore active source

    # generate extracts

    SaveExtracts(ExtractsOutputDirectory='extracts')
    SaveScreenshot('figure_npfe_low.png',
                   renderView1,
                   ImageResolution=[size_factor * i for i in renderView1.ViewSize])


def make_joint_figure():

    binmaps = [plt.imread(f"figure_fe_low.png"),
               plt.imread(f"figure_npfe_low.png")]
    texts = ['a) Poisson\nFloor Ensemble',
             'b) Negbinom\nFloor Ensemble']
    fig, axs = plt.subplots(1, 2,
                            figsize=(6, 5),
                            gridspec_kw={'wspace': -0.001},
                            constrained_layout=False)

    cities = {'Wellington': [1781, 2035],
              'Auckland': [1813, 821],
              'Tauranga': [2125, 1066],
              'Gisborne': [2486, 1356],
              'Napier': [2259, 1603],
              'Christchurch': [1355, 2685],
              'Queenstown': [593, 3115],
              'Dunedin': [947, 3300],
              'Invercargill': [530, 3545]}

    # print(cities_2193)
    for i, j in enumerate(binmaps):
        if i == 0:
            for x, city in cities.items():
                axs[i].scatter(city[0], city[1], edgecolor='black',
                               color='white', s=20)
                axs[i].text(city[0] + 20, city[1] - 40, x, fontsize=8,
                            horizontalalignment='center',
                            verticalalignment='bottom',
                            **{'fontname': 'Ubuntu'})
            # axs[i].scatter(cities[:, 0], cities[:, 1], c='k', s=10)

        axs[i].imshow(j)
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)
        axs[i].text(100, 100, texts[i], fontsize=10, verticalalignment='top',
                    **{'fontname': 'Ubuntu'})

    plt.savefig("forecasts_fe.jpg", dpi=400, bbox_inches='tight')
    plt.savefig("fig3.jpg", dpi=1200, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    makefig_fe_low(5)
    makefig_npfe_low(5)
    make_joint_figure()

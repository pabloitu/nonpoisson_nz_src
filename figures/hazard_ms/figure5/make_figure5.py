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


def makefig_m(size_factor):
    
    # get the material library
    materialLibrary1 = GetMaterialLibrary()
    
    # Create a new 'Render View'
    renderView1 = CreateView('RenderView')
    renderView1.ViewSize = [534, 786]
    renderView1.InteractionMode = '2D'
    renderView1.AxesGrid = 'GridAxes3DActor'
    renderView1.OrientationAxesVisibility = 0
    renderView1.CenterOfRotation = [1575927.8735133181, 5400949.511356351, -10.0]
    renderView1.StereoType = 'Crystal Eyes'
    renderView1.CameraPosition = [1590343.8998113205, 5462560.284089721, 9266090.0]
    renderView1.CameraFocalPoint = [1590343.8998113205, 5462560.284089721, -10.0]
    renderView1.CameraFocalDisk = 1.0
    renderView1.CameraParallelScale = 744957.9411375008
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
    layout1.SetSize(534, 786)
    
    # ----------------------------------------------------------------
    # restore active view
    SetActiveView(renderView1)
    # ----------------------------------------------------------------
    
    # ----------------------------------------------------------------
    # setup the data processing pipelines
    # ----------------------------------------------------------------
    
    # create a new 'XML Image Data Reader'
    fe_statsvti = XMLImageDataReader(registrationName='fe_stats.vti', FileName=['paraview/fe_stats.vti'])
    fe_statsvti.CellArrayStatus = ['PGA_0.1', 'mask']
    fe_statsvti.TimeArray = 'None'
    
    # create a new 'XML Image Data Reader'
    pua_3_statsvti = XMLImageDataReader(registrationName='pua_3_stats.vti', FileName=['paraview/pua_3_stats.vti'])
    pua_3_statsvti.CellArrayStatus = ['PGA_0.1', 'mask']
    pua_3_statsvti.TimeArray = 'None'
    
    # create a new 'XML Image Data Reader'
    basemap_2193vti = XMLImageDataReader(registrationName='basemap_2193.vti', FileName=['paraview/basemap_2193.vti'])
    basemap_2193vti.CellArrayStatus = ['basemap', 'mask']
    basemap_2193vti.TimeArray = 'None'
    
    # create a new 'XML MultiBlock Data Reader'
    coastlinevtm = XMLMultiBlockDataReader(registrationName='coastline.vtm', FileName=['paraview/coastline.vtm'])
    coastlinevtm.CellArrayStatus = ['t50_fid', 'elevation']
    coastlinevtm.TimeArray = 'None'
    
    # create a new 'XML Image Data Reader'
    m_statsvti = XMLImageDataReader(registrationName='m_stats.vti', FileName=['paraview/m_stats.vti'])
    m_statsvti.CellArrayStatus = ['PGA_0.1', 'mask']
    m_statsvti.TimeArray = 'None'
    
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
    basemapLUT.AutomaticRescaleRangeMode = 'Clamp and update every timestep'
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
    
    # show data from m_statsvti
    m_statsvtiDisplay = Show(m_statsvti, renderView1, 'UniformGridRepresentation')
    
    # get 2D transfer function for 'PGA_01'
    pGA_01TF2D = GetTransferFunction2D('PGA_01')
    pGA_01TF2D.ScalarRangeInitialized = 1
    pGA_01TF2D.Range = [0.0, 0.6, 0.0, 1.0]
    
    # get color transfer function/color map for 'PGA_01'
    pGA_01LUT = GetColorTransferFunction('PGA_01')
    pGA_01LUT.AutomaticRescaleRangeMode = 'Never'
    pGA_01LUT.EnableOpacityMapping = 1
    pGA_01LUT.TransferFunction2D = pGA_01TF2D
    pGA_01LUT.RGBPoints = [0.0, 0.278431372549, 0.278431372549, 0.858823529412, 0.08579999999999997, 0.0, 0.0, 0.360784313725, 0.171, 0.0, 1.0, 1.0, 0.2573999999999999, 0.0, 0.501960784314, 0.0, 0.34259999999999996, 1.0, 1.0, 0.0, 0.42840000000000006, 1.0, 0.380392156863, 0.0, 0.5142, 0.419607843137, 0.0, 0.0, 0.6, 0.878431372549, 0.301960784314, 0.301960784314]
    pGA_01LUT.ColorSpace = 'RGB'
    pGA_01LUT.NanOpacity = 0.0
    pGA_01LUT.NumberOfTableValues = 526
    pGA_01LUT.ScalarRangeInitialized = 1.0
    pGA_01LUT.VectorMode = 'Component'
    
    # get opacity transfer function/opacity map for 'PGA_01'
    pGA_01PWF = GetOpacityTransferFunction('PGA_01')
    pGA_01PWF.Points = [0.0, 0.0, 0.5, 0.0, 0.027000000700354576, 0.2663043439388275, 0.5, 0.0, 0.055800002068281174, 0.45652174949645996, 0.5, 0.0, 0.09240000694990158, 0.635869562625885, 0.5, 0.0, 0.131400004029274, 0.7771739363670349, 0.5, 0.0, 0.19740000367164612, 0.8586956858634949, 0.5, 0.0, 0.6, 1.0, 0.5, 0.0]
    pGA_01PWF.ScalarRangeInitialized = 1
    
    # trace defaults for the display properties.
    m_statsvtiDisplay.Representation = 'Slice'
    m_statsvtiDisplay.ColorArrayName = ['CELLS', 'PGA_0.1']
    m_statsvtiDisplay.LookupTable = pGA_01LUT
    m_statsvtiDisplay.SelectTCoordArray = 'None'
    m_statsvtiDisplay.SelectNormalArray = 'None'
    m_statsvtiDisplay.SelectTangentArray = 'None'
    m_statsvtiDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    m_statsvtiDisplay.SelectOrientationVectors = 'None'
    m_statsvtiDisplay.ScaleFactor = 154400.0
    m_statsvtiDisplay.SelectScaleArray = 'None'
    m_statsvtiDisplay.GlyphType = 'Arrow'
    m_statsvtiDisplay.GlyphTableIndexArray = 'None'
    m_statsvtiDisplay.GaussianRadius = 7720.0
    m_statsvtiDisplay.SetScaleArray = [None, '']
    m_statsvtiDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    m_statsvtiDisplay.OpacityArray = [None, '']
    m_statsvtiDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    m_statsvtiDisplay.DataAxesGrid = 'GridAxesRepresentation'
    m_statsvtiDisplay.PolarAxes = 'PolarAxesRepresentation'
    m_statsvtiDisplay.ScalarOpacityUnitDistance = 25222.71540033692
    m_statsvtiDisplay.ScalarOpacityFunction = pGA_01PWF
    m_statsvtiDisplay.TransferFunction2D = pGA_01TF2D
    m_statsvtiDisplay.OpacityArrayName = ['CELLS', 'PGA_0.1']
    m_statsvtiDisplay.ColorArray2Name = ['CELLS', 'PGA_0.1']
    m_statsvtiDisplay.SliceFunction = 'Plane'
    m_statsvtiDisplay.SelectInputVectors = [None, '']
    m_statsvtiDisplay.WriteLog = ''
    
    # init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
    m_statsvtiDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    
    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    m_statsvtiDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    
    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    m_statsvtiDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    
    # init the 'Plane' selected for 'SliceFunction'
    m_statsvtiDisplay.SliceFunction.Origin = [1590196.015518637, 5469386.7749801995, 10.0]
    
    # show data from pua_3_statsvti
    pua_3_statsvtiDisplay = Show(pua_3_statsvti, renderView1, 'UniformGridRepresentation')
    
    # trace defaults for the display properties.
    pua_3_statsvtiDisplay.Representation = 'Slice'
    pua_3_statsvtiDisplay.ColorArrayName = ['CELLS', 'PGA_0.1']
    pua_3_statsvtiDisplay.LookupTable = pGA_01LUT
    pua_3_statsvtiDisplay.SelectTCoordArray = 'None'
    pua_3_statsvtiDisplay.SelectNormalArray = 'None'
    pua_3_statsvtiDisplay.SelectTangentArray = 'None'
    pua_3_statsvtiDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    pua_3_statsvtiDisplay.SelectOrientationVectors = 'None'
    pua_3_statsvtiDisplay.ScaleFactor = 154400.0
    pua_3_statsvtiDisplay.SelectScaleArray = 'None'
    pua_3_statsvtiDisplay.GlyphType = 'Arrow'
    pua_3_statsvtiDisplay.GlyphTableIndexArray = 'None'
    pua_3_statsvtiDisplay.GaussianRadius = 7720.0
    pua_3_statsvtiDisplay.SetScaleArray = [None, '']
    pua_3_statsvtiDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    pua_3_statsvtiDisplay.OpacityArray = [None, '']
    pua_3_statsvtiDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    pua_3_statsvtiDisplay.DataAxesGrid = 'GridAxesRepresentation'
    pua_3_statsvtiDisplay.PolarAxes = 'PolarAxesRepresentation'
    pua_3_statsvtiDisplay.ScalarOpacityUnitDistance = 25222.71540033692
    pua_3_statsvtiDisplay.ScalarOpacityFunction = pGA_01PWF
    pua_3_statsvtiDisplay.TransferFunction2D = pGA_01TF2D
    pua_3_statsvtiDisplay.OpacityArrayName = ['CELLS', 'PGA_0.1']
    pua_3_statsvtiDisplay.ColorArray2Name = ['CELLS', 'PGA_0.1']
    pua_3_statsvtiDisplay.SliceFunction = 'Plane'
    pua_3_statsvtiDisplay.SelectInputVectors = [None, '']
    pua_3_statsvtiDisplay.WriteLog = ''
    
    # init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
    pua_3_statsvtiDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    
    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    pua_3_statsvtiDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    
    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    pua_3_statsvtiDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    
    # init the 'Plane' selected for 'SliceFunction'
    pua_3_statsvtiDisplay.SliceFunction.Origin = [1590196.015518637, 5469386.7749801995, 10.0]
    
    # show data from fe_statsvti
    fe_statsvtiDisplay = Show(fe_statsvti, renderView1, 'UniformGridRepresentation')
    
    # trace defaults for the display properties.
    fe_statsvtiDisplay.Representation = 'Slice'
    fe_statsvtiDisplay.ColorArrayName = ['CELLS', 'PGA_0.1']
    fe_statsvtiDisplay.LookupTable = pGA_01LUT
    fe_statsvtiDisplay.SelectTCoordArray = 'None'
    fe_statsvtiDisplay.SelectNormalArray = 'None'
    fe_statsvtiDisplay.SelectTangentArray = 'None'
    fe_statsvtiDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    fe_statsvtiDisplay.SelectOrientationVectors = 'None'
    fe_statsvtiDisplay.ScaleFactor = 154400.0
    fe_statsvtiDisplay.SelectScaleArray = 'None'
    fe_statsvtiDisplay.GlyphType = 'Arrow'
    fe_statsvtiDisplay.GlyphTableIndexArray = 'None'
    fe_statsvtiDisplay.GaussianRadius = 7720.0
    fe_statsvtiDisplay.SetScaleArray = [None, '']
    fe_statsvtiDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    fe_statsvtiDisplay.OpacityArray = [None, '']
    fe_statsvtiDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    fe_statsvtiDisplay.DataAxesGrid = 'GridAxesRepresentation'
    fe_statsvtiDisplay.PolarAxes = 'PolarAxesRepresentation'
    fe_statsvtiDisplay.ScalarOpacityUnitDistance = 25222.71540033692
    fe_statsvtiDisplay.ScalarOpacityFunction = pGA_01PWF
    fe_statsvtiDisplay.TransferFunction2D = pGA_01TF2D
    fe_statsvtiDisplay.OpacityArrayName = ['CELLS', 'PGA_0.1']
    fe_statsvtiDisplay.ColorArray2Name = ['CELLS', 'PGA_0.1']
    fe_statsvtiDisplay.SliceFunction = 'Plane'
    fe_statsvtiDisplay.SelectInputVectors = [None, '']
    fe_statsvtiDisplay.WriteLog = ''
    
    # init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
    fe_statsvtiDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    
    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    fe_statsvtiDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    
    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    fe_statsvtiDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    
    # init the 'Plane' selected for 'SliceFunction'
    fe_statsvtiDisplay.SliceFunction.Origin = [1590196.015518637, 5469386.7749801995, 10.0]
    
    # setup the color legend parameters for each legend in this view
    
    # get 2D transfer function for 'bin'
    binTF2D = GetTransferFunction2D('bin')
    binTF2D.ScalarRangeInitialized = 1
    binTF2D.Range = [0.0, 3.0, 0.0, 1.0]
    
    # get color transfer function/color map for 'bin'
    binLUT = GetColorTransferFunction('bin')
    binLUT.AutomaticRescaleRangeMode = 'Clamp and update every timestep'
    binLUT.EnableOpacityMapping = 1
    binLUT.TransferFunction2D = binTF2D
    binLUT.RGBPoints = [0.0, 0.8901960784313725, 0.9568627450980393, 0.984313725490196, 0.049763931135892725, 0.807843137254902, 0.9019607843137255, 0.9607843137254902, 0.09952786227178578, 0.7607843137254902, 0.8588235294117647, 0.9294117647058824, 0.19905572454357157, 0.6862745098039216, 0.7843137254901961, 0.8705882352941177, 0.298583586815357, 0.6, 0.6980392156862745, 0.8, 0.39811144908714313, 0.5176470588235295, 0.615686274509804, 0.7254901960784313, 0.4976393113589286, 0.44313725490196076, 0.5333333333333333, 0.6431372549019608, 0.5971671736307144, 0.37254901960784315, 0.4588235294117647, 0.5686274509803921, 0.6966950359025001, 0.3058823529411765, 0.38823529411764707, 0.49411764705882355, 0.7962228981742859, 0.24705882352941178, 0.3176470588235294, 0.41568627450980394, 0.8957507604460717, 0.2, 0.2549019607843137, 0.34509803921568627, 0.9952786227178572, 0.14902, 0.196078, 0.278431, 1.046370953123331, 0.2, 0.1450980392156863, 0.13725490196078433, 1.0974332835288048, 0.23137254901960785, 0.17254901960784313, 0.16470588235294117, 1.1484956139342786, 0.2549019607843137, 0.2, 0.1843137254901961, 1.1995579443397522, 0.28627450980392155, 0.23137254901960785, 0.21176470588235294, 1.2506202747452257, 0.3176470588235294, 0.2627450980392157, 0.23921568627450981, 1.3016826051506996, 0.34509803921568627, 0.29411764705882354, 0.26666666666666666, 1.3527449355561734, 0.37254901960784315, 0.3215686274509804, 0.29411764705882354, 1.403807265961647, 0.403921568627451, 0.3568627450980392, 0.3176470588235294, 1.4548695963671205, 0.43137254901960786, 0.38823529411764707, 0.34509803921568627, 1.5059319267725944, 0.4627450980392157, 0.4196078431372549, 0.37254901960784315, 1.5569942571780682, 0.4980392156862745, 0.4588235294117647, 0.40784313725490196, 1.6080565875835418, 0.5372549019607843, 0.5058823529411764, 0.44313725490196076, 1.6591189179890158, 0.5803921568627451, 0.5568627450980392, 0.48627450980392156, 1.7101812483944892, 0.6313725490196078, 0.6078431372549019, 0.5333333333333333, 1.7612435787999632, 0.6784313725490196, 0.6627450980392157, 0.5803921568627451, 1.8123059092054365, 0.7254901960784313, 0.7137254901960784, 0.6274509803921569, 1.8633682396109106, 0.7764705882352941, 0.7647058823529411, 0.6784313725490196, 1.914430570016384, 0.8274509803921568, 0.8235294117647058, 0.7372549019607844, 1.965492900421858, 0.8901960784313725, 0.8901960784313725, 0.807843137254902, 2.0165552308273313, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 2.0165552308273313, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 2.0165852308273315, 0.9882352941176471, 0.9882352941176471, 0.8588235294117647, 2.018552060365677, 0.984313725490196, 0.9882352941176471, 0.8549019607843137, 2.0657559692859646, 0.9882352941176471, 0.9882352941176471, 0.7568627450980392, 2.1149267077445986, 0.9803921568627451, 0.9725490196078431, 0.6705882352941176, 2.1640974462032316, 0.9803921568627451, 0.9490196078431372, 0.596078431372549, 2.2132681846618656, 0.9725490196078431, 0.9176470588235294, 0.5137254901960784, 2.2624389231204987, 0.9686274509803922, 0.8784313725490196, 0.4235294117647059, 2.3116096615791317, 0.9647058823529412, 0.8274509803921568, 0.3254901960784314, 2.3607804000377657, 0.9529411764705882, 0.7686274509803922, 0.24705882352941178, 2.4099511384963987, 0.9450980392156862, 0.7098039215686275, 0.1843137254901961, 2.4591218769550327, 0.9372549019607843, 0.6431372549019608, 0.12156862745098039, 2.5082926154136658, 0.9254901960784314, 0.5803921568627451, 0.0784313725490196, 2.557463353872299, 0.9176470588235294, 0.5215686274509804, 0.047058823529411764, 2.606634092330933, 0.9019607843137255, 0.4588235294117647, 0.027450980392156862, 2.655804830789566, 0.8627450980392157, 0.3803921568627451, 0.011764705882352941, 2.7049755692482, 0.788235294117647, 0.2901960784313726, 0.0, 2.754146307706833, 0.7411764705882353, 0.22745098039215686, 0.00392156862745098, 2.803317046165466, 0.6627450980392157, 0.16862745098039217, 0.01568627450980392, 2.8524877846240995, 0.5882352941176471, 0.11372549019607843, 0.023529411764705882, 2.9016585230827334, 0.49411764705882355, 0.054901960784313725, 0.03529411764705882, 2.950829261541367, 0.396078431372549, 0.0392156862745098, 0.058823529411764705, 3.0, 0.301961, 0.047059, 0.090196]
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
    
    # get 2D transfer function for 'rate'
    rateTF2D = GetTransferFunction2D('rate')
    rateTF2D.ScalarRangeInitialized = 1
    rateTF2D.Range = [2.3055461497278884e-05, 0.003394088940694928, 0.0, 1.0]
    
    # get color transfer function/color map for 'rate'
    rateLUT = GetColorTransferFunction('rate')
    rateLUT.AutomaticRescaleRangeMode = 'Clamp and update every timestep'
    rateLUT.EnableOpacityMapping = 1
    rateLUT.TransferFunction2D = rateTF2D
    rateLUT.RGBPoints = [2.3055461497278884e-05, 0.8901960784313725, 0.9568627450980393, 0.984313725490196, 7.901803532200734e-05, 0.807843137254902, 0.9019607843137255, 0.9607843137254902, 0.00013498060914673616, 0.7607843137254902, 0.8588235294117647, 0.9294117647058824, 0.00024690575679619343, 0.6862745098039216, 0.7843137254901961, 0.8705882352941177, 0.00035883090444565035, 0.6, 0.6980392156862745, 0.8, 0.000470756052095108, 0.5176470588235295, 0.615686274509804, 0.7254901960784313, 0.0005826811997445649, 0.44313725490196076, 0.5333333333333333, 0.6431372549019608, 0.0006946063473940221, 0.37254901960784315, 0.4588235294117647, 0.5686274509803921, 0.0008065314950434794, 0.3058823529411765, 0.38823529411764707, 0.49411764705882355, 0.0009184566426929367, 0.24705882352941178, 0.3176470588235294, 0.41568627450980394, 0.001030381790342394, 0.2, 0.2549019607843137, 0.34509803921568627, 0.001142306937991851, 0.14902, 0.196078, 0.278431, 0.0011997633777651848, 0.2, 0.1450980392156863, 0.13725490196078433, 0.0012571860807099274, 0.23137254901960785, 0.17254901960784313, 0.16470588235294117, 0.00131460878365467, 0.2549019607843137, 0.2, 0.1843137254901961, 0.0013720314865994126, 0.28627450980392155, 0.23137254901960785, 0.21176470588235294, 0.0014294541895441555, 0.3176470588235294, 0.2627450980392157, 0.23921568627450981, 0.0014868768924888981, 0.34509803921568627, 0.29411764705882354, 0.26666666666666666, 0.0015442995954336408, 0.37254901960784315, 0.3215686274509804, 0.29411764705882354, 0.0016017222983783833, 0.403921568627451, 0.3568627450980392, 0.3176470588235294, 0.001659145001323126, 0.43137254901960786, 0.38823529411764707, 0.34509803921568627, 0.0017165677042678686, 0.4627450980392157, 0.4196078431372549, 0.37254901960784315, 0.0017739904072126113, 0.4980392156862745, 0.4588235294117647, 0.40784313725490196, 0.001831413110157354, 0.5372549019607843, 0.5058823529411764, 0.44313725490196076, 0.0018888358131020969, 0.5803921568627451, 0.5568627450980392, 0.48627450980392156, 0.0019462585160468393, 0.6313725490196078, 0.6078431372549019, 0.5333333333333333, 0.0020036812189915825, 0.6784313725490196, 0.6627450980392157, 0.5803921568627451, 0.0020611039219363245, 0.7254901960784313, 0.7137254901960784, 0.6274509803921569, 0.002118526624881068, 0.7764705882352941, 0.7647058823529411, 0.6784313725490196, 0.00217594932782581, 0.8274509803921568, 0.8235294117647058, 0.7372549019607844, 0.0022333720307705527, 0.8901960784313725, 0.8901960784313725, 0.807843137254902, 0.002290794733715295, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 0.002290794733715295, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 0.002290828470543887, 0.9882352941176471, 0.9882352941176471, 0.8588235294117647, 0.002293040290243977, 0.984313725490196, 0.9882352941176471, 0.8549019607843137, 0.0023461239630461453, 0.9882352941176471, 0.9882352941176471, 0.7568627450980392, 0.0024014194555484047, 0.9803921568627451, 0.9725490196078431, 0.6705882352941176, 0.002456714948050663, 0.9803921568627451, 0.9490196078431372, 0.596078431372549, 0.0025120104405529226, 0.9725490196078431, 0.9176470588235294, 0.5137254901960784, 0.002567305933055181, 0.9686274509803922, 0.8784313725490196, 0.4235294117647059, 0.00262260142555744, 0.9647058823529412, 0.8274509803921568, 0.3254901960784314, 0.0026778969180596994, 0.9529411764705882, 0.7686274509803922, 0.24705882352941178, 0.002733192410561958, 0.9450980392156862, 0.7098039215686275, 0.1843137254901961, 0.0027884879030642172, 0.9372549019607843, 0.6431372549019608, 0.12156862745098039, 0.002843783395566476, 0.9254901960784314, 0.5803921568627451, 0.0784313725490196, 0.0028990788880687347, 0.9176470588235294, 0.5215686274509804, 0.047058823529411764, 0.002954374380570994, 0.9019607843137255, 0.4588235294117647, 0.027450980392156862, 0.0030096698730732526, 0.8627450980392157, 0.3803921568627451, 0.011764705882352941, 0.003064965365575512, 0.788235294117647, 0.2901960784313726, 0.0, 0.003120260858077771, 0.7411764705882353, 0.22745098039215686, 0.00392156862745098, 0.0031755563505800294, 0.6627450980392157, 0.16862745098039217, 0.01568627450980392, 0.0032308518430822887, 0.5882352941176471, 0.11372549019607843, 0.023529411764705882, 0.003286147335584548, 0.49411764705882355, 0.054901960784313725, 0.03529411764705882, 0.0033414428280868066, 0.396078431372549, 0.0392156862745098, 0.058823529411764705, 0.0033967383205890656, 0.301961, 0.047059, 0.090196]
    rateLUT.ColorSpace = 'Lab'
    rateLUT.NanColor = [0.25, 0.0, 0.0]
    rateLUT.NanOpacity = 0.0
    rateLUT.NumberOfTableValues = 10
    rateLUT.ScalarRangeInitialized = 1.0
    
    # get color legend/bar for rateLUT in view renderView1
    rateLUTColorBar = GetScalarBar(rateLUT, renderView1)
    rateLUTColorBar.WindowLocation = 'Any Location'
    rateLUTColorBar.Title = 'rate'
    rateLUTColorBar.ComponentTitle = ''
    rateLUTColorBar.HorizontalTitle = 1
    rateLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    rateLUTColorBar.TitleFontSize = 24
    rateLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    rateLUTColorBar.LabelFormat = '%.4f'
    rateLUTColorBar.RangeLabelFormat = '%.3f'
    
    # set color bar visibility
    rateLUTColorBar.Visibility = 0
    
    # get 2D transfer function for 'rate_learning'
    rate_learningTF2D = GetTransferFunction2D('rate_learning')
    rate_learningTF2D.ScalarRangeInitialized = 1
    rate_learningTF2D.Range = [0.006257875356823206, 0.9212474822998047, 0.0, 1.0]
    
    # get color transfer function/color map for 'rate_learning'
    rate_learningLUT = GetColorTransferFunction('rate_learning')
    rate_learningLUT.AutomaticRescaleRangeMode = 'Clamp and update every timestep'
    rate_learningLUT.EnableOpacityMapping = 1
    rate_learningLUT.TransferFunction2D = rate_learningTF2D
    rate_learningLUT.RGBPoints = [0.006257875356823206, 0.8901960784313725, 0.9568627450980393, 0.984313725490196, 0.021435701953479235, 0.807843137254902, 0.9019607843137255, 0.9607843137254902, 0.03661352855013536, 0.7607843137254902, 0.8588235294117647, 0.9294117647058824, 0.06696918174344751, 0.6862745098039216, 0.7843137254901961, 0.8705882352941177, 0.09732483493675957, 0.6, 0.6980392156862745, 0.8, 0.12768048813007182, 0.5176470588235295, 0.615686274509804, 0.7254901960784313, 0.1580361413233839, 0.44313725490196076, 0.5333333333333333, 0.6431372549019608, 0.18839179451669605, 0.37254901960784315, 0.4588235294117647, 0.5686274509803921, 0.2187474477100082, 0.3058823529411765, 0.38823529411764707, 0.49411764705882355, 0.24910310090332036, 0.24705882352941178, 0.3176470588235294, 0.41568627450980394, 0.2794587540966325, 0.2, 0.2549019607843137, 0.34509803921568627, 0.3098144072899446, 0.14902, 0.196078, 0.278431, 0.32539739106177973, 0.2, 0.1450980392156863, 0.13725490196078433, 0.3409712249375454, 0.23137254901960785, 0.17254901960784313, 0.16470588235294117, 0.3565450588133111, 0.2549019607843137, 0.2, 0.1843137254901961, 0.3721188926890768, 0.28627450980392155, 0.23137254901960785, 0.21176470588235294, 0.38769272656484244, 0.3176470588235294, 0.2627450980392157, 0.23921568627450981, 0.4032665604406081, 0.34509803921568627, 0.29411764705882354, 0.26666666666666666, 0.4188403943163738, 0.37254901960784315, 0.3215686274509804, 0.29411764705882354, 0.4344142281921395, 0.403921568627451, 0.3568627450980392, 0.3176470588235294, 0.44998806206790515, 0.43137254901960786, 0.38823529411764707, 0.34509803921568627, 0.4655618959436708, 0.4627450980392157, 0.4196078431372549, 0.37254901960784315, 0.48113572981943653, 0.4980392156862745, 0.4588235294117647, 0.40784313725490196, 0.4967095636952022, 0.5372549019607843, 0.5058823529411764, 0.44313725490196076, 0.512283397570968, 0.5803921568627451, 0.5568627450980392, 0.48627450980392156, 0.5278572314467336, 0.6313725490196078, 0.6078431372549019, 0.5333333333333333, 0.5434310653224993, 0.6784313725490196, 0.6627450980392157, 0.5803921568627451, 0.5590048991982649, 0.7254901960784313, 0.7137254901960784, 0.6274509803921569, 0.5745787330740306, 0.7764705882352941, 0.7647058823529411, 0.6784313725490196, 0.5901525669497962, 0.8274509803921568, 0.8235294117647058, 0.7372549019607844, 0.6057264008255621, 0.8901960784313725, 0.8901960784313725, 0.807843137254902, 0.6213002347013276, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 0.6213002347013276, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 0.6213093845973972, 0.9882352941176471, 0.9882352941176471, 0.8588235294117647, 0.621909260792802, 0.984313725490196, 0.9882352941176471, 0.8549019607843137, 0.6363062894825174, 0.9882352941176471, 0.9882352941176471, 0.7568627450980392, 0.6513031943676378, 0.9803921568627451, 0.9725490196078431, 0.6705882352941176, 0.6663000992527582, 0.9803921568627451, 0.9490196078431372, 0.596078431372549, 0.6812970041378786, 0.9725490196078431, 0.9176470588235294, 0.5137254901960784, 0.696293909022999, 0.9686274509803922, 0.8784313725490196, 0.4235294117647059, 0.7112908139081193, 0.9647058823529412, 0.8274509803921568, 0.3254901960784314, 0.7262877187932397, 0.9529411764705882, 0.7686274509803922, 0.24705882352941178, 0.7412846236783601, 0.9450980392156862, 0.7098039215686275, 0.1843137254901961, 0.7562815285634805, 0.9372549019607843, 0.6431372549019608, 0.12156862745098039, 0.7712784334486009, 0.9254901960784314, 0.5803921568627451, 0.0784313725490196, 0.7862753383337212, 0.9176470588235294, 0.5215686274509804, 0.047058823529411764, 0.8012722432188417, 0.9019607843137255, 0.4588235294117647, 0.027450980392156862, 0.816269148103962, 0.8627450980392157, 0.3803921568627451, 0.011764705882352941, 0.8312660529890824, 0.788235294117647, 0.2901960784313726, 0.0, 0.8462629578742028, 0.7411764705882353, 0.22745098039215686, 0.00392156862745098, 0.8612598627593231, 0.6627450980392157, 0.16862745098039217, 0.01568627450980392, 0.8762567676444435, 0.5882352941176471, 0.11372549019607843, 0.023529411764705882, 0.891253672529564, 0.49411764705882355, 0.054901960784313725, 0.03529411764705882, 0.9062505774146844, 0.396078431372549, 0.0392156862745098, 0.058823529411764705, 0.9212474822998047, 0.301961, 0.047059, 0.090196]
    rate_learningLUT.ColorSpace = 'Lab'
    rate_learningLUT.NanColor = [0.25, 0.0, 0.0]
    rate_learningLUT.NanOpacity = 0.0
    rate_learningLUT.NumberOfTableValues = 10
    rate_learningLUT.ScalarRangeInitialized = 1.0
    
    # get color legend/bar for rate_learningLUT in view renderView1
    rate_learningLUTColorBar = GetScalarBar(rate_learningLUT, renderView1)
    rate_learningLUTColorBar.WindowLocation = 'Any Location'
    rate_learningLUTColorBar.Title = 'rate_learning'
    rate_learningLUTColorBar.ComponentTitle = ''
    rate_learningLUTColorBar.HorizontalTitle = 1
    rate_learningLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    rate_learningLUTColorBar.TitleFontSize = 24
    rate_learningLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    rate_learningLUTColorBar.LabelFormat = '%.4f'
    rate_learningLUTColorBar.RangeLabelFormat = '%.3f'
    
    # set color bar visibility
    rate_learningLUTColorBar.Visibility = 0
    
    # get 2D transfer function for 'params'
    paramsTF2D = GetTransferFunction2D('params')
    paramsTF2D.ScalarRangeInitialized = 1
    paramsTF2D.Range = [2.3055461497278884e-05, 0.003394088940694928, 0.0, 1.0]
    
    # get color transfer function/color map for 'params'
    paramsLUT = GetColorTransferFunction('params')
    paramsLUT.AutomaticRescaleRangeMode = 'Clamp and update every timestep'
    paramsLUT.TransferFunction2D = paramsTF2D
    paramsLUT.RGBPoints = [4.9999999999999996e-06, 0.02, 0.3813, 0.9981, 5.839729084962496e-06, 0.02000006, 0.424267768, 0.96906969, 6.820487157151409e-06, 0.02, 0.467233763, 0.940033043, 7.965959445046067e-06, 0.02, 0.5102, 0.911, 9.303809012173444e-06, 0.02000006, 0.546401494, 0.872669438, 1.086634481786509e-05, 0.02, 0.582600362, 0.83433295, 1.2691301976023705e-05, 0.02, 0.6188, 0.796, 1.4822753055085525e-05, 0.02000006, 0.652535156, 0.749802434, 1.7312172426999927e-05, 0.02, 0.686267004, 0.703599538, 2.021967936916749e-05, 0.02, 0.72, 0.6574, 2.361548994014875e-05, 0.02000006, 0.757035456, 0.603735359, 2.758161269182518e-05, 0.02, 0.794067037, 0.55006613, 3.221382916932451e-05, 0.02, 0.8311, 0.4964, 3.7624007027623516e-05, 0.021354336738172372, 0.8645368555261631, 0.4285579460761159, 4.394280162640946e-05, 0.023312914349117714, 0.897999359924484, 0.36073871343115577, 5.132281134649612e-05, 0.015976108242848862, 0.9310479513349017, 0.2925631815088092, 5.994226282843569e-05, 0.27421074700988196, 0.952562960995083, 0.15356836602739213, 7.000931513153629e-05, 0.4933546281681699, 0.9619038625309482, 0.11119493614749336, 8.176708675838766e-05, 0.6439, 0.9773, 0.0469, 9.549952694712183e-05, 0.762401813, 0.984669591, 0.034600153, 0.00011153827302265329, 0.880901185, 0.992033407, 0.022299877, 0.0001302706594113756, 0.9995285432627147, 0.9995193706781492, 0.0134884641450013, 0.00015214907173637132, 0.999402998, 0.955036376, 0.079066628, 0.00017770187189378658, 0.9994, 0.910666223, 0.148134024, 0.0002075461579500854, 0.9994, 0.8663, 0.2172, 0.0002424026670106663, 0.999269665, 0.818035981, 0.217200652, 0.00028311318096293424, 0.999133332, 0.769766184, 0.2172, 0.0003306608554410999, 0.999, 0.7215, 0.2172, 0.00038619396295559445, 0.99913633, 0.673435546, 0.217200652, 0.0004510536235817441, 0.999266668, 0.625366186, 0.2172, 0.0005268061929016095, 0.9994, 0.5773, 0.2172, 0.0006152810893651771, 0.999402998, 0.521068455, 0.217200652, 0.0007186149745986482, 0.9994, 0.464832771, 0.2172, 0.0008393033536106648, 0.9994, 0.4086, 0.2172, 0.0009802608410373523, 0.9947599917687346, 0.33177297300202935, 0.2112309638520206, 0.001144891548851126, 0.9867129505479589, 0.2595183410914934, 0.19012239549291934, 0.0013371712953907375, 0.9912458875646419, 0.14799417507952672, 0.21078892136920357, 0.0015617436210540582, 0.949903037, 0.116867171, 0.252900603, 0.0018240319294248117, 0.903199533, 0.078432949, 0.291800389, 0.0021303704620324665, 0.8565, 0.04, 0.3307, 0.002488157269775202, 0.798902627, 0.04333345, 0.358434298, 0.0029060328752534247, 0.741299424, 0.0466667, 0.386166944, 0.0033940889406949295, 0.6837, 0.05, 0.4139]
    paramsLUT.UseLogScale = 1
    paramsLUT.ColorSpace = 'RGB'
    paramsLUT.NanColor = [1.0, 0.0, 0.0]
    paramsLUT.NanOpacity = 0.0
    paramsLUT.NumberOfTableValues = 668
    paramsLUT.ScalarRangeInitialized = 1.0
    paramsLUT.VectorMode = 'Component'
    
    # get color legend/bar for paramsLUT in view renderView1
    paramsLUTColorBar = GetScalarBar(paramsLUT, renderView1)
    paramsLUTColorBar.WindowLocation = 'Any Location'
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
    
    # get 2D transfer function for 'rates_bin'
    rates_binTF2D = GetTransferFunction2D('rates_bin')
    rates_binTF2D.ScalarRangeInitialized = 1
    rates_binTF2D.Range = [1.5241377582242421e-07, 2.1670205114787677e-06, 0.0, 1.0]
    
    # get color transfer function/color map for 'rates_bin'
    rates_binLUT = GetColorTransferFunction('rates_bin')
    rates_binLUT.AutomaticRescaleRangeMode = 'Clamp and update every timestep'
    rates_binLUT.TransferFunction2D = rates_binTF2D
    rates_binLUT.RGBPoints = [1.5241377582242416e-07, 0.02, 0.3813, 0.9981, 1.6235765315698968e-07, 0.02000006, 0.424267768, 0.96906969, 1.7295029531554424e-07, 0.02, 0.467233763, 0.940033043, 1.842340294289119e-07, 0.02, 0.5102, 0.911, 1.9625394416173947e-07, 0.02000006, 0.546401494, 0.872669438, 2.0905806988225645e-07, 0.02, 0.582600362, 0.83433295, 2.226975705867879e-07, 0.02, 0.6188, 0.796, 2.3722694834592718e-07, 0.02000006, 0.652535156, 0.749802434, 2.5270426108932124e-07, 0.02, 0.686267004, 0.703599538, 2.69191354599307e-07, 0.02, 0.72, 0.6574, 2.867541096404246e-07, 0.02000006, 0.757035456, 0.603735359, 3.0546270521231786e-07, 0.02, 0.794067037, 0.55006613, 3.2539189897794425e-07, 0.02, 0.8311, 0.4964, 3.466213259876652e-07, 0.021354336738172372, 0.8645368555261631, 0.4285579460761159, 3.692358168928809e-07, 0.023312914349117714, 0.897999359924484, 0.36073871343115577, 3.933257369207706e-07, 0.015976108242848862, 0.9310479513349017, 0.2925631815088092, 4.1898734696463397e-07, 0.27421074700988196, 0.952562960995083, 0.15356836602739213, 4.4632318823271956e-07, 0.4933546281681699, 0.9619038625309482, 0.11119493614749336, 4.754424919925664e-07, 0.6439, 0.9773, 0.0469, 5.064616160481415e-07, 0.762401813, 0.984669591, 0.034600153, 5.395045096938981e-07, 0.880901185, 0.992033407, 0.022299877, 5.747032090036736e-07, 0.9995285432627147, 0.9995193706781492, 0.0134884641450013, 6.12198364433534e-07, 0.999402998, 0.955036376, 0.079066628, 6.521398028468278e-07, 0.9994, 0.910666223, 0.148134024, 6.946871262072305e-07, 0.9994, 0.8663, 0.2172, 7.400103493321195e-07, 0.999269665, 0.818035981, 0.217200652, 7.882905792546511e-07, 0.999133332, 0.769766184, 0.2172, 8.397207389092128e-07, 0.999, 0.7215, 0.2172, 8.945063380320411e-07, 0.99913633, 0.673435546, 0.217200652, 9.528662943574151e-07, 0.999266668, 0.625366186, 0.2172, 1.0150338083908704e-06, 0.9994, 0.5773, 0.2172, 1.0812572952549147e-06, 0.999402998, 0.521068455, 0.217200652, 1.1518013773308415e-06, 0.9994, 0.464832771, 0.2172, 1.2269479416630953e-06, 0.9994, 0.4086, 0.2172, 1.3069972663515062e-06, 0.9947599917687346, 0.33177297300202935, 0.2112309638520206, 1.392269220432311e-06, 0.9867129505479589, 0.2595183410914934, 0.19012239549291934, 1.483104542042611e-06, 0.9912458875646419, 0.14799417507952672, 0.21078892136920357, 1.5798661999756265e-06, 0.949903037, 0.116867171, 0.252900603, 1.6829408440674303e-06, 0.903199533, 0.078432949, 0.291800389, 1.792740350210723e-06, 0.8565, 0.04, 0.3307, 1.909703466169423e-06, 0.798902627, 0.04333345, 0.358434298, 2.0342975647705156e-06, 0.741299424, 0.0466667, 0.386166944, 2.16702051147877e-06, 0.6837, 0.05, 0.4139]
    rates_binLUT.UseLogScale = 1
    rates_binLUT.ColorSpace = 'RGB'
    rates_binLUT.NanColor = [1.0, 0.0, 0.0]
    rates_binLUT.NanOpacity = 0.0
    rates_binLUT.NumberOfTableValues = 521
    rates_binLUT.ScalarRangeInitialized = 1.0
    rates_binLUT.VectorComponent = 18
    rates_binLUT.VectorMode = 'Component'
    
    # get color legend/bar for rates_binLUT in view renderView1
    rates_binLUTColorBar = GetScalarBar(rates_binLUT, renderView1)
    rates_binLUTColorBar.WindowLocation = 'Any Location'
    rates_binLUTColorBar.Title = 'rates_bin'
    rates_binLUTColorBar.ComponentTitle = '18'
    rates_binLUTColorBar.HorizontalTitle = 1
    rates_binLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    rates_binLUTColorBar.TitleFontSize = 24
    rates_binLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    rates_binLUTColorBar.LabelFormat = '%.4f'
    rates_binLUTColorBar.RangeLabelFormat = '%.3f'
    
    # set color bar visibility
    rates_binLUTColorBar.Visibility = 0
    
    # get 2D transfer function for 'poly'
    polyTF2D = GetTransferFunction2D('poly')
    polyTF2D.ScalarRangeInitialized = 1
    polyTF2D.Range = [0.0, 12.0, 0.0, 1.0]
    
    # get color transfer function/color map for 'poly'
    polyLUT = GetColorTransferFunction('poly')
    polyLUT.AutomaticRescaleRangeMode = 'Clamp and update every timestep'
    polyLUT.EnableOpacityMapping = 1
    polyLUT.TransferFunction2D = polyTF2D
    polyLUT.RGBPoints = [0.0, 0.8901960784313725, 0.9568627450980393, 0.984313725490196, 0.1990557245435709, 0.807843137254902, 0.9019607843137255, 0.9607843137254902, 0.39811144908714313, 0.7607843137254902, 0.8588235294117647, 0.9294117647058824, 0.7962228981742863, 0.6862745098039216, 0.7843137254901961, 0.8705882352941177, 1.194334347261428, 0.6, 0.6980392156862745, 0.8, 1.5924457963485725, 0.5176470588235295, 0.615686274509804, 0.7254901960784313, 1.9905572454357143, 0.44313725490196076, 0.5333333333333333, 0.6431372549019608, 2.3886686945228575, 0.37254901960784315, 0.4588235294117647, 0.5686274509803921, 2.7867801436100006, 0.3058823529411765, 0.38823529411764707, 0.49411764705882355, 3.1848915926971437, 0.24705882352941178, 0.3176470588235294, 0.41568627450980394, 3.583003041784287, 0.2, 0.2549019607843137, 0.34509803921568627, 3.9811144908714287, 0.14902, 0.196078, 0.278431, 4.185483812493324, 0.2, 0.1450980392156863, 0.13725490196078433, 4.389733134115219, 0.23137254901960785, 0.17254901960784313, 0.16470588235294117, 4.593982455737114, 0.2549019607843137, 0.2, 0.1843137254901961, 4.798231777359009, 0.28627450980392155, 0.23137254901960785, 0.21176470588235294, 5.002481098980903, 0.3176470588235294, 0.2627450980392157, 0.23921568627450981, 5.206730420602798, 0.34509803921568627, 0.29411764705882354, 0.26666666666666666, 5.4109797422246935, 0.37254901960784315, 0.3215686274509804, 0.29411764705882354, 5.615229063846588, 0.403921568627451, 0.3568627450980392, 0.3176470588235294, 5.819478385468482, 0.43137254901960786, 0.38823529411764707, 0.34509803921568627, 6.023727707090377, 0.4627450980392157, 0.4196078431372549, 0.37254901960784315, 6.227977028712273, 0.4980392156862745, 0.4588235294117647, 0.40784313725490196, 6.432226350334167, 0.5372549019607843, 0.5058823529411764, 0.44313725490196076, 6.636475671956063, 0.5803921568627451, 0.5568627450980392, 0.48627450980392156, 6.840724993577957, 0.6313725490196078, 0.6078431372549019, 0.5333333333333333, 7.044974315199853, 0.6784313725490196, 0.6627450980392157, 0.5803921568627451, 7.249223636821746, 0.7254901960784313, 0.7137254901960784, 0.6274509803921569, 7.453472958443642, 0.7764705882352941, 0.7647058823529411, 0.6784313725490196, 7.657722280065536, 0.8274509803921568, 0.8235294117647058, 0.7372549019607844, 7.861971601687432, 0.8901960784313725, 0.8901960784313725, 0.807843137254902, 8.066220923309325, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 8.066220923309325, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 8.066340923309326, 0.9882352941176471, 0.9882352941176471, 0.8588235294117647, 8.074208241462708, 0.984313725490196, 0.9882352941176471, 0.8549019607843137, 8.263023877143858, 0.9882352941176471, 0.9882352941176471, 0.7568627450980392, 8.459706830978394, 0.9803921568627451, 0.9725490196078431, 0.6705882352941176, 8.656389784812927, 0.9803921568627451, 0.9490196078431372, 0.596078431372549, 8.853072738647462, 0.9725490196078431, 0.9176470588235294, 0.5137254901960784, 9.049755692481995, 0.9686274509803922, 0.8784313725490196, 0.4235294117647059, 9.246438646316527, 0.9647058823529412, 0.8274509803921568, 0.3254901960784314, 9.443121600151063, 0.9529411764705882, 0.7686274509803922, 0.24705882352941178, 9.639804553985595, 0.9450980392156862, 0.7098039215686275, 0.1843137254901961, 9.83648750782013, 0.9372549019607843, 0.6431372549019608, 0.12156862745098039, 10.033170461654663, 0.9254901960784314, 0.5803921568627451, 0.0784313725490196, 10.229853415489195, 0.9176470588235294, 0.5215686274509804, 0.047058823529411764, 10.426536369323731, 0.9019607843137255, 0.4588235294117647, 0.027450980392156862, 10.623219323158263, 0.8627450980392157, 0.3803921568627451, 0.011764705882352941, 10.8199022769928, 0.788235294117647, 0.2901960784313726, 0.0, 11.016585230827332, 0.7411764705882353, 0.22745098039215686, 0.00392156862745098, 11.213268184661864, 0.6627450980392157, 0.16862745098039217, 0.01568627450980392, 11.409951138496398, 0.5882352941176471, 0.11372549019607843, 0.023529411764705882, 11.606634092330934, 0.49411764705882355, 0.054901960784313725, 0.03529411764705882, 11.803317046165468, 0.396078431372549, 0.0392156862745098, 0.058823529411764705, 12.0, 0.301961, 0.047059, 0.090196]
    polyLUT.ColorSpace = 'Lab'
    polyLUT.NanColor = [0.25, 0.0, 0.0]
    polyLUT.NanOpacity = 0.0
    polyLUT.NumberOfTableValues = 28
    polyLUT.ScalarRangeInitialized = 1.0
    
    # get color legend/bar for polyLUT in view renderView1
    polyLUTColorBar = GetScalarBar(polyLUT, renderView1)
    polyLUTColorBar.WindowLocation = 'Any Location'
    polyLUTColorBar.Title = 'poly'
    polyLUTColorBar.ComponentTitle = ''
    polyLUTColorBar.HorizontalTitle = 1
    polyLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    polyLUTColorBar.TitleFontSize = 24
    polyLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    polyLUTColorBar.LabelFormat = '%.4f'
    polyLUTColorBar.RangeLabelFormat = '%.3f'
    
    # set color bar visibility
    polyLUTColorBar.Visibility = 0
    
    # get color legend/bar for pGA_01LUT in view renderView1
    pGA_01LUTColorBar = GetScalarBar(pGA_01LUT, renderView1)
    pGA_01LUTColorBar.WindowLocation = 'Any Location'
    pGA_01LUTColorBar.Position = [0.7439194248756514, 0.05562340966921131]
    pGA_01LUTColorBar.Title = 'PGA$_{10\\%-50yr}$'
    pGA_01LUTColorBar.ComponentTitle = ''
    pGA_01LUTColorBar.HorizontalTitle = 1
    pGA_01LUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    pGA_01LUTColorBar.TitleFontSize = 24
    pGA_01LUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    pGA_01LUTColorBar.LabelFontSize = 20
    pGA_01LUTColorBar.ScalarBarLength = 0.32999999999999996
    pGA_01LUTColorBar.DrawScalarBarOutline = 1
    pGA_01LUTColorBar.ScalarBarOutlineColor = [0.0, 0.0, 0.0]
    pGA_01LUTColorBar.AutomaticLabelFormat = 0
    pGA_01LUTColorBar.LabelFormat = '%.1f'
    pGA_01LUTColorBar.RangeLabelFormat = '%.1f'
    
    # set color bar visibility
    pGA_01LUTColorBar.Visibility = 1
    
    # get 2D transfer function for 'mask'
    maskTF2D = GetTransferFunction2D('mask')
    
    # get color transfer function/color map for 'mask'
    maskLUT = GetColorTransferFunction('mask')
    maskLUT.AutomaticRescaleRangeMode = 'Clamp and update every timestep'
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
    vtkBlockColorsLUT.AutomaticRescaleRangeMode = 'Clamp and update every timestep'
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
    
    # show color legend
    m_statsvtiDisplay.SetScalarBarVisibility(renderView1, True)
    
    # show color legend
    pua_3_statsvtiDisplay.SetScalarBarVisibility(renderView1, True)
    
    # hide data in view
    Hide(pua_3_statsvti, renderView1)
    
    # show color legend
    fe_statsvtiDisplay.SetScalarBarVisibility(renderView1, True)
    
    # hide data in view
    Hide(fe_statsvti, renderView1)
    
    # ----------------------------------------------------------------
    # setup color maps and opacity mapes used in the visualization
    # note: the Get..() functions create a new object, if needed
    # ----------------------------------------------------------------
    
    # get opacity transfer function/opacity map for 'poly'
    polyPWF = GetOpacityTransferFunction('poly')
    polyPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.18241043016314507, 0.5955882668495178, 0.5, 0.0, 1.1074918806552887, 0.625, 0.5, 0.0, 1.2117264568805695, 0.8897058963775635, 0.5, 0.0, 2.710097908973694, 0.9705882668495178, 0.5, 0.0, 12.0, 1.0, 0.5, 0.0]
    polyPWF.ScalarRangeInitialized = 1
    
    # get opacity transfer function/opacity map for 'rate'
    ratePWF = GetOpacityTransferFunction('rate')
    ratePWF.Points = [2.3055461497278884e-05, 0.0, 0.5, 0.0, 7.433837329402571e-05, 0.5955882668495178, 0.5, 0.0, 0.00033441599269311837, 0.625, 0.5, 0.0, 0.00036372052628777887, 0.8897058963775635, 0.5, 0.0, 0.0007849730333276991, 0.9705882668495178, 0.5, 0.0, 0.0033967383205890656, 1.0, 0.5, 0.0]
    ratePWF.ScalarRangeInitialized = 1
    
    # get opacity transfer function/opacity map for 'mask'
    maskPWF = GetOpacityTransferFunction('mask')
    maskPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    maskPWF.ScalarRangeInitialized = 1
    
    # get opacity transfer function/opacity map for 'rate_learning'
    rate_learningPWF = GetOpacityTransferFunction('rate_learning')
    rate_learningPWF.Points = [0.006257875356823206, 0.0, 0.5, 0.0, 0.020166512673262894, 0.5955882668495178, 0.5, 0.0, 0.09070317207126703, 0.625, 0.5, 0.0, 0.09865096823212019, 0.8897058963775635, 0.5, 0.0, 0.21290049373255954, 0.9705882668495178, 0.5, 0.0, 0.9212474822998047, 1.0, 0.5, 0.0]
    rate_learningPWF.ScalarRangeInitialized = 1
    
    # get opacity transfer function/opacity map for 'vtkBlockColors'
    vtkBlockColorsPWF = GetOpacityTransferFunction('vtkBlockColors')
    vtkBlockColorsPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    
    # get opacity transfer function/opacity map for 'params'
    paramsPWF = GetOpacityTransferFunction('params')
    paramsPWF.Points = [5e-06, 0.0, 0.5, 0.0, 5.6517097627776626e-05, 0.5955882668495178, 0.5, 0.0, 0.0003177823737198556, 0.625, 0.5, 0.0, 0.00034722072784678233, 0.8897058963775635, 0.5, 0.0, 0.0007703969042919331, 0.9705882668495178, 0.5, 0.0, 0.003394088940694928, 1.0, 0.5, 0.0]
    paramsPWF.ScalarRangeInitialized = 1
    
    # get opacity transfer function/opacity map for 'rates_bin'
    rates_binPWF = GetOpacityTransferFunction('rates_bin')
    rates_binPWF.Points = [1.5241377582242421e-07, 0.0, 0.5, 0.0, 1.8303754926081116e-07, 0.5955882668495178, 0.5, 0.0, 3.383438260268289e-07, 0.625, 0.5, 0.0, 3.558431326394735e-07, 0.8897058963775635, 0.5, 0.0, 6.073955676313055e-07, 0.9705882668495178, 0.5, 0.0, 2.1670205114787677e-06, 1.0, 0.5, 0.0]
    rates_binPWF.ScalarRangeInitialized = 1
    
    # get opacity transfer function/opacity map for 'bin'
    binPWF = GetOpacityTransferFunction('bin')
    binPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.045602607540786266, 0.5955882668495178, 0.5, 0.0, 0.2768729701638222, 0.625, 0.5, 0.0, 0.30293161422014236, 0.8897058963775635, 0.5, 0.0, 0.6775244772434235, 0.9705882668495178, 0.5, 0.0, 3.0, 1.0, 0.5, 0.0]
    binPWF.ScalarRangeInitialized = 1
    coastlinevtmDisplay.LineWidth = size_factor

    # ----------------------------------------------------------------
    # restore active source
    SetActiveSource(m_statsvti)



    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='extracts')

    SaveScreenshot('figure_m.png',
               renderView1,
               ImageResolution=[size_factor * i for i in renderView1.ViewSize])


def makefig_fe(size_factor):

    # get the material library
    materialLibrary1 = GetMaterialLibrary()
    
    # Create a new 'Render View'
    renderView1 = CreateView('RenderView')
    renderView1.ViewSize = [534, 786]
    renderView1.InteractionMode = '2D'
    renderView1.AxesGrid = 'GridAxes3DActor'
    renderView1.OrientationAxesVisibility = 0
    renderView1.CenterOfRotation = [1575927.8735133181, 5400949.511356351, -10.0]
    renderView1.StereoType = 'Crystal Eyes'
    renderView1.CameraPosition = [1590343.8998113205, 5462560.284089721, 9266090.0]
    renderView1.CameraFocalPoint = [1590343.8998113205, 5462560.284089721, -10.0]
    renderView1.CameraFocalDisk = 1.0
    renderView1.CameraParallelScale = 744957.9411375008
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
    layout1.SetSize(534, 786)
    
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
    basemap_2193vti = XMLImageDataReader(registrationName='basemap_2193.vti', FileName=['paraview/basemap_2193.vti'])
    basemap_2193vti.CellArrayStatus = ['basemap', 'mask']
    basemap_2193vti.TimeArray = 'None'
    
    # create a new 'XML Image Data Reader'
    pua_3_statsvti = XMLImageDataReader(registrationName='pua_3_stats.vti', FileName=['paraview/pua_3_stats.vti'])
    pua_3_statsvti.CellArrayStatus = ['PGA_0.1', 'mask']
    pua_3_statsvti.TimeArray = 'None'
    
    # create a new 'XML Image Data Reader'
    fe_statsvti = XMLImageDataReader(registrationName='fe_stats.vti', FileName=['paraview/fe_stats.vti'])
    fe_statsvti.CellArrayStatus = ['PGA_0.1', 'mask']
    fe_statsvti.TimeArray = 'None'
    
    # create a new 'XML Image Data Reader'
    m_statsvti = XMLImageDataReader(registrationName='m_stats.vti', FileName=['paraview/m_stats.vti'])
    m_statsvti.CellArrayStatus = ['PGA_0.1', 'mask']
    m_statsvti.TimeArray = 'None'
    
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
    
    # show data from fe_statsvti
    fe_statsvtiDisplay = Show(fe_statsvti, renderView1, 'UniformGridRepresentation')
    
    # get 2D transfer function for 'PGA_01'
    pGA_01TF2D = GetTransferFunction2D('PGA_01')
    pGA_01TF2D.ScalarRangeInitialized = 1
    pGA_01TF2D.Range = [0.0, 0.6, 0.0, 1.0]
    
    # get color transfer function/color map for 'PGA_01'
    pGA_01LUT = GetColorTransferFunction('PGA_01')
    pGA_01LUT.AutomaticRescaleRangeMode = 'Never'
    pGA_01LUT.EnableOpacityMapping = 1
    pGA_01LUT.TransferFunction2D = pGA_01TF2D
    pGA_01LUT.RGBPoints = [0.0, 0.278431372549, 0.278431372549, 0.858823529412, 0.08579999999999997, 0.0, 0.0, 0.360784313725, 0.171, 0.0, 1.0, 1.0, 0.2573999999999999, 0.0, 0.501960784314, 0.0, 0.34259999999999996, 1.0, 1.0, 0.0, 0.42840000000000006, 1.0, 0.380392156863, 0.0, 0.5142, 0.419607843137, 0.0, 0.0, 0.6, 0.878431372549, 0.301960784314, 0.301960784314]
    pGA_01LUT.ColorSpace = 'RGB'
    pGA_01LUT.NanOpacity = 0.0
    pGA_01LUT.NumberOfTableValues = 526
    pGA_01LUT.ScalarRangeInitialized = 1.0
    pGA_01LUT.VectorMode = 'Component'
    
    # get opacity transfer function/opacity map for 'PGA_01'
    pGA_01PWF = GetOpacityTransferFunction('PGA_01')
    pGA_01PWF.Points = [0.0, 0.0, 0.5, 0.0, 0.027000000700354576, 0.2663043439388275, 0.5, 0.0, 0.055800002068281174, 0.45652174949645996, 0.5, 0.0, 0.09240000694990158, 0.635869562625885, 0.5, 0.0, 0.131400004029274, 0.7771739363670349, 0.5, 0.0, 0.19740000367164612, 0.8586956858634949, 0.5, 0.0, 0.6, 1.0, 0.5, 0.0]
    pGA_01PWF.ScalarRangeInitialized = 1

    # trace defaults for the display properties.
    fe_statsvtiDisplay.Representation = 'Slice'
    fe_statsvtiDisplay.ColorArrayName = ['CELLS', 'PGA_0.1']
    fe_statsvtiDisplay.LookupTable = pGA_01LUT
    fe_statsvtiDisplay.SelectTCoordArray = 'None'
    fe_statsvtiDisplay.SelectNormalArray = 'None'
    fe_statsvtiDisplay.SelectTangentArray = 'None'
    fe_statsvtiDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    fe_statsvtiDisplay.SelectOrientationVectors = 'None'
    fe_statsvtiDisplay.ScaleFactor = 154400.0
    fe_statsvtiDisplay.SelectScaleArray = 'None'
    fe_statsvtiDisplay.GlyphType = 'Arrow'
    fe_statsvtiDisplay.GlyphTableIndexArray = 'None'
    fe_statsvtiDisplay.GaussianRadius = 7720.0
    fe_statsvtiDisplay.SetScaleArray = [None, '']
    fe_statsvtiDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    fe_statsvtiDisplay.OpacityArray = [None, '']
    fe_statsvtiDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    fe_statsvtiDisplay.DataAxesGrid = 'GridAxesRepresentation'
    fe_statsvtiDisplay.PolarAxes = 'PolarAxesRepresentation'
    fe_statsvtiDisplay.ScalarOpacityUnitDistance = 25222.71540033692
    fe_statsvtiDisplay.ScalarOpacityFunction = pGA_01PWF
    fe_statsvtiDisplay.TransferFunction2D = pGA_01TF2D
    fe_statsvtiDisplay.OpacityArrayName = ['CELLS', 'PGA_0.1']
    fe_statsvtiDisplay.ColorArray2Name = ['CELLS', 'PGA_0.1']
    fe_statsvtiDisplay.SliceFunction = 'Plane'
    fe_statsvtiDisplay.SelectInputVectors = [None, '']
    fe_statsvtiDisplay.WriteLog = ''
    
    # init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
    fe_statsvtiDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    
    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    fe_statsvtiDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    
    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    fe_statsvtiDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    
    # init the 'Plane' selected for 'SliceFunction'
    fe_statsvtiDisplay.SliceFunction.Origin = [1590196.015518637, 5469386.7749801995, 10.0]
    
    # setup the color legend parameters for each legend in this view
    
    # get color legend/bar for pGA_01LUT in view renderView1
    pGA_01LUTColorBar = GetScalarBar(pGA_01LUT, renderView1)
    pGA_01LUTColorBar.WindowLocation = 'Any Location'
    pGA_01LUTColorBar.Position = [0.7439194248756514, 0.05562340966921131]
    pGA_01LUTColorBar.Title = 'PGA$_{10\\%-50yr}$'
    pGA_01LUTColorBar.ComponentTitle = ''
    pGA_01LUTColorBar.HorizontalTitle = 1
    pGA_01LUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    pGA_01LUTColorBar.TitleFontSize = 24
    pGA_01LUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    pGA_01LUTColorBar.LabelFontSize = 20
    pGA_01LUTColorBar.ScalarBarLength = 0.32999999999999996
    pGA_01LUTColorBar.DrawScalarBarOutline = 1
    pGA_01LUTColorBar.ScalarBarOutlineColor = [0.0, 0.0, 0.0]
    pGA_01LUTColorBar.AutomaticLabelFormat = 0
    pGA_01LUTColorBar.LabelFormat = '%.1f'
    pGA_01LUTColorBar.RangeLabelFormat = '%.1f'
    
    # set color bar visibility
    pGA_01LUTColorBar.Visibility = 1
    
    # show color legend
    fe_statsvtiDisplay.SetScalarBarVisibility(renderView1, False)

    # ----------------------------------------------------------------
    # setup color maps and opacity mapes used in the visualization
    # note: the Get..() functions create a new object, if needed
    # ----------------------------------------------------------------
    
    # ----------------------------------------------------------------
    # restore active source
    SetActiveSource(fe_statsvti)
    # ----------------------------------------------------------------
    coastlinevtmDisplay.LineWidth = size_factor

    # generate extracts

    SaveExtracts(ExtractsOutputDirectory='extracts')
    SaveScreenshot('figure_fe.png',
                   renderView1,
                   ImageResolution=[size_factor * i for i in renderView1.ViewSize])


def makefig_urz(size_factor):
    
    # get the material library
    materialLibrary1 = GetMaterialLibrary()
    
    # Create a new 'Render View'
    renderView1 = CreateView('RenderView')
    renderView1.ViewSize = [534, 786]
    renderView1.InteractionMode = '2D'
    renderView1.AxesGrid = 'GridAxes3DActor'
    renderView1.OrientationAxesVisibility = 0
    renderView1.CenterOfRotation = [1575927.8735133181, 5400949.511356351, -10.0]
    renderView1.StereoType = 'Crystal Eyes'
    renderView1.CameraPosition = [1590343.8998113205, 5462560.284089721, 9266090.0]
    renderView1.CameraFocalPoint = [1590343.8998113205, 5462560.284089721, -10.0]
    renderView1.CameraFocalDisk = 1.0
    renderView1.CameraParallelScale = 744957.9411375008
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
    layout1.SetSize(534, 786)
    
    # ----------------------------------------------------------------
    # restore active view
    SetActiveView(renderView1)
    # ----------------------------------------------------------------
    
    # ----------------------------------------------------------------
    # setup the data processing pipelines
    # ----------------------------------------------------------------
    
    # create a new 'XML Image Data Reader'
    fe_statsvti = XMLImageDataReader(registrationName='fe_stats.vti', FileName=['paraview/fe_stats.vti'])
    fe_statsvti.CellArrayStatus = ['PGA_0.1', 'mask']
    fe_statsvti.TimeArray = 'None'
    
    # create a new 'XML MultiBlock Data Reader'
    coastlinevtm = XMLMultiBlockDataReader(registrationName='coastline.vtm', FileName=['paraview/coastline.vtm'])
    coastlinevtm.CellArrayStatus = ['t50_fid', 'elevation']
    coastlinevtm.TimeArray = 'None'
    
    # create a new 'XML Image Data Reader'
    m_statsvti = XMLImageDataReader(registrationName='m_stats.vti', FileName=['paraview/m_stats.vti'])
    m_statsvti.CellArrayStatus = ['PGA_0.1', 'mask']
    m_statsvti.TimeArray = 'None'
    
    # create a new 'XML Image Data Reader'
    pua_3_statsvti = XMLImageDataReader(registrationName='pua_3_stats.vti', FileName=['paraview/pua_3_stats.vti'])
    pua_3_statsvti.CellArrayStatus = ['PGA_0.1', 'mask']
    pua_3_statsvti.TimeArray = 'None'
    
    # create a new 'XML Image Data Reader'
    basemap_2193vti = XMLImageDataReader(registrationName='basemap_2193.vti', FileName=['paraview/basemap_2193.vti'])
    basemap_2193vti.CellArrayStatus = ['basemap', 'mask']
    basemap_2193vti.TimeArray = 'None'
    
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
    
    # show data from m_statsvti
    m_statsvtiDisplay = Show(m_statsvti, renderView1, 'UniformGridRepresentation')
    
    # get 2D transfer function for 'PGA_01'
    pGA_01TF2D = GetTransferFunction2D('PGA_01')
    pGA_01TF2D.ScalarRangeInitialized = 1
    pGA_01TF2D.Range = [0.0, 0.6, 0.0, 1.0]
    
    # get color transfer function/color map for 'PGA_01'
    pGA_01LUT = GetColorTransferFunction('PGA_01')
    pGA_01LUT.AutomaticRescaleRangeMode = 'Never'
    pGA_01LUT.EnableOpacityMapping = 1
    pGA_01LUT.TransferFunction2D = pGA_01TF2D
    pGA_01LUT.RGBPoints = [0.0, 0.278431372549, 0.278431372549, 0.858823529412, 0.08579999999999997, 0.0, 0.0, 0.360784313725, 0.171, 0.0, 1.0, 1.0, 0.2573999999999999, 0.0, 0.501960784314, 0.0, 0.34259999999999996, 1.0, 1.0, 0.0, 0.42840000000000006, 1.0, 0.380392156863, 0.0, 0.5142, 0.419607843137, 0.0, 0.0, 0.6, 0.878431372549, 0.301960784314, 0.301960784314]
    pGA_01LUT.ColorSpace = 'RGB'
    pGA_01LUT.NanOpacity = 0.0
    pGA_01LUT.NumberOfTableValues = 526
    pGA_01LUT.ScalarRangeInitialized = 1.0
    pGA_01LUT.VectorMode = 'Component'
    
    # get opacity transfer function/opacity map for 'PGA_01'
    pGA_01PWF = GetOpacityTransferFunction('PGA_01')
    pGA_01PWF.Points = [0.0, 0.0, 0.5, 0.0, 0.027000000700354576, 0.2663043439388275, 0.5, 0.0, 0.055800002068281174, 0.45652174949645996, 0.5, 0.0, 0.09240000694990158, 0.635869562625885, 0.5, 0.0, 0.131400004029274, 0.7771739363670349, 0.5, 0.0, 0.19740000367164612, 0.8586956858634949, 0.5, 0.0, 0.6, 1.0, 0.5, 0.0]
    pGA_01PWF.ScalarRangeInitialized = 1
    
    # trace defaults for the display properties.
    m_statsvtiDisplay.Representation = 'Slice'
    m_statsvtiDisplay.ColorArrayName = ['CELLS', 'PGA_0.1']
    m_statsvtiDisplay.LookupTable = pGA_01LUT
    m_statsvtiDisplay.SelectTCoordArray = 'None'
    m_statsvtiDisplay.SelectNormalArray = 'None'
    m_statsvtiDisplay.SelectTangentArray = 'None'
    m_statsvtiDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    m_statsvtiDisplay.SelectOrientationVectors = 'None'
    m_statsvtiDisplay.ScaleFactor = 154400.0
    m_statsvtiDisplay.SelectScaleArray = 'None'
    m_statsvtiDisplay.GlyphType = 'Arrow'
    m_statsvtiDisplay.GlyphTableIndexArray = 'None'
    m_statsvtiDisplay.GaussianRadius = 7720.0
    m_statsvtiDisplay.SetScaleArray = [None, '']
    m_statsvtiDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    m_statsvtiDisplay.OpacityArray = [None, '']
    m_statsvtiDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    m_statsvtiDisplay.DataAxesGrid = 'GridAxesRepresentation'
    m_statsvtiDisplay.PolarAxes = 'PolarAxesRepresentation'
    m_statsvtiDisplay.ScalarOpacityUnitDistance = 25222.71540033692
    m_statsvtiDisplay.ScalarOpacityFunction = pGA_01PWF
    m_statsvtiDisplay.TransferFunction2D = pGA_01TF2D
    m_statsvtiDisplay.OpacityArrayName = ['CELLS', 'PGA_0.1']
    m_statsvtiDisplay.ColorArray2Name = ['CELLS', 'PGA_0.1']
    m_statsvtiDisplay.SliceFunction = 'Plane'
    m_statsvtiDisplay.SelectInputVectors = [None, '']
    m_statsvtiDisplay.WriteLog = ''
    
    # init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
    m_statsvtiDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    
    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    m_statsvtiDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    
    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    m_statsvtiDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    
    # init the 'Plane' selected for 'SliceFunction'
    m_statsvtiDisplay.SliceFunction.Origin = [1590196.015518637, 5469386.7749801995, 10.0]
    
    # show data from pua_3_statsvti
    pua_3_statsvtiDisplay = Show(pua_3_statsvti, renderView1, 'UniformGridRepresentation')
    
    # trace defaults for the display properties.
    pua_3_statsvtiDisplay.Representation = 'Slice'
    pua_3_statsvtiDisplay.ColorArrayName = ['CELLS', 'PGA_0.1']
    pua_3_statsvtiDisplay.LookupTable = pGA_01LUT
    pua_3_statsvtiDisplay.SelectTCoordArray = 'None'
    pua_3_statsvtiDisplay.SelectNormalArray = 'None'
    pua_3_statsvtiDisplay.SelectTangentArray = 'None'
    pua_3_statsvtiDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    pua_3_statsvtiDisplay.SelectOrientationVectors = 'None'
    pua_3_statsvtiDisplay.ScaleFactor = 154400.0
    pua_3_statsvtiDisplay.SelectScaleArray = 'None'
    pua_3_statsvtiDisplay.GlyphType = 'Arrow'
    pua_3_statsvtiDisplay.GlyphTableIndexArray = 'None'
    pua_3_statsvtiDisplay.GaussianRadius = 7720.0
    pua_3_statsvtiDisplay.SetScaleArray = [None, '']
    pua_3_statsvtiDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    pua_3_statsvtiDisplay.OpacityArray = [None, '']
    pua_3_statsvtiDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    pua_3_statsvtiDisplay.DataAxesGrid = 'GridAxesRepresentation'
    pua_3_statsvtiDisplay.PolarAxes = 'PolarAxesRepresentation'
    pua_3_statsvtiDisplay.ScalarOpacityUnitDistance = 25222.71540033692
    pua_3_statsvtiDisplay.ScalarOpacityFunction = pGA_01PWF
    pua_3_statsvtiDisplay.TransferFunction2D = pGA_01TF2D
    pua_3_statsvtiDisplay.OpacityArrayName = ['CELLS', 'PGA_0.1']
    pua_3_statsvtiDisplay.ColorArray2Name = ['CELLS', 'PGA_0.1']
    pua_3_statsvtiDisplay.SliceFunction = 'Plane'
    pua_3_statsvtiDisplay.SelectInputVectors = [None, '']
    pua_3_statsvtiDisplay.WriteLog = ''
    
    # init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
    pua_3_statsvtiDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    
    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    pua_3_statsvtiDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    
    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    pua_3_statsvtiDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    
    # init the 'Plane' selected for 'SliceFunction'
    pua_3_statsvtiDisplay.SliceFunction.Origin = [1590196.015518637, 5469386.7749801995, 10.0]
    
    # show data from fe_statsvti
    fe_statsvtiDisplay = Show(fe_statsvti, renderView1, 'UniformGridRepresentation')
    
    # trace defaults for the display properties.
    fe_statsvtiDisplay.Representation = 'Slice'
    fe_statsvtiDisplay.ColorArrayName = ['CELLS', 'PGA_0.1']
    fe_statsvtiDisplay.LookupTable = pGA_01LUT
    fe_statsvtiDisplay.SelectTCoordArray = 'None'
    fe_statsvtiDisplay.SelectNormalArray = 'None'
    fe_statsvtiDisplay.SelectTangentArray = 'None'
    fe_statsvtiDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    fe_statsvtiDisplay.SelectOrientationVectors = 'None'
    fe_statsvtiDisplay.ScaleFactor = 154400.0
    fe_statsvtiDisplay.SelectScaleArray = 'None'
    fe_statsvtiDisplay.GlyphType = 'Arrow'
    fe_statsvtiDisplay.GlyphTableIndexArray = 'None'
    fe_statsvtiDisplay.GaussianRadius = 7720.0
    fe_statsvtiDisplay.SetScaleArray = [None, '']
    fe_statsvtiDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    fe_statsvtiDisplay.OpacityArray = [None, '']
    fe_statsvtiDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    fe_statsvtiDisplay.DataAxesGrid = 'GridAxesRepresentation'
    fe_statsvtiDisplay.PolarAxes = 'PolarAxesRepresentation'
    fe_statsvtiDisplay.ScalarOpacityUnitDistance = 25222.71540033692
    fe_statsvtiDisplay.ScalarOpacityFunction = pGA_01PWF
    fe_statsvtiDisplay.TransferFunction2D = pGA_01TF2D
    fe_statsvtiDisplay.OpacityArrayName = ['CELLS', 'PGA_0.1']
    fe_statsvtiDisplay.ColorArray2Name = ['CELLS', 'PGA_0.1']
    fe_statsvtiDisplay.SliceFunction = 'Plane'
    fe_statsvtiDisplay.SelectInputVectors = [None, '']
    fe_statsvtiDisplay.WriteLog = ''
    
    # init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
    fe_statsvtiDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    
    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    fe_statsvtiDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    
    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    fe_statsvtiDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    
    # init the 'Plane' selected for 'SliceFunction'
    fe_statsvtiDisplay.SliceFunction.Origin = [1590196.015518637, 5469386.7749801995, 10.0]
    
    # setup the color legend parameters for each legend in this view
    
    # get 2D transfer function for 'bin'
    binTF2D = GetTransferFunction2D('bin')
    binTF2D.ScalarRangeInitialized = 1
    binTF2D.Range = [0.0, 3.0, 0.0, 1.0]
    
    # get color transfer function/color map for 'bin'
    binLUT = GetColorTransferFunction('bin')
    binLUT.AutomaticRescaleRangeMode = 'Never'
    binLUT.EnableOpacityMapping = 1
    binLUT.TransferFunction2D = binTF2D
    binLUT.RGBPoints = [0.0, 0.8901960784313725, 0.9568627450980393, 0.984313725490196, 0.049763931135892725, 0.807843137254902, 0.9019607843137255, 0.9607843137254902, 0.09952786227178578, 0.7607843137254902, 0.8588235294117647, 0.9294117647058824, 0.19905572454357157, 0.6862745098039216, 0.7843137254901961, 0.8705882352941177, 0.298583586815357, 0.6, 0.6980392156862745, 0.8, 0.39811144908714313, 0.5176470588235295, 0.615686274509804, 0.7254901960784313, 0.4976393113589286, 0.44313725490196076, 0.5333333333333333, 0.6431372549019608, 0.5971671736307144, 0.37254901960784315, 0.4588235294117647, 0.5686274509803921, 0.6966950359025001, 0.3058823529411765, 0.38823529411764707, 0.49411764705882355, 0.7962228981742859, 0.24705882352941178, 0.3176470588235294, 0.41568627450980394, 0.8957507604460717, 0.2, 0.2549019607843137, 0.34509803921568627, 0.9952786227178572, 0.14902, 0.196078, 0.278431, 1.046370953123331, 0.2, 0.1450980392156863, 0.13725490196078433, 1.0974332835288048, 0.23137254901960785, 0.17254901960784313, 0.16470588235294117, 1.1484956139342786, 0.2549019607843137, 0.2, 0.1843137254901961, 1.1995579443397522, 0.28627450980392155, 0.23137254901960785, 0.21176470588235294, 1.2506202747452257, 0.3176470588235294, 0.2627450980392157, 0.23921568627450981, 1.3016826051506996, 0.34509803921568627, 0.29411764705882354, 0.26666666666666666, 1.3527449355561734, 0.37254901960784315, 0.3215686274509804, 0.29411764705882354, 1.403807265961647, 0.403921568627451, 0.3568627450980392, 0.3176470588235294, 1.4548695963671205, 0.43137254901960786, 0.38823529411764707, 0.34509803921568627, 1.5059319267725944, 0.4627450980392157, 0.4196078431372549, 0.37254901960784315, 1.5569942571780682, 0.4980392156862745, 0.4588235294117647, 0.40784313725490196, 1.6080565875835418, 0.5372549019607843, 0.5058823529411764, 0.44313725490196076, 1.6591189179890158, 0.5803921568627451, 0.5568627450980392, 0.48627450980392156, 1.7101812483944892, 0.6313725490196078, 0.6078431372549019, 0.5333333333333333, 1.7612435787999632, 0.6784313725490196, 0.6627450980392157, 0.5803921568627451, 1.8123059092054365, 0.7254901960784313, 0.7137254901960784, 0.6274509803921569, 1.8633682396109106, 0.7764705882352941, 0.7647058823529411, 0.6784313725490196, 1.914430570016384, 0.8274509803921568, 0.8235294117647058, 0.7372549019607844, 1.965492900421858, 0.8901960784313725, 0.8901960784313725, 0.807843137254902, 2.0165552308273313, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 2.0165552308273313, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 2.0165852308273315, 0.9882352941176471, 0.9882352941176471, 0.8588235294117647, 2.018552060365677, 0.984313725490196, 0.9882352941176471, 0.8549019607843137, 2.0657559692859646, 0.9882352941176471, 0.9882352941176471, 0.7568627450980392, 2.1149267077445986, 0.9803921568627451, 0.9725490196078431, 0.6705882352941176, 2.1640974462032316, 0.9803921568627451, 0.9490196078431372, 0.596078431372549, 2.2132681846618656, 0.9725490196078431, 0.9176470588235294, 0.5137254901960784, 2.2624389231204987, 0.9686274509803922, 0.8784313725490196, 0.4235294117647059, 2.3116096615791317, 0.9647058823529412, 0.8274509803921568, 0.3254901960784314, 2.3607804000377657, 0.9529411764705882, 0.7686274509803922, 0.24705882352941178, 2.4099511384963987, 0.9450980392156862, 0.7098039215686275, 0.1843137254901961, 2.4591218769550327, 0.9372549019607843, 0.6431372549019608, 0.12156862745098039, 2.5082926154136658, 0.9254901960784314, 0.5803921568627451, 0.0784313725490196, 2.557463353872299, 0.9176470588235294, 0.5215686274509804, 0.047058823529411764, 2.606634092330933, 0.9019607843137255, 0.4588235294117647, 0.027450980392156862, 2.655804830789566, 0.8627450980392157, 0.3803921568627451, 0.011764705882352941, 2.7049755692482, 0.788235294117647, 0.2901960784313726, 0.0, 2.754146307706833, 0.7411764705882353, 0.22745098039215686, 0.00392156862745098, 2.803317046165466, 0.6627450980392157, 0.16862745098039217, 0.01568627450980392, 2.8524877846240995, 0.5882352941176471, 0.11372549019607843, 0.023529411764705882, 2.9016585230827334, 0.49411764705882355, 0.054901960784313725, 0.03529411764705882, 2.950829261541367, 0.396078431372549, 0.0392156862745098, 0.058823529411764705, 3.0, 0.301961, 0.047059, 0.090196]
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
    
    # get 2D transfer function for 'rate'
    rateTF2D = GetTransferFunction2D('rate')
    rateTF2D.ScalarRangeInitialized = 1
    rateTF2D.Range = [2.3055461497278884e-05, 0.003394088940694928, 0.0, 1.0]
    
    # get color transfer function/color map for 'rate'
    rateLUT = GetColorTransferFunction('rate')
    rateLUT.AutomaticRescaleRangeMode = 'Never'
    rateLUT.EnableOpacityMapping = 1
    rateLUT.TransferFunction2D = rateTF2D
    rateLUT.RGBPoints = [2.3055461497278884e-05, 0.8901960784313725, 0.9568627450980393, 0.984313725490196, 7.901803532200734e-05, 0.807843137254902, 0.9019607843137255, 0.9607843137254902, 0.00013498060914673616, 0.7607843137254902, 0.8588235294117647, 0.9294117647058824, 0.00024690575679619343, 0.6862745098039216, 0.7843137254901961, 0.8705882352941177, 0.00035883090444565035, 0.6, 0.6980392156862745, 0.8, 0.000470756052095108, 0.5176470588235295, 0.615686274509804, 0.7254901960784313, 0.0005826811997445649, 0.44313725490196076, 0.5333333333333333, 0.6431372549019608, 0.0006946063473940221, 0.37254901960784315, 0.4588235294117647, 0.5686274509803921, 0.0008065314950434794, 0.3058823529411765, 0.38823529411764707, 0.49411764705882355, 0.0009184566426929367, 0.24705882352941178, 0.3176470588235294, 0.41568627450980394, 0.001030381790342394, 0.2, 0.2549019607843137, 0.34509803921568627, 0.001142306937991851, 0.14902, 0.196078, 0.278431, 0.0011997633777651848, 0.2, 0.1450980392156863, 0.13725490196078433, 0.0012571860807099274, 0.23137254901960785, 0.17254901960784313, 0.16470588235294117, 0.00131460878365467, 0.2549019607843137, 0.2, 0.1843137254901961, 0.0013720314865994126, 0.28627450980392155, 0.23137254901960785, 0.21176470588235294, 0.0014294541895441555, 0.3176470588235294, 0.2627450980392157, 0.23921568627450981, 0.0014868768924888981, 0.34509803921568627, 0.29411764705882354, 0.26666666666666666, 0.0015442995954336408, 0.37254901960784315, 0.3215686274509804, 0.29411764705882354, 0.0016017222983783833, 0.403921568627451, 0.3568627450980392, 0.3176470588235294, 0.001659145001323126, 0.43137254901960786, 0.38823529411764707, 0.34509803921568627, 0.0017165677042678686, 0.4627450980392157, 0.4196078431372549, 0.37254901960784315, 0.0017739904072126113, 0.4980392156862745, 0.4588235294117647, 0.40784313725490196, 0.001831413110157354, 0.5372549019607843, 0.5058823529411764, 0.44313725490196076, 0.0018888358131020969, 0.5803921568627451, 0.5568627450980392, 0.48627450980392156, 0.0019462585160468393, 0.6313725490196078, 0.6078431372549019, 0.5333333333333333, 0.0020036812189915825, 0.6784313725490196, 0.6627450980392157, 0.5803921568627451, 0.0020611039219363245, 0.7254901960784313, 0.7137254901960784, 0.6274509803921569, 0.002118526624881068, 0.7764705882352941, 0.7647058823529411, 0.6784313725490196, 0.00217594932782581, 0.8274509803921568, 0.8235294117647058, 0.7372549019607844, 0.0022333720307705527, 0.8901960784313725, 0.8901960784313725, 0.807843137254902, 0.002290794733715295, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 0.002290794733715295, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 0.002290828470543887, 0.9882352941176471, 0.9882352941176471, 0.8588235294117647, 0.002293040290243977, 0.984313725490196, 0.9882352941176471, 0.8549019607843137, 0.0023461239630461453, 0.9882352941176471, 0.9882352941176471, 0.7568627450980392, 0.0024014194555484047, 0.9803921568627451, 0.9725490196078431, 0.6705882352941176, 0.002456714948050663, 0.9803921568627451, 0.9490196078431372, 0.596078431372549, 0.0025120104405529226, 0.9725490196078431, 0.9176470588235294, 0.5137254901960784, 0.002567305933055181, 0.9686274509803922, 0.8784313725490196, 0.4235294117647059, 0.00262260142555744, 0.9647058823529412, 0.8274509803921568, 0.3254901960784314, 0.0026778969180596994, 0.9529411764705882, 0.7686274509803922, 0.24705882352941178, 0.002733192410561958, 0.9450980392156862, 0.7098039215686275, 0.1843137254901961, 0.0027884879030642172, 0.9372549019607843, 0.6431372549019608, 0.12156862745098039, 0.002843783395566476, 0.9254901960784314, 0.5803921568627451, 0.0784313725490196, 0.0028990788880687347, 0.9176470588235294, 0.5215686274509804, 0.047058823529411764, 0.002954374380570994, 0.9019607843137255, 0.4588235294117647, 0.027450980392156862, 0.0030096698730732526, 0.8627450980392157, 0.3803921568627451, 0.011764705882352941, 0.003064965365575512, 0.788235294117647, 0.2901960784313726, 0.0, 0.003120260858077771, 0.7411764705882353, 0.22745098039215686, 0.00392156862745098, 0.0031755563505800294, 0.6627450980392157, 0.16862745098039217, 0.01568627450980392, 0.0032308518430822887, 0.5882352941176471, 0.11372549019607843, 0.023529411764705882, 0.003286147335584548, 0.49411764705882355, 0.054901960784313725, 0.03529411764705882, 0.0033414428280868066, 0.396078431372549, 0.0392156862745098, 0.058823529411764705, 0.0033967383205890656, 0.301961, 0.047059, 0.090196]
    rateLUT.ColorSpace = 'Lab'
    rateLUT.NanColor = [0.25, 0.0, 0.0]
    rateLUT.NanOpacity = 0.0
    rateLUT.NumberOfTableValues = 10
    rateLUT.ScalarRangeInitialized = 1.0
    
    # get color legend/bar for rateLUT in view renderView1
    rateLUTColorBar = GetScalarBar(rateLUT, renderView1)
    rateLUTColorBar.WindowLocation = 'Any Location'
    rateLUTColorBar.Title = 'rate'
    rateLUTColorBar.ComponentTitle = ''
    rateLUTColorBar.HorizontalTitle = 1
    rateLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    rateLUTColorBar.TitleFontSize = 24
    rateLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    rateLUTColorBar.LabelFormat = '%.4f'
    rateLUTColorBar.RangeLabelFormat = '%.3f'
    
    # set color bar visibility
    rateLUTColorBar.Visibility = 0
    
    # get 2D transfer function for 'rate_learning'
    rate_learningTF2D = GetTransferFunction2D('rate_learning')
    rate_learningTF2D.ScalarRangeInitialized = 1
    rate_learningTF2D.Range = [0.006257875356823206, 0.9212474822998047, 0.0, 1.0]
    
    # get color transfer function/color map for 'rate_learning'
    rate_learningLUT = GetColorTransferFunction('rate_learning')
    rate_learningLUT.AutomaticRescaleRangeMode = 'Never'
    rate_learningLUT.EnableOpacityMapping = 1
    rate_learningLUT.TransferFunction2D = rate_learningTF2D
    rate_learningLUT.RGBPoints = [0.006257875356823206, 0.8901960784313725, 0.9568627450980393, 0.984313725490196, 0.021435701953479235, 0.807843137254902, 0.9019607843137255, 0.9607843137254902, 0.03661352855013536, 0.7607843137254902, 0.8588235294117647, 0.9294117647058824, 0.06696918174344751, 0.6862745098039216, 0.7843137254901961, 0.8705882352941177, 0.09732483493675957, 0.6, 0.6980392156862745, 0.8, 0.12768048813007182, 0.5176470588235295, 0.615686274509804, 0.7254901960784313, 0.1580361413233839, 0.44313725490196076, 0.5333333333333333, 0.6431372549019608, 0.18839179451669605, 0.37254901960784315, 0.4588235294117647, 0.5686274509803921, 0.2187474477100082, 0.3058823529411765, 0.38823529411764707, 0.49411764705882355, 0.24910310090332036, 0.24705882352941178, 0.3176470588235294, 0.41568627450980394, 0.2794587540966325, 0.2, 0.2549019607843137, 0.34509803921568627, 0.3098144072899446, 0.14902, 0.196078, 0.278431, 0.32539739106177973, 0.2, 0.1450980392156863, 0.13725490196078433, 0.3409712249375454, 0.23137254901960785, 0.17254901960784313, 0.16470588235294117, 0.3565450588133111, 0.2549019607843137, 0.2, 0.1843137254901961, 0.3721188926890768, 0.28627450980392155, 0.23137254901960785, 0.21176470588235294, 0.38769272656484244, 0.3176470588235294, 0.2627450980392157, 0.23921568627450981, 0.4032665604406081, 0.34509803921568627, 0.29411764705882354, 0.26666666666666666, 0.4188403943163738, 0.37254901960784315, 0.3215686274509804, 0.29411764705882354, 0.4344142281921395, 0.403921568627451, 0.3568627450980392, 0.3176470588235294, 0.44998806206790515, 0.43137254901960786, 0.38823529411764707, 0.34509803921568627, 0.4655618959436708, 0.4627450980392157, 0.4196078431372549, 0.37254901960784315, 0.48113572981943653, 0.4980392156862745, 0.4588235294117647, 0.40784313725490196, 0.4967095636952022, 0.5372549019607843, 0.5058823529411764, 0.44313725490196076, 0.512283397570968, 0.5803921568627451, 0.5568627450980392, 0.48627450980392156, 0.5278572314467336, 0.6313725490196078, 0.6078431372549019, 0.5333333333333333, 0.5434310653224993, 0.6784313725490196, 0.6627450980392157, 0.5803921568627451, 0.5590048991982649, 0.7254901960784313, 0.7137254901960784, 0.6274509803921569, 0.5745787330740306, 0.7764705882352941, 0.7647058823529411, 0.6784313725490196, 0.5901525669497962, 0.8274509803921568, 0.8235294117647058, 0.7372549019607844, 0.6057264008255621, 0.8901960784313725, 0.8901960784313725, 0.807843137254902, 0.6213002347013276, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 0.6213002347013276, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 0.6213093845973972, 0.9882352941176471, 0.9882352941176471, 0.8588235294117647, 0.621909260792802, 0.984313725490196, 0.9882352941176471, 0.8549019607843137, 0.6363062894825174, 0.9882352941176471, 0.9882352941176471, 0.7568627450980392, 0.6513031943676378, 0.9803921568627451, 0.9725490196078431, 0.6705882352941176, 0.6663000992527582, 0.9803921568627451, 0.9490196078431372, 0.596078431372549, 0.6812970041378786, 0.9725490196078431, 0.9176470588235294, 0.5137254901960784, 0.696293909022999, 0.9686274509803922, 0.8784313725490196, 0.4235294117647059, 0.7112908139081193, 0.9647058823529412, 0.8274509803921568, 0.3254901960784314, 0.7262877187932397, 0.9529411764705882, 0.7686274509803922, 0.24705882352941178, 0.7412846236783601, 0.9450980392156862, 0.7098039215686275, 0.1843137254901961, 0.7562815285634805, 0.9372549019607843, 0.6431372549019608, 0.12156862745098039, 0.7712784334486009, 0.9254901960784314, 0.5803921568627451, 0.0784313725490196, 0.7862753383337212, 0.9176470588235294, 0.5215686274509804, 0.047058823529411764, 0.8012722432188417, 0.9019607843137255, 0.4588235294117647, 0.027450980392156862, 0.816269148103962, 0.8627450980392157, 0.3803921568627451, 0.011764705882352941, 0.8312660529890824, 0.788235294117647, 0.2901960784313726, 0.0, 0.8462629578742028, 0.7411764705882353, 0.22745098039215686, 0.00392156862745098, 0.8612598627593231, 0.6627450980392157, 0.16862745098039217, 0.01568627450980392, 0.8762567676444435, 0.5882352941176471, 0.11372549019607843, 0.023529411764705882, 0.891253672529564, 0.49411764705882355, 0.054901960784313725, 0.03529411764705882, 0.9062505774146844, 0.396078431372549, 0.0392156862745098, 0.058823529411764705, 0.9212474822998047, 0.301961, 0.047059, 0.090196]
    rate_learningLUT.ColorSpace = 'Lab'
    rate_learningLUT.NanColor = [0.25, 0.0, 0.0]
    rate_learningLUT.NanOpacity = 0.0
    rate_learningLUT.NumberOfTableValues = 10
    rate_learningLUT.ScalarRangeInitialized = 1.0
    
    # get color legend/bar for rate_learningLUT in view renderView1
    rate_learningLUTColorBar = GetScalarBar(rate_learningLUT, renderView1)
    rate_learningLUTColorBar.WindowLocation = 'Any Location'
    rate_learningLUTColorBar.Title = 'rate_learning'
    rate_learningLUTColorBar.ComponentTitle = ''
    rate_learningLUTColorBar.HorizontalTitle = 1
    rate_learningLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    rate_learningLUTColorBar.TitleFontSize = 24
    rate_learningLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    rate_learningLUTColorBar.LabelFormat = '%.4f'
    rate_learningLUTColorBar.RangeLabelFormat = '%.3f'
    
    # set color bar visibility
    rate_learningLUTColorBar.Visibility = 0
    
    # get 2D transfer function for 'params'
    paramsTF2D = GetTransferFunction2D('params')
    paramsTF2D.ScalarRangeInitialized = 1
    paramsTF2D.Range = [2.3055461497278884e-05, 0.003394088940694928, 0.0, 1.0]
    
    # get color transfer function/color map for 'params'
    paramsLUT = GetColorTransferFunction('params')
    paramsLUT.AutomaticRescaleRangeMode = 'Never'
    paramsLUT.TransferFunction2D = paramsTF2D
    paramsLUT.RGBPoints = [4.9999999999999996e-06, 0.02, 0.3813, 0.9981, 5.839729084962496e-06, 0.02000006, 0.424267768, 0.96906969, 6.820487157151409e-06, 0.02, 0.467233763, 0.940033043, 7.965959445046067e-06, 0.02, 0.5102, 0.911, 9.303809012173444e-06, 0.02000006, 0.546401494, 0.872669438, 1.086634481786509e-05, 0.02, 0.582600362, 0.83433295, 1.2691301976023705e-05, 0.02, 0.6188, 0.796, 1.4822753055085525e-05, 0.02000006, 0.652535156, 0.749802434, 1.7312172426999927e-05, 0.02, 0.686267004, 0.703599538, 2.021967936916749e-05, 0.02, 0.72, 0.6574, 2.361548994014875e-05, 0.02000006, 0.757035456, 0.603735359, 2.758161269182518e-05, 0.02, 0.794067037, 0.55006613, 3.221382916932451e-05, 0.02, 0.8311, 0.4964, 3.7624007027623516e-05, 0.021354336738172372, 0.8645368555261631, 0.4285579460761159, 4.394280162640946e-05, 0.023312914349117714, 0.897999359924484, 0.36073871343115577, 5.132281134649612e-05, 0.015976108242848862, 0.9310479513349017, 0.2925631815088092, 5.994226282843569e-05, 0.27421074700988196, 0.952562960995083, 0.15356836602739213, 7.000931513153629e-05, 0.4933546281681699, 0.9619038625309482, 0.11119493614749336, 8.176708675838766e-05, 0.6439, 0.9773, 0.0469, 9.549952694712183e-05, 0.762401813, 0.984669591, 0.034600153, 0.00011153827302265329, 0.880901185, 0.992033407, 0.022299877, 0.0001302706594113756, 0.9995285432627147, 0.9995193706781492, 0.0134884641450013, 0.00015214907173637132, 0.999402998, 0.955036376, 0.079066628, 0.00017770187189378658, 0.9994, 0.910666223, 0.148134024, 0.0002075461579500854, 0.9994, 0.8663, 0.2172, 0.0002424026670106663, 0.999269665, 0.818035981, 0.217200652, 0.00028311318096293424, 0.999133332, 0.769766184, 0.2172, 0.0003306608554410999, 0.999, 0.7215, 0.2172, 0.00038619396295559445, 0.99913633, 0.673435546, 0.217200652, 0.0004510536235817441, 0.999266668, 0.625366186, 0.2172, 0.0005268061929016095, 0.9994, 0.5773, 0.2172, 0.0006152810893651771, 0.999402998, 0.521068455, 0.217200652, 0.0007186149745986482, 0.9994, 0.464832771, 0.2172, 0.0008393033536106648, 0.9994, 0.4086, 0.2172, 0.0009802608410373523, 0.9947599917687346, 0.33177297300202935, 0.2112309638520206, 0.001144891548851126, 0.9867129505479589, 0.2595183410914934, 0.19012239549291934, 0.0013371712953907375, 0.9912458875646419, 0.14799417507952672, 0.21078892136920357, 0.0015617436210540582, 0.949903037, 0.116867171, 0.252900603, 0.0018240319294248117, 0.903199533, 0.078432949, 0.291800389, 0.0021303704620324665, 0.8565, 0.04, 0.3307, 0.002488157269775202, 0.798902627, 0.04333345, 0.358434298, 0.0029060328752534247, 0.741299424, 0.0466667, 0.386166944, 0.0033940889406949295, 0.6837, 0.05, 0.4139]
    paramsLUT.UseLogScale = 1
    paramsLUT.ColorSpace = 'RGB'
    paramsLUT.NanColor = [1.0, 0.0, 0.0]
    paramsLUT.NanOpacity = 0.0
    paramsLUT.NumberOfTableValues = 668
    paramsLUT.ScalarRangeInitialized = 1.0
    paramsLUT.VectorMode = 'Component'
    
    # get color legend/bar for paramsLUT in view renderView1
    paramsLUTColorBar = GetScalarBar(paramsLUT, renderView1)
    paramsLUTColorBar.WindowLocation = 'Any Location'
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
    
    # get 2D transfer function for 'rates_bin'
    rates_binTF2D = GetTransferFunction2D('rates_bin')
    rates_binTF2D.ScalarRangeInitialized = 1
    rates_binTF2D.Range = [1.5241377582242421e-07, 2.1670205114787677e-06, 0.0, 1.0]
    
    # get color transfer function/color map for 'rates_bin'
    rates_binLUT = GetColorTransferFunction('rates_bin')
    rates_binLUT.AutomaticRescaleRangeMode = 'Never'
    rates_binLUT.TransferFunction2D = rates_binTF2D
    rates_binLUT.RGBPoints = [1.5241377582242416e-07, 0.02, 0.3813, 0.9981, 1.6235765315698968e-07, 0.02000006, 0.424267768, 0.96906969, 1.7295029531554424e-07, 0.02, 0.467233763, 0.940033043, 1.842340294289119e-07, 0.02, 0.5102, 0.911, 1.9625394416173947e-07, 0.02000006, 0.546401494, 0.872669438, 2.0905806988225645e-07, 0.02, 0.582600362, 0.83433295, 2.226975705867879e-07, 0.02, 0.6188, 0.796, 2.3722694834592718e-07, 0.02000006, 0.652535156, 0.749802434, 2.5270426108932124e-07, 0.02, 0.686267004, 0.703599538, 2.69191354599307e-07, 0.02, 0.72, 0.6574, 2.867541096404246e-07, 0.02000006, 0.757035456, 0.603735359, 3.0546270521231786e-07, 0.02, 0.794067037, 0.55006613, 3.2539189897794425e-07, 0.02, 0.8311, 0.4964, 3.466213259876652e-07, 0.021354336738172372, 0.8645368555261631, 0.4285579460761159, 3.692358168928809e-07, 0.023312914349117714, 0.897999359924484, 0.36073871343115577, 3.933257369207706e-07, 0.015976108242848862, 0.9310479513349017, 0.2925631815088092, 4.1898734696463397e-07, 0.27421074700988196, 0.952562960995083, 0.15356836602739213, 4.4632318823271956e-07, 0.4933546281681699, 0.9619038625309482, 0.11119493614749336, 4.754424919925664e-07, 0.6439, 0.9773, 0.0469, 5.064616160481415e-07, 0.762401813, 0.984669591, 0.034600153, 5.395045096938981e-07, 0.880901185, 0.992033407, 0.022299877, 5.747032090036736e-07, 0.9995285432627147, 0.9995193706781492, 0.0134884641450013, 6.12198364433534e-07, 0.999402998, 0.955036376, 0.079066628, 6.521398028468278e-07, 0.9994, 0.910666223, 0.148134024, 6.946871262072305e-07, 0.9994, 0.8663, 0.2172, 7.400103493321195e-07, 0.999269665, 0.818035981, 0.217200652, 7.882905792546511e-07, 0.999133332, 0.769766184, 0.2172, 8.397207389092128e-07, 0.999, 0.7215, 0.2172, 8.945063380320411e-07, 0.99913633, 0.673435546, 0.217200652, 9.528662943574151e-07, 0.999266668, 0.625366186, 0.2172, 1.0150338083908704e-06, 0.9994, 0.5773, 0.2172, 1.0812572952549147e-06, 0.999402998, 0.521068455, 0.217200652, 1.1518013773308415e-06, 0.9994, 0.464832771, 0.2172, 1.2269479416630953e-06, 0.9994, 0.4086, 0.2172, 1.3069972663515062e-06, 0.9947599917687346, 0.33177297300202935, 0.2112309638520206, 1.392269220432311e-06, 0.9867129505479589, 0.2595183410914934, 0.19012239549291934, 1.483104542042611e-06, 0.9912458875646419, 0.14799417507952672, 0.21078892136920357, 1.5798661999756265e-06, 0.949903037, 0.116867171, 0.252900603, 1.6829408440674303e-06, 0.903199533, 0.078432949, 0.291800389, 1.792740350210723e-06, 0.8565, 0.04, 0.3307, 1.909703466169423e-06, 0.798902627, 0.04333345, 0.358434298, 2.0342975647705156e-06, 0.741299424, 0.0466667, 0.386166944, 2.16702051147877e-06, 0.6837, 0.05, 0.4139]
    rates_binLUT.UseLogScale = 1
    rates_binLUT.ColorSpace = 'RGB'
    rates_binLUT.NanColor = [1.0, 0.0, 0.0]
    rates_binLUT.NanOpacity = 0.0
    rates_binLUT.NumberOfTableValues = 521
    rates_binLUT.ScalarRangeInitialized = 1.0
    rates_binLUT.VectorComponent = 18
    rates_binLUT.VectorMode = 'Component'
    
    # get color legend/bar for rates_binLUT in view renderView1
    rates_binLUTColorBar = GetScalarBar(rates_binLUT, renderView1)
    rates_binLUTColorBar.WindowLocation = 'Any Location'
    rates_binLUTColorBar.Title = 'rates_bin'
    rates_binLUTColorBar.ComponentTitle = '18'
    rates_binLUTColorBar.HorizontalTitle = 1
    rates_binLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    rates_binLUTColorBar.TitleFontSize = 24
    rates_binLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    rates_binLUTColorBar.LabelFormat = '%.4f'
    rates_binLUTColorBar.RangeLabelFormat = '%.3f'
    
    # set color bar visibility
    rates_binLUTColorBar.Visibility = 0
    
    # get 2D transfer function for 'poly'
    polyTF2D = GetTransferFunction2D('poly')
    polyTF2D.ScalarRangeInitialized = 1
    polyTF2D.Range = [0.0, 12.0, 0.0, 1.0]
    
    # get color transfer function/color map for 'poly'
    polyLUT = GetColorTransferFunction('poly')
    polyLUT.AutomaticRescaleRangeMode = 'Never'
    polyLUT.EnableOpacityMapping = 1
    polyLUT.TransferFunction2D = polyTF2D
    polyLUT.RGBPoints = [0.0, 0.8901960784313725, 0.9568627450980393, 0.984313725490196, 0.1990557245435709, 0.807843137254902, 0.9019607843137255, 0.9607843137254902, 0.39811144908714313, 0.7607843137254902, 0.8588235294117647, 0.9294117647058824, 0.7962228981742863, 0.6862745098039216, 0.7843137254901961, 0.8705882352941177, 1.194334347261428, 0.6, 0.6980392156862745, 0.8, 1.5924457963485725, 0.5176470588235295, 0.615686274509804, 0.7254901960784313, 1.9905572454357143, 0.44313725490196076, 0.5333333333333333, 0.6431372549019608, 2.3886686945228575, 0.37254901960784315, 0.4588235294117647, 0.5686274509803921, 2.7867801436100006, 0.3058823529411765, 0.38823529411764707, 0.49411764705882355, 3.1848915926971437, 0.24705882352941178, 0.3176470588235294, 0.41568627450980394, 3.583003041784287, 0.2, 0.2549019607843137, 0.34509803921568627, 3.9811144908714287, 0.14902, 0.196078, 0.278431, 4.185483812493324, 0.2, 0.1450980392156863, 0.13725490196078433, 4.389733134115219, 0.23137254901960785, 0.17254901960784313, 0.16470588235294117, 4.593982455737114, 0.2549019607843137, 0.2, 0.1843137254901961, 4.798231777359009, 0.28627450980392155, 0.23137254901960785, 0.21176470588235294, 5.002481098980903, 0.3176470588235294, 0.2627450980392157, 0.23921568627450981, 5.206730420602798, 0.34509803921568627, 0.29411764705882354, 0.26666666666666666, 5.4109797422246935, 0.37254901960784315, 0.3215686274509804, 0.29411764705882354, 5.615229063846588, 0.403921568627451, 0.3568627450980392, 0.3176470588235294, 5.819478385468482, 0.43137254901960786, 0.38823529411764707, 0.34509803921568627, 6.023727707090377, 0.4627450980392157, 0.4196078431372549, 0.37254901960784315, 6.227977028712273, 0.4980392156862745, 0.4588235294117647, 0.40784313725490196, 6.432226350334167, 0.5372549019607843, 0.5058823529411764, 0.44313725490196076, 6.636475671956063, 0.5803921568627451, 0.5568627450980392, 0.48627450980392156, 6.840724993577957, 0.6313725490196078, 0.6078431372549019, 0.5333333333333333, 7.044974315199853, 0.6784313725490196, 0.6627450980392157, 0.5803921568627451, 7.249223636821746, 0.7254901960784313, 0.7137254901960784, 0.6274509803921569, 7.453472958443642, 0.7764705882352941, 0.7647058823529411, 0.6784313725490196, 7.657722280065536, 0.8274509803921568, 0.8235294117647058, 0.7372549019607844, 7.861971601687432, 0.8901960784313725, 0.8901960784313725, 0.807843137254902, 8.066220923309325, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 8.066220923309325, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 8.066340923309326, 0.9882352941176471, 0.9882352941176471, 0.8588235294117647, 8.074208241462708, 0.984313725490196, 0.9882352941176471, 0.8549019607843137, 8.263023877143858, 0.9882352941176471, 0.9882352941176471, 0.7568627450980392, 8.459706830978394, 0.9803921568627451, 0.9725490196078431, 0.6705882352941176, 8.656389784812927, 0.9803921568627451, 0.9490196078431372, 0.596078431372549, 8.853072738647462, 0.9725490196078431, 0.9176470588235294, 0.5137254901960784, 9.049755692481995, 0.9686274509803922, 0.8784313725490196, 0.4235294117647059, 9.246438646316527, 0.9647058823529412, 0.8274509803921568, 0.3254901960784314, 9.443121600151063, 0.9529411764705882, 0.7686274509803922, 0.24705882352941178, 9.639804553985595, 0.9450980392156862, 0.7098039215686275, 0.1843137254901961, 9.83648750782013, 0.9372549019607843, 0.6431372549019608, 0.12156862745098039, 10.033170461654663, 0.9254901960784314, 0.5803921568627451, 0.0784313725490196, 10.229853415489195, 0.9176470588235294, 0.5215686274509804, 0.047058823529411764, 10.426536369323731, 0.9019607843137255, 0.4588235294117647, 0.027450980392156862, 10.623219323158263, 0.8627450980392157, 0.3803921568627451, 0.011764705882352941, 10.8199022769928, 0.788235294117647, 0.2901960784313726, 0.0, 11.016585230827332, 0.7411764705882353, 0.22745098039215686, 0.00392156862745098, 11.213268184661864, 0.6627450980392157, 0.16862745098039217, 0.01568627450980392, 11.409951138496398, 0.5882352941176471, 0.11372549019607843, 0.023529411764705882, 11.606634092330934, 0.49411764705882355, 0.054901960784313725, 0.03529411764705882, 11.803317046165468, 0.396078431372549, 0.0392156862745098, 0.058823529411764705, 12.0, 0.301961, 0.047059, 0.090196]
    polyLUT.ColorSpace = 'Lab'
    polyLUT.NanColor = [0.25, 0.0, 0.0]
    polyLUT.NanOpacity = 0.0
    polyLUT.NumberOfTableValues = 28
    polyLUT.ScalarRangeInitialized = 1.0
    
    # get color legend/bar for polyLUT in view renderView1
    polyLUTColorBar = GetScalarBar(polyLUT, renderView1)
    polyLUTColorBar.WindowLocation = 'Any Location'
    polyLUTColorBar.Title = 'poly'
    polyLUTColorBar.ComponentTitle = ''
    polyLUTColorBar.HorizontalTitle = 1
    polyLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    polyLUTColorBar.TitleFontSize = 24
    polyLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    polyLUTColorBar.LabelFormat = '%.4f'
    polyLUTColorBar.RangeLabelFormat = '%.3f'
    
    # set color bar visibility
    polyLUTColorBar.Visibility = 0
    
    # get color legend/bar for pGA_01LUT in view renderView1
    pGA_01LUTColorBar = GetScalarBar(pGA_01LUT, renderView1)
    pGA_01LUTColorBar.WindowLocation = 'Any Location'
    pGA_01LUTColorBar.Position = [0.7439194248756514, 0.05562340966921131]
    pGA_01LUTColorBar.Title = 'PGA$_{10\\%-50yr}$'
    pGA_01LUTColorBar.ComponentTitle = ''
    pGA_01LUTColorBar.HorizontalTitle = 1
    pGA_01LUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    pGA_01LUTColorBar.TitleFontSize = 24
    pGA_01LUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    pGA_01LUTColorBar.LabelFontSize = 20
    pGA_01LUTColorBar.ScalarBarLength = 0.32999999999999996
    pGA_01LUTColorBar.DrawScalarBarOutline = 1
    pGA_01LUTColorBar.ScalarBarOutlineColor = [0.0, 0.0, 0.0]
    pGA_01LUTColorBar.AutomaticLabelFormat = 0
    pGA_01LUTColorBar.LabelFormat = '%.1f'
    pGA_01LUTColorBar.RangeLabelFormat = '%.1f'
    
    # set color bar visibility
    pGA_01LUTColorBar.Visibility = 1
    
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
    
    # show color legend
    m_statsvtiDisplay.SetScalarBarVisibility(renderView1, False)
    
    # hide data in view
    Hide(m_statsvti, renderView1)
    
    # show color legend
    pua_3_statsvtiDisplay.SetScalarBarVisibility(renderView1, False)
    
    # show color legend
    fe_statsvtiDisplay.SetScalarBarVisibility(renderView1, False)
    
    # hide data in view
    Hide(fe_statsvti, renderView1)
    
    # ----------------------------------------------------------------
    # setup color maps and opacity mapes used in the visualization
    # note: the Get..() functions create a new object, if needed
    # ----------------------------------------------------------------
    
    # get opacity transfer function/opacity map for 'params'
    paramsPWF = GetOpacityTransferFunction('params')
    paramsPWF.Points = [5e-06, 0.0, 0.5, 0.0, 5.6517097627776626e-05, 0.5955882668495178, 0.5, 0.0, 0.0003177823737198556, 0.625, 0.5, 0.0, 0.00034722072784678233, 0.8897058963775635, 0.5, 0.0, 0.0007703969042919331, 0.9705882668495178, 0.5, 0.0, 0.003394088940694928, 1.0, 0.5, 0.0]
    paramsPWF.ScalarRangeInitialized = 1
    
    # get opacity transfer function/opacity map for 'poly'
    polyPWF = GetOpacityTransferFunction('poly')
    polyPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.18241043016314507, 0.5955882668495178, 0.5, 0.0, 1.1074918806552887, 0.625, 0.5, 0.0, 1.2117264568805695, 0.8897058963775635, 0.5, 0.0, 2.710097908973694, 0.9705882668495178, 0.5, 0.0, 12.0, 1.0, 0.5, 0.0]
    polyPWF.ScalarRangeInitialized = 1
    
    # get opacity transfer function/opacity map for 'rate_learning'
    rate_learningPWF = GetOpacityTransferFunction('rate_learning')
    rate_learningPWF.Points = [0.006257875356823206, 0.0, 0.5, 0.0, 0.020166512673262894, 0.5955882668495178, 0.5, 0.0, 0.09070317207126703, 0.625, 0.5, 0.0, 0.09865096823212019, 0.8897058963775635, 0.5, 0.0, 0.21290049373255954, 0.9705882668495178, 0.5, 0.0, 0.9212474822998047, 1.0, 0.5, 0.0]
    rate_learningPWF.ScalarRangeInitialized = 1
    
    # get opacity transfer function/opacity map for 'bin'
    binPWF = GetOpacityTransferFunction('bin')
    binPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.045602607540786266, 0.5955882668495178, 0.5, 0.0, 0.2768729701638222, 0.625, 0.5, 0.0, 0.30293161422014236, 0.8897058963775635, 0.5, 0.0, 0.6775244772434235, 0.9705882668495178, 0.5, 0.0, 3.0, 1.0, 0.5, 0.0]
    binPWF.ScalarRangeInitialized = 1
    
    # get opacity transfer function/opacity map for 'rates_bin'
    rates_binPWF = GetOpacityTransferFunction('rates_bin')
    rates_binPWF.Points = [1.5241377582242421e-07, 0.0, 0.5, 0.0, 1.8303754926081116e-07, 0.5955882668495178, 0.5, 0.0, 3.383438260268289e-07, 0.625, 0.5, 0.0, 3.558431326394735e-07, 0.8897058963775635, 0.5, 0.0, 6.073955676313055e-07, 0.9705882668495178, 0.5, 0.0, 2.1670205114787677e-06, 1.0, 0.5, 0.0]
    rates_binPWF.ScalarRangeInitialized = 1
    
    # get opacity transfer function/opacity map for 'mask'
    maskPWF = GetOpacityTransferFunction('mask')
    maskPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    maskPWF.ScalarRangeInitialized = 1
    
    # get opacity transfer function/opacity map for 'vtkBlockColors'
    vtkBlockColorsPWF = GetOpacityTransferFunction('vtkBlockColors')
    vtkBlockColorsPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    
    # get opacity transfer function/opacity map for 'rate'
    ratePWF = GetOpacityTransferFunction('rate')
    ratePWF.Points = [2.3055461497278884e-05, 0.0, 0.5, 0.0, 7.433837329402571e-05, 0.5955882668495178, 0.5, 0.0, 0.00033441599269311837, 0.625, 0.5, 0.0, 0.00036372052628777887, 0.8897058963775635, 0.5, 0.0, 0.0007849730333276991, 0.9705882668495178, 0.5, 0.0, 0.0033967383205890656, 1.0, 0.5, 0.0]
    ratePWF.ScalarRangeInitialized = 1

    pGA_01LUT.RescaleTransferFunction(0.0, 0.6)
    pGA_01PWF.RescaleTransferFunction(0.0, 0.6)

    # ----------------------------------------------------------------
    # restore active source
    SetActiveSource(pua_3_statsvti)
    # ----------------------------------------------------------------



    # ----------------------------------------------------------------
    coastlinevtmDisplay.LineWidth = size_factor

    # generate extracts

    SaveExtracts(ExtractsOutputDirectory='extracts')
    SaveScreenshot('figure_urz2.png',
                   renderView1,
                   ImageResolution=[size_factor * i for i in renderView1.ViewSize])


def make_joint_figure():

    binmaps = [plt.imread(f"figure_m.png"),
               plt.imread(f"figure_fe.png"),
               plt.imread(f"figure_urz.png")]
    texts = ['a) Hybrid\nMultiplicative',
             'b) Poisson FE',
             'b) Poisson URZ']
    fig, axs = plt.subplots(1, 3,
                            figsize=(10, 6),
                            gridspec_kw={'wspace': -0.001},
                            constrained_layout=False)

    cities = {'Wellington': [1750, 2052],
              'Auckland': [1788, 758],
              'Tauranga': [2096, 989],
              'Gisborne': [2483, 1332],
              'Napier': [2255, 1585],
              'Christchurch': [1306, 2726],
              'Queenstown': [481, 3216],
              'Dunedin': [868, 3361],
              'Invercargill': [429, 3609]}

    offset = {'Wellington': [120, 0],
              'Auckland': [0, 0],
              'Tauranga': [30, 0],
              'Gisborne': [-50, 0],
              'Napier': [0, 0],
              'Christchurch': [0, 0],
              'Queenstown': [100, 0],
              'Dunedin': [0, 0],
              'Invercargill': [0, 0]}

    for i, j in enumerate(binmaps):

        for x, city in cities.items():
            axs[i].scatter(city[0], city[1], edgecolor='black',
                           color='white', s=20)
            axs[i].text(city[0] + offset[x][0], city[1] - 40, x, fontsize=8,
                        horizontalalignment='center',
                        verticalalignment='bottom',
                        **{'fontname': 'Ubuntu'})
        # axs[i].scatter(cities[:, 0], cities[:, 1], c='k', s=10)

        axs[i].imshow(j)
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)
        axs[i].text(100, 100, texts[i], fontsize=10, verticalalignment='top',
                    **{'fontname': 'Ubuntu'})

    plt.savefig("hazard_poisson.jpg", dpi=400, bbox_inches='tight')
    plt.savefig("fig5.jpg", dpi=1200, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    makefig_m(5)
    makefig_fe(5)
    makefig_urz(5)
    make_joint_figure()

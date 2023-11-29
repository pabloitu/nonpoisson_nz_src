# state file generated using paraview version 5.11.2
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

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
coastlinevtm = XMLMultiBlockDataReader(registrationName='coastline.vtm', FileName=['/home/pciturri/PycharmProjects/nonpoisson_nz/figures/hazard_ms/figure8/paraview/coastline.vtm'])
coastlinevtm.CellArrayStatus = ['t50_fid', 'elevation']
coastlinevtm.TimeArray = 'None'

# create a new 'XML Image Data Reader'
m_statsvti = XMLImageDataReader(registrationName='m_stats.vti', FileName=['/home/pciturri/PycharmProjects/nonpoisson_nz/figures/hazard_ms/figure8/paraview/m_stats.vti'])
m_statsvti.CellArrayStatus = ['PGA_0.1', 'mask']
m_statsvti.TimeArray = 'None'

# create a new 'XML Image Data Reader'
basemap_2193vti = XMLImageDataReader(registrationName='basemap_2193.vti', FileName=['/home/pciturri/PycharmProjects/nonpoisson_nz/figures/hazard_ms/figure8/paraview/basemap_2193.vti'])
basemap_2193vti.CellArrayStatus = ['basemap', 'mask']
basemap_2193vti.TimeArray = 'None'

# create a new 'XML Image Data Reader'
pua_3_statsvti = XMLImageDataReader(registrationName='pua_3_stats.vti', FileName=['/home/pciturri/PycharmProjects/nonpoisson_nz/figures/hazard_ms/figure8/paraview/pua_3_stats.vti'])
pua_3_statsvti.CellArrayStatus = ['PGA_0.1', 'mask']
pua_3_statsvti.TimeArray = 'None'

# create a new 'XML Image Data Reader'
npua_statsvti = XMLImageDataReader(registrationName='npua_stats.vti', FileName=['/home/pciturri/PycharmProjects/nonpoisson_nz/figures/hazard_ms/figure8/paraview/npua_stats.vti'])
npua_statsvti.CellArrayStatus = ['PGA_0.1', 'mask']
npua_statsvti.TimeArray = 'None'

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

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

# show data from npua_statsvti
npua_statsvtiDisplay = Show(npua_statsvti, renderView1, 'UniformGridRepresentation')

# get 2D transfer function for 'PGA_01'
pGA_01TF2D = GetTransferFunction2D('PGA_01')
pGA_01TF2D.ScalarRangeInitialized = 1
pGA_01TF2D.Range = [0.0, 0.6, 0.0, 1.0]

# get color transfer function/color map for 'PGA_01'
pGA_01LUT = GetColorTransferFunction('PGA_01')
pGA_01LUT.AutomaticRescaleRangeMode = 'Clamp and update every timestep'
pGA_01LUT.TransferFunction2D = pGA_01TF2D
pGA_01LUT.RGBPoints = [0.0, 0.278431372549, 0.278431372549, 0.858823529412, 0.08579999999999992, 0.0, 0.0, 0.360784313725, 0.17100000000000004, 0.0, 1.0, 1.0, 0.25739999999999996, 0.0, 0.501960784314, 0.0, 0.34259999999999996, 1.0, 1.0, 0.0, 0.42840000000000017, 1.0, 0.380392156863, 0.0, 0.5141999999999999, 0.419607843137, 0.0, 0.0, 0.6, 0.878431372549, 0.301960784314, 0.301960784314]
pGA_01LUT.ColorSpace = 'RGB'
pGA_01LUT.NanOpacity = 0.0
pGA_01LUT.NumberOfTableValues = 526
pGA_01LUT.ScalarRangeInitialized = 1.0
pGA_01LUT.VectorMode = 'Component'

# get opacity transfer function/opacity map for 'PGA_01'
pGA_01PWF = GetOpacityTransferFunction('PGA_01')
pGA_01PWF.Points = [0.0, 0.0, 0.5, 0.0, 0.027000000700354555, 0.2663043439388275, 0.5, 0.0, 0.05580000206828115, 0.45652174949645996, 0.5, 0.0, 0.09240000694990162, 0.635869562625885, 0.5, 0.0, 0.13140000402927393, 0.7771739363670349, 0.5, 0.0, 0.1974000036716462, 0.8586956858634949, 0.5, 0.0, 0.6, 1.0, 0.5, 0.0]
pGA_01PWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
npua_statsvtiDisplay.Representation = 'Slice'
npua_statsvtiDisplay.ColorArrayName = ['CELLS', 'PGA_0.1']
npua_statsvtiDisplay.LookupTable = pGA_01LUT
npua_statsvtiDisplay.SelectTCoordArray = 'None'
npua_statsvtiDisplay.SelectNormalArray = 'None'
npua_statsvtiDisplay.SelectTangentArray = 'None'
npua_statsvtiDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
npua_statsvtiDisplay.SelectOrientationVectors = 'None'
npua_statsvtiDisplay.ScaleFactor = 154400.0
npua_statsvtiDisplay.SelectScaleArray = 'None'
npua_statsvtiDisplay.GlyphType = 'Arrow'
npua_statsvtiDisplay.GlyphTableIndexArray = 'None'
npua_statsvtiDisplay.GaussianRadius = 7720.0
npua_statsvtiDisplay.SetScaleArray = [None, '']
npua_statsvtiDisplay.ScaleTransferFunction = 'PiecewiseFunction'
npua_statsvtiDisplay.OpacityArray = [None, '']
npua_statsvtiDisplay.OpacityTransferFunction = 'PiecewiseFunction'
npua_statsvtiDisplay.DataAxesGrid = 'GridAxesRepresentation'
npua_statsvtiDisplay.PolarAxes = 'PolarAxesRepresentation'
npua_statsvtiDisplay.ScalarOpacityUnitDistance = 25222.71540033692
npua_statsvtiDisplay.ScalarOpacityFunction = pGA_01PWF
npua_statsvtiDisplay.TransferFunction2D = pGA_01TF2D
npua_statsvtiDisplay.OpacityArrayName = ['CELLS', 'PGA_0.1']
npua_statsvtiDisplay.ColorArray2Name = ['CELLS', 'PGA_0.1']
npua_statsvtiDisplay.SliceFunction = 'Plane'
npua_statsvtiDisplay.SelectInputVectors = [None, '']
npua_statsvtiDisplay.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
npua_statsvtiDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
npua_statsvtiDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
npua_statsvtiDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.015200869180262089, 0.5955882668495178, 0.5, 0.0, 0.09229099005460739, 0.625, 0.5, 0.0, 0.10097720474004745, 0.8897058963775635, 0.5, 0.0, 0.2258414924144745, 0.9705882668495178, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

# init the 'Plane' selected for 'SliceFunction'
npua_statsvtiDisplay.SliceFunction.Origin = [1590196.015518637, 5469386.7749801995, 10.0]

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# restore active source
SetActiveSource(npua_statsvti)
# ----------------------------------------------------------------


if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='extracts')
# trace generated using paraview version 5.9.0-RC1-304-ge1d70e4363

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'Legacy VTK Reader'
sol_data = LegacyVTKReader(registrationName='test.vtk', FileNames=['/home/lkogler/src/nvssolver/test.vtk'])

# create a new 'Legacy VTK Reader'
src_data = LegacyVTKReader(registrationName='testsources.vtk', FileNames=['/home/lkogler/src/nvssolver/testsources.vtk'])

# get active view
pv_view = GetActiveViewOrCreate('RenderView')

UpdatePipeline(proxy=sol_data)

# create a new 'Stream Tracer With Custom Source'
sl_data = StreamTracerWithCustomSource(registrationName='StreamTracerWithCustomSource1', Input=sol_data,
    SeedSource=src_data)
sl_data.Vectors = ['POINTS', 'vel']

# show data in view
sl_rep = Show(sl_data, pv_view, 'GeometryRepresentation')


# trace defaults for the display properties.
sl_rep.Representation = 'Surface'
sl_rep.ColorArrayName = ['POINTS', 'vel']
# sl_rep.LookupTable = velLUT
sl_rep.SelectTCoordArray = 'None'
sl_rep.SelectNormalArray = 'None'
sl_rep.SelectTangentArray = 'None'
sl_rep.OSPRayScaleArray = 'vel'
sl_rep.OSPRayScaleFunction = 'PiecewiseFunction'
sl_rep.SelectOrientationVectors = 'Normals'
sl_rep.ScaleFactor = 0.09929866092279555
sl_rep.SelectScaleArray = 'vel'
sl_rep.GlyphType = 'Arrow'
sl_rep.GlyphTableIndexArray = 'vel'
sl_rep.GaussianRadius = 0.004964933046139776
sl_rep.SetScaleArray = ['POINTS', 'vel']
sl_rep.ScaleTransferFunction = 'PiecewiseFunction'
sl_rep.OpacityArray = ['POINTS', 'vel']
sl_rep.OpacityTransferFunction = 'PiecewiseFunction'
sl_rep.DataAxesGrid = 'GridAxesRepresentation'
sl_rep.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
sl_rep.ScaleTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 1.000244140625, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
sl_rep.OpacityTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 1.000244140625, 1.0, 0.5, 0.0]

sl_rep.AmbientColor = [1.0, 1.0, 0]
sl_rep.DiffuseColor = [1.0, 1.0, 0]

# reset view to fit data
pv_view.ResetCamera()

# update the view to ensure updated data information
pv_view.Update()

#change interaction mode for render view
pv_view.InteractionMode = '3D'

# reset view to fit data
pv_view.ResetCamera()

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(1212, 572)

# save screenshot
SaveScreenshot('/home/lkogler/src/nvssolver/yippie.png', pv_view, ImageResolution=[1212, 572], 
    # PNG options
    CompressionLevel='0')

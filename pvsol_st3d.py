import sys
from paraview.simple import *

ps_file = "cyl_surf.vtk"
vtk_file = "sol_st3d.vtk"
src_file = "sol_sources_st3d_generated.vtk"

def draw_plane(pcos, fl, ns, fpts = lambda ka, kb : True):
    va = [ ca - cb for ca, cb in zip(pcos[1], pcos[0]) ]
    la = (sum(x**2 for x in va))**0.5
    va = [x/la for x in va]
    vb = [ ca - cb for ca, cb in zip(pcos[2], pcos[0]) ]
    lb = (sum(x**2 for x in vb))**0.5
    vb = [x/lb for x in vb]
    ha, hb = la/ns[0], lb/ns[1]
    npts = 0
    for ka in range(ns[0]+1):
        for kb in range(ns[1]+1):
            if fpts(ka, kb):
                # for cz, ca, cb in zip (pcos[0], va, vb):
                    # print(cz, ca, cb, ka*ha*ca, kb*hb*cb)
                cos = [ cz + ka*ha*ca + kb*hb*cb for cz, ca, cb in zip (pcos[0], va, vb) ]
                # print(ka, kb, " -> ", cos)
                tempfile.write("{} {} {}\n".format(*cos))
                npts = npts + 1
    # quit()
    return npts

if True:
    tfn = "tempfile"
    tempfile = open(tfn, "w")
    npnts = 0
    H, W, L = 0.41, 0.41, 2.5
    # pos = (0.5, 0.2)
    # r = 0.05
    # cyl = csg.Cylinder(csg.Pnt(pos[0], pos[1], 0), csg.Pnt(pos[0], pos[1], 1), r).bc("obstacle")
    frt_plane = [(0.3, 0.175, 0.05), (0.3, 0.225, 0.05), (0.3, 0.175, W-0.05)] # front
    npnts = npnts + draw_plane(frt_plane, tempfile, [30, 5])#, fpts = lambda ka, kb : abs(ka+kb)%30==0)
    tempfile.close()
    sf = open("sol_sources_st3d_generated.vtk", "w")
    sf.write("# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS {} float\n".format(npnts))
    sf.write(open(tfn, "r").read())
    sf.close()
    
# get the active view: manages collection of representations, render-context
pv_view = GetActiveViewOrCreate('RenderView')

print("load plane surf")
sys.stdout.flush()
ps_data = LegacyVTKReader(registrationName='plane_surf', FileNames=[ps_file])
print("load sol")
sys.stdout.flush()
sol_data = LegacyVTKReader(registrationName='sol', FileNames=[vtk_file])
print("load sl source")
sys.stdout.flush()
source_data = LegacyVTKReader(registrationName='sources', FileNames=[src_file])

print("update plane surf")
sys.stdout.flush()
UpdatePipeline(proxy=ps_data)
print("update sol")
sys.stdout.flush()
UpdatePipeline(proxy=sol_data)
print("update sl source")
sys.stdout.flush()
UpdatePipeline(proxy=source_data)
sl_data = StreamTracerWithCustomSource(registrationName='StreamTracerWithCustomSource1', Input=sol_data,
                                                             SeedSource=source_data)
sl_data.Vectors = ['POINTS', 'vel']
sl_data.IntegrationStepUnit = 'Length'
sl_data.InitialStepLength = 0.001
sl_data.MaximumStepLength = 0.005
sl_data.MinimumStepLength = 0.0001
sl_data.MaximumSteps = 200000
sl_data.MaximumStreamlineLength = 181.0
sl_data.MaximumError = 0.0015
sl_data.TerminalSpeed = 1e-14
sl_data.IntegrationDirection = 'FORWARD'
UpdatePipeline(proxy=sl_data)


sl_rep = Show(sl_data, pv_view, 'GeometryRepresentation')
velLUT = GetColorTransferFunction('vel')
velLUT.ApplyPreset('Jet', True)
sl_rep.Representation = 'Surface'
sl_rep.ColorArrayName = ['POINTS', 'vel']
sl_rep.LookupTable = velLUT
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
sl_rep.LineWidth = 2.0

ps_rep = Show(ps_data, pv_view, 'GeometryRepresentation')
ps_rep.Representation = 'Surface'
ColorBy(ps_rep, None)
# green
# ps_rep.AmbientColor = [0.00784313725490196, 1.0, 0.1568627450980392]
# ps_rep.DiffuseColor = [0.00784313725490196, 1.0, 0.1568627450980392]
# muted blue-green
ps_rep.AmbientColor = [0.396078431372549, 0.5843137254901961, 0.596078431372549]
ps_rep.DiffuseColor = [0.396078431372549, 0.5843137254901961, 0.596078431372549]
ps_rep.AmbientColor = [0.42745098039215684, 0.5764705882352941, 0.6784313725490196]
ps_rep.DiffuseColor = [0.42745098039215684, 0.5764705882352941, 0.6784313725490196]

pv_view.InteractionMode = '3D'
pv_view.OrientationAxesVisibility = 0
sl_rep.SetScalarBarVisibility(pv_view, False)

# show color bar/color legend
# sl_rep.SetScalarBarVisibility(pv_view, False)

# change solid color
# ColorBy(sl_rep, None)
# sl_rep.AmbientColor = [1.0, 0.0, 0]
# sl_rep.DiffuseColor = [1.0, 0.0, 0]

HideScalarBarIfNotNeeded(velLUT, pv_view)

# update the view to ensure updated data information
pv_view.Update()

# reset view to fit data
pv_view.ResetCamera()

# layout/tab size in pixels
layout1 = GetLayout()
layout1.SetSize(1920, 1080)

# # current camera placement for pv_view
# pv_view.CameraPosition = [-7.240724508624635, -10.823021263908547, 3.4894515790447063]
# pv_view.CameraFocalPoint = [9.132000058029345, 15.759498490537496, -5.314899107698316]
# pv_view.CameraViewUp = [0.19894994612175126, 0.19552418278498634, 0.9603068326761065]
# pv_view.CameraParallelScale = 8.395529151795584

# # save screenshot
# SaveScreenshot('/home/lkogler/src/nvssolver/yippie.png', pv_view, ImageResolution=[1920, 1080],
#     TransparentBackground=1, 
#     # PNG options
#     CompressionLevel='0')

# # save screenshot
# SaveScreenshot('/home/lkogler/src/nvssolver/yippie_notrans.png', pv_view, ImageResolution=[1920, 1080],
#     TransparentBackground=0, 
#     # PNG options
#     CompressionLevel='0')

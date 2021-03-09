import sys
from paraview.simple import *

ps_file = "plane_surf.vtk"
vtk_file = "sol_vtk.vtk"
src_file = "sol_sources_generated.vtk"

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
    if False:
        my, mz = [ 0.0, -0.5 ]
        ly, lz = [ 5.0, 1.0 ]
        ny, nz = [ 50, 10 ]
        hy, hz = [ ly/ny, lz/nz ]
        for ky in range(-ny, ny):
            for kz in range(-nz, nz):
                x = -5
                y = my + ky*hy
                z = mz + kz*hz
                tempfile.write("{} {} {}\n".format(x,y,z))
                npnts = npnts+1
    elif True:
        frt_plane = [(-6.5, -0.4, -0.8), (-6.5, 0.4, -0.8), (-6.5, -0.4, 0.15)] # front
        # npnts = npnts + draw_plane(frt_plane, tempfile, [100, 8])#, fpts = lambda ka, kb : abs(ka-15)+abs(kb-15)<15)
        npnts = npnts + draw_plane(frt_plane, tempfile, [100, 100], fpts = lambda ka, kb : abs(ka+kb)%30==0)
        wl_plane  = [(-1.7, 0.6, -0.55), (-0.4, 4.5, -0.35), (-1.7, 0.6, -0.35)] # wing
        npnts = npnts + draw_plane(wl_plane, tempfile, [8, 30])
        wr_plane  = [(-1.7, -0.6, -0.55), (-0.4, -4.5, -0.35), (-1.7, -0.6, -0.35)] # wing
        npnts = npnts + draw_plane(wr_plane, tempfile, [8, 30])
        wfl_plane = [(-0.1, 4.35, -0.4), (0.1, 4.75, -0.4), (0.5, 4.65, 0.6)] # wing foil
        npnts = npnts + draw_plane(wfl_plane, tempfile, [30, 3], fpts = lambda ka, kb : kb > 0 )
        wfr_plane = [(-0.1, -4.35, -0.4), (0.1, -4.75, -0.4), (0.5, -4.65, 0.6)] #
        npnts = npnts + draw_plane(wfr_plane, tempfile, [30, 3], fpts = lambda ka, kb : kb > 0 )
        fc_plane  = [(0.8, -0.1, 0.5), (0.8, 0.1, 0.5), (2.6, -0.1, 1.8)] # foil center
        npnts = npnts + draw_plane(fc_plane, tempfile, [30, 5])
        fwl_plane = [(3.1, 0.05, 1.3), (4.3, 1.6, 1.3), (3.1, 0.05, 1.7)] # foil wing
        npnts = npnts + draw_plane(fwl_plane, tempfile, [5, 30])
        fwr_plane = [(3.1, -0.05, 1.3), (4.3, -1.6, 1.3), (3.1, -0.05, 1.7)] # foil w
        npnts = npnts + draw_plane(fwr_plane, tempfile, [5, 30])
    tempfile.close()
    sf = open("sol_sources_generated.vtk", "w")
    sf.write("# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS {} float\n".format(npnts))
    sf.write(open(tfn, "r").read())
    quit()
    
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
sl_data.MinimumStepLength = 0.0001
sl_data.MaximumSteps = 200001
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

# current camera placement for pv_view
pv_view.CameraPosition = [-7.240724508624635, -10.823021263908547, 3.4894515790447063]
pv_view.CameraFocalPoint = [9.132000058029345, 15.759498490537496, -5.314899107698316]
pv_view.CameraViewUp = [0.19894994612175126, 0.19552418278498634, 0.9603068326761065]
pv_view.CameraParallelScale = 8.395529151795584

# save screenshot
SaveScreenshot('/home/lkogler/src/nvssolver/yippie.png', pv_view, ImageResolution=[1920, 1080],
    TransparentBackground=1, 
    # PNG options
    CompressionLevel='0')

# save screenshot
SaveScreenshot('/home/lkogler/src/nvssolver/yippie_notrans.png', pv_view, ImageResolution=[1920, 1080],
    TransparentBackground=0, 
    # PNG options
    CompressionLevel='0')

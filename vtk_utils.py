import logging
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import time
from typing import List, Dict, Tuple, Union, Callable
import pydicom
import os

# 자동완성 기능이 안되서 다음과 같이 패키지 임포트 처리.
# from importlib.metadata import version as libversion
from packaging import version as packversion

import vtk

vtk_version = vtk.vtkVersion().GetVTKVersion()
if packversion.parse(vtk_version) > packversion.parse("9.0.0"):
    import vtkmodules.all as vtk
    from vtkmodules.util import numpy_support
    from vtkmodules.util.colors import *
    from vtkmodules.util.colors import antique_white
    from vtkmodules.util import colors
else:
    from vtk.util import numpy_support
    from vtk.util.colors import antique_white
    from vtk.util import colors
    from vtk.util.colors import *


# from commons import get_runtime_logger
def get_runtime_logger():
    logging.basicConfig(filename="example.log", level=logging.WARNING)
    return logging

# def get_runtime_logger():
# from teethnet import common
# from teethnet.teethConfigure import get_teeth_color_table

DARK_1ST = .45
DARK_LOWER = .30

g_next_items_fun = None  # type: Callable[[str],  Tuple[List, List]]

viewup = -1

rainbow_table = [
    [255, 0, 0],
    [255, 125, 0],
    [255, 255, 0],
    [125, 255, 0],
    [0, 255, 0],
    [0, 255, 125],
    [0, 255, 255],
    [0, 0, 255],
    [0, 5, 70],
    [100, 0, 255]
]


def get_teeth_color_table(normalize=True):
    "https://github.com/tensorflow/models/blob/master/research/deeplab/utils/get_dataset_colormap.py"
    color_table = np.asarray([
        [255, 255, 255],  # bg
        [6, 230, 230],  # 1
        [80, 50, 50],  # 2
        [4, 200, 3],  # 3
        [30, 20, 240],  # 4
        [240, 10, 7],  # 5
        [224, 5, 255],  # 6
        [235, 255, 7],  # 7
        [150, 5, 61],  # 8
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
    ])
    if normalize:
        color_table = color_table / 255.

    return color_table


def show_plots(points):
    """
    :param points:[N, 3]
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pts = points.T
    ax.plot(*pts, '*')
    plt.show()


def compute_boundary_edge(polydata):
    featureEdges = vtk.vtkFeatureEdges()
    featureEdges.SetInputData(polydata)
    featureEdges.BoundaryEdgesOn()
    featureEdges.FeatureEdgesOff()
    featureEdges.ManifoldEdgesOff()
    featureEdges.NonManifoldEdgesOff()
    featureEdges.Update()
    return featureEdges.GetOutput()


def compute_curvature(polydata):
    curvaturesFilter = vtk.vtkCurvatures()
    curvaturesFilter.SetInputData(polydata)
    curvaturesFilter.SetCurvatureTypeToMinimum()
    curvaturesFilter.SetCurvatureTypeToMaximum()
    curvaturesFilter.SetCurvatureTypeToGaussian()
    curvaturesFilter.SetCurvatureTypeToMean()
    curvaturesFilter.Update()

    return curvaturesFilter.GetOutput()


def get_axes(scales=None):
    v = 50 if scales is None else scales
    import vtk
    axes = vtk.vtkAxesActor()
    t = vtk.vtkTransform()
    t.Scale(v, v, v)
    axes.SetUserTransform(t)

    return axes


def get_transform_axes(afm, scales=50):
    axes = get_axes(scales)
    post = afm
    pre = axes.GetUserTransform()
    concat_t = myTransform()
    concat_t.Concatenate(post)
    concat_t.Concatenate(pre)
    axes.SetUserTransform(concat_t)
    return axes


def create_points_actor(pts, invert=False, point_size=None, color_value: Union[Tuple[float], List[float]] = None):
    assert pts.shape[1] == 3
    if invert:
        pts = pts[:, ::-1]
    # pt = x
    # pt = np.concatenate([x, y])
    points = vtk.vtkPoints()
    vtkarray = numpy_support.numpy_to_vtk(pts, array_type=vtk.VTK_FLOAT)
    points.SetData(vtkarray)

    numpy_polys = np.arange(pts.shape[0])
    numpy_polys = np.stack([np.ones([pts.shape[0]], dtype=numpy_polys.dtype), numpy_polys], axis=-1)

    cell_array = numpy_support.numpy_to_vtk(numpy_polys.reshape([-1, 2]), array_type=vtk.VTK_ID_TYPE)

    cells = vtk.vtkCellArray()
    # polys_reshape = polys.reshape([-1, 4])
    cells.SetCells(numpy_polys.shape[0], cell_array)
    # vtk_utils.reconstruct_polydata()

    point_polydata = vtk.vtkPolyData()
    point_polydata.SetPoints(points)
    point_polydata.SetVerts(cells)
    actor = polydata2actor(point_polydata)
    if point_size is not None:
        actor.GetProperty().SetPointSize(point_size)
    if color_value:
        actor.GetProperty().SetColor(*color_value)
        # for a in actors
        # change_actor_color()
    return actor


def create_curve_actor(x, closed=False, line_width=1.0, color: Tuple[float] = None):
    line_polydata = vtk.vtkPolyData()

    points = vtk.vtkPoints()

    vtkpoints = numpy_support.numpy_to_vtk(x, array_type=vtk.VTK_FLOAT)
    points.SetData(vtkpoints)
    N = x.shape[0]
    lines = vtk.vtkCellArray()
    for i in range(x.shape[0] - 1):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i)
        line.GetPointIds().SetId(1, i + 1)
        lines.InsertNextCell(line)

    if closed:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, x.shape[0] - 1)
        line.GetPointIds().SetId(1, 0)
        lines.InsertNextCell(line)

    line_polydata.SetPoints(points)
    line_polydata.SetLines(lines)

    line_actor = polydata2actor(line_polydata)
    line_actor.GetProperty().SetColor(color or tuple(np.random.uniform(0, 1, 3)))
    line_actor.GetProperty().SetLineWidth(line_width)
    return line_actor


def keypress_event(obj, event):
    key = obj.GetKeySym()
    print(key)
    viewup = 1
    if key == "d" or key == "c":
        save_dir = "test_image"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        imagefilter = vtk.vtkWindowToImageFilter()

        imagefilter.SetInput(obj.GetRenderWindow())
        # imagefilter.SetM
        imagefilter.ReadFrontBufferOff()
        if key == "d":
            imagefilter.SetInputBufferTypeToRGBA()
        else:
            imagefilter.SetInputBufferTypeToRGB()

        imagefilter.Update()

        lt = time.localtime()

        str_time = time.strftime("%Y%m%d%H%M%S")  # "{}{}{}".format(lt.tm_hour, lt.tm_min, lt.tm_sec)
        time.time()
        """
        %Y  Year with century as a decimal number.
        %m  Month as a decimal number [01,12].
        %d  Day of the month as a decimal number [01,31].
        %H  Hour (24-hour clock) as a decimal number [00,23].
        %M  Minute as a decimal number [00,59].
        %S  Second as a decimal number [00,61].
        """
        # time.strftime("%Y%m%d%h%M%s", time.time())
        # time.strftime("%Y%m%d%h%M%s", time.localtime())

        writer = vtk.vtkPNGWriter()
        writer.SetFileName("test_image/{}.png".format(str_time))
        writer.SetInputConnection(imagefilter.GetOutputPort())
        writer.Write()

        obj.GetRenderWindow().GetRenderers().GetFirstRenderer().ResetCamera()
        obj.Render()

        # elif key in ['n', 'm']:

        # item1, item2 = get_next_items()
        # obj.GetRenderWindow().GetRenderers()
        pass
        # for
        # ren.RemoveAllViewProps()
        # ren = obj.GetRenderWindow().GetRenderers().GetFirstRenderer()

        # cam = ren.GetActiveCamera()


    elif key in ["Left", "Up", "Right", "Down"]:

        ren = obj.GetRenderWindow().GetRenderers().GetFirstRenderer()
        props = [a for a in ren.GetViewProps()]
        prop = props[0]

        ctr = prop.GetCenter()
        bounds = prop.GetBounds()

        pos = [ctr[0], ctr[1], ctr[2]]
        if key == "Left":
            pos[0] = pos[0] - bounds[1] * 3
        elif key == "Right":
            pos[0] = pos[0] + bounds[1] * 3
        elif key == "Up":
            pos[1] = pos[1] + bounds[1] * 3
        elif key == "Down":
            pos[1] = pos[1] - bounds[1] * 3
        elif key.isdigit():
            key_int = int(key)
            dl = 100
            cam_list = (
                (0, dl, 0),
                (0, -dl, 0),
                (dl, 0, 0),
                (-dl, 0, 0),
                (0, 0, dl),
                (0, 0, -dl),
            )
            viewup = 1
            if 0 <= key_int < 6:
                pos = cam_list[key_int]
                viewup = 1
        else:
            pass

        cam = ren.GetActiveCamera()

        cam.SetViewUp(0, 0, viewup)
        cam.SetPosition(*pos)
    else:
        if g_next_items_fun and callable(g_next_items_fun):
            items = g_next_items_fun(key)
            all_renderer = [r for r in obj.GetRenderWindow().GetRenderers()]
            # assert len(items) == len(all_renderer)
            for i, ren in enumerate(all_renderer):
                # 기존 아이템 제거
                ren.RemoveAllViewProps()
                ctrs = add_actors(ren, items[i])
                ren.Render()

            ctrs = np.mean(ctrs, axis=0)
            # ren.GetActiveCamera()->SetViewUp(0, 1, 0);
            ren.GetActiveCamera().SetFocalPoint(*tuple(ctrs))

        # obj.GetRenderWindow.Render()
        # BUG:이벤트 강제로 발생 시켜줘야된다? 이유를 모르겠다.
    # obj.MouseWheelForwardEvent()
    # obj.MouseWheelBackwardEvent()
    # ren.Render()
    obj.GetRenderWindow().Render()


def _get_opacity_property(threshold=140, opacity_scalar=1.0):
    import vtk
    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(-3024, 0)
    # opacity.AddPoint(0, 0)
    opacity.AddPoint(0, 0.00)
    opacity.AddPoint(threshold * 0.8, 0.0)
    opacity.AddPoint(threshold, opacity_scalar)

    return opacity


def _get_normal_property(threshold=140, max_value=255, division=3, opacity=0.8, colors=None):
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.ShadeOn()
    volumeProperty.SetInterpolationType(vtk.VTK_LINEAR_INTERPOLATION)

    color = vtk.vtkColorTransferFunction()
    if colors is None:
        colors_list = [antique_white, cyan, green_pale, blue_light, cadmium_red_light]
        # colors_list = [antique_white, chartreuse, blue_light, carrot]

        # colors_list = [antique_white,]
    else:
        colors_list = [colors] * 5
    # color.AddRGBPoint(threshold, *antique_white)
    # color.AddRGBPoint(threshold, *cadmium_red_deep)
    # color.AddRGBPoint(threshold, *chartreuse)
    # color.AddRGBPoint(threshold, *blue_light)
    values = np.linspace(threshold, max_value, division)
    for v, c in zip(values, colors_list):
        color.AddRGBPoint(v, *c)
        # color.AddHSVPoint(v, c)
    # color.AddRGBPoint(threshold, *antique_white)
    color.AddRGBPoint(threshold, *colors_list[0])

    volumeProperty.SetColor(color)
    volumeProperty.SetScalarOpacity(_get_opacity_property(threshold, opacity))
    return volumeProperty


def to_vtk_cubes_from_box(aabb_boxes) -> List[vtk.vtkActor]:
    """

    :param aabb_boxes: (6)
    :return: [vtk.Actors]
    """
    return get_cube(convert_box_norm2vtk(aabb_boxes[np.newaxis])[0])


def shape_to_cube(shape: np.ndarray):
    assert len(shape) == 3

    # obb = shape_to_obb(shape)
    return to_vtk_cubes_from_box(np.concatenate([np.zeros(3), np.asarray(shape)]))


def convert_numpy_2_vtkmarching(volume_array, threshold, radius=1., dev=2., spacing=None):
    vtkimage = convert_numpy_vtkimag(volume_array)
    if spacing is not None:
        vtkimage.SetSpacing(*spacing)
    pd = convert_voxel_to_polydata(vtkimage, threshold, radius, dev)
    return polydata2actor(pd)


def convert_vtkimag_numpy(vtk_image_array: vtk.vtkImageData):
    numpy_array = numpy_support.vtk_to_numpy(vtk_image_array.GetPointData().GetScalars())
    return numpy_array.reshape(vtk_image_array.GetDimensions()[::-1])


def convert_numpy_vtkimag(volume_array):
    vtk_array = numpy_support.numpy_to_vtk(volume_array.astype(np.uint16).ravel(), array_type=vtk.VTK_UNSIGNED_SHORT)

    pd = vtk.vtkImageData()
    pd.GetPointData().SetScalars(vtk_array)
    pd.SetDimensions(volume_array.shape[::-1])
    return pd


def show_volume(imgdata):
    volumeMapper = vtk.vtkSmartVolumeMapper()
    volumeMapper.SetInputData(imgdata)
    volumeMapper.SetBlendModeToComposite()

    volume = vtk.vtkVolume()

    volume.SetMapper(volumeMapper)
    volume.SetProperty(_get_normal_property())
    volume.Update()

    ren = vtk.vtkRenderer()
    ren.AddVolume(volume)
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    # ren.GetActiveCamera()->SetViewUp(0, 1, 0);
    ren.GetActiveCamera().SetFocalPoint(*volume.GetCenter())
    # ren.GetActiveCamera()->SetPosition(c[0], c[1], c[2]);

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.Initialize()
    iren.Start()


def random_coloring_actor(actors):
    if isinstance(actors, vtk.vtkActor):
        actors.GetProperty().SetColor(*np.random.uniform(0, 1, [3]))
    elif isinstance(actors, (tuple, list)):
        for a in actors:
            random_coloring_actor(a)


def change_actor_color(actors, color):
    if color is not None:
        if len(color) == 3 and not isinstance(color[0], (tuple, list)):
            if isinstance(actors, list):
                for a in actors:
                    if isinstance(a, vtk.vtkActor):
                        a.GetProperty().SetColor(*color)
                    elif isinstance(a, vtk.vtkPolyData):
                        polydata_coloring(a, color)
                    # if isinstance(actors[0], vtk.vtkActor):
                # [ for a in actors]
            elif isinstance(actors, vtk.vtkActor):
                actors.GetProperty().SetColor(*color)
            elif isinstance(actors, vtk.vtkPolyData):
                polydata_coloring(actors, color)

            else:
                pass
                # raise ValueError
        elif len(color) == len(actors) and len(color) > 0 and \
                isinstance(color[0], (tuple, list, np.ndarray)):
            for a, c in zip(actors, color):
                if isinstance(a, vtk.vtkActor):
                    a.GetProperty().SetColor(*color)
                elif isinstance(a, vtk.vtkPolyData):
                    polydata_coloring(a, color)
            # [a.GetProperty().SetColor(*c) for a, c in zip(actors, color)]

        else:
            pass
    return actors
    # raise ValueError


def get_cube_from_shape(shape):
    shape = np.asarray(shape)
    volume_box = np.concatenate([np.zeros_like(shape), shape])
    return get_aabb_cubes(volume_box)


def get_aabb_cubes(bbox, color=None, invert: Union[bool, str] = True) -> List[vtk.vtkActor]:
    """
    :param bbox: [N, 6]
    :return:
    """
    if isinstance(invert, (bool, str)):
        if isinstance(invert, str):
            assert invert in ['xyz', 'zyx'], 'invert opation as string must to be "xyz" or "zyx"'
    else:
        raise ValueError

    bbox = np.asarray(bbox)
    _1dim = bbox.ndim == 1

    if _1dim:
        bbox = np.expand_dims(bbox, axis=0)

    bbox = convert_box_norm2vtk(bbox, invert=invert)
    res = [get_cube(b) for b in bbox]

    if color is not None:
        change_actor_color(res, color)
    # squeeze for keeping original-shape
    if _1dim:
        res = res[0]
    return res


def get_cube(bounds, color=None):
    """
    create cube from vtk format (6)
    vtk format, (x1,x2,y1,y2,z1,z2)
    :param bounds: vtk format, (x1,x2,y1,y2,z1,z2)
    :return:
    """
    # import vtk
    cube = vtk.vtkCubeSource()
    cube.SetBounds(*list(bounds))
    cube.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(cube.GetOutput())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetRepresentationToWireframe()
    actor.GetProperty().SetLineWidth(3)
    actor.GetProperty().LightingOff()
    color = color or tuple(np.random.uniform(0, 1, 3))
    actor.GetProperty().SetColor(color)

    return actor


#
def auto_refinement_mask(mask_volume, random_coloring=False, max_trunc=-1):
    """
    after cropping bounds non-zero value,
    converting voxel to mesh unsing marching cube
    :param mask_volume:
    :return:
    """

    # box

    uniquevalue = np.unique(mask_volume)

    fgvalue = uniquevalue[uniquevalue > 0]
    if max_trunc > 0:
        fgvalue = fgvalue[:max_trunc]
    # remove noise out of box
    # self.apply_box_mask(number)
    offset = 5
    max_inds = np.array(mask_volume.shape) - 1
    crop_actors = []
    for number in fgvalue:
        # center of index
        bool_mask = mask_volume == number
        # ix = np.stack(np.where(bool_mask), axis=-1)
        # center = np.round(np.mean(ix, axis=0)).astype(np.int32)

        # region growing in center of mass, remove other index
        # umask = bool_mask.astype(np.uint8)
        # region_mask = np.zeros_like(umask)

        inds = np.stack(np.where(bool_mask), axis=-1)
        p1 = inds.min(axis=0) - offset
        p2 = inds.max(axis=0) + offset
        p1 = np.clip(p1, 0, max_inds)
        p2 = np.clip(p2, 0, max_inds)
        z1, y1, x1 = p1
        z2, y2, x2 = p2
        cropmask = bool_mask[z1:z2, y1:y2, x1:x2]
        box = np.concatenate([p1, p2])

        # rendering update
        actor = convert_vtkvolume(cropmask, box, number, radius=1., dev=1.)
        if random_coloring:
            random_coloring_actor(actor)
        crop_actors.append(actor)
        # show_actors(crop_actors)
    return crop_actors


def compute_normal(polydata: vtk.vtkPolyData, splitting=False):
    # teeth_scans = register_mesh.pipeine_to_act(teeth_polydata)
    normal_gen = vtk.vtkPolyDataNormals()
    normal_gen.SetInputData(polydata)
    normal_gen.ComputeCellNormalsOn()
    normal_gen.ComputePointNormalsOff()

    if splitting is False:
        normal_gen.SplittingOff()

    normal_gen.Update()
    normal_polydata = normal_gen.GetOutput()
    polydata.GetCellData().SetNormals(normal_polydata.GetCellData().GetNormals())


def compute_normal(polydata: vtk.vtkPolyData, splitting=False, norm_copy=True, recompute=False):
    # = normdata.GetPointData().GetNormals()
    norm = polydata.GetPointData().GetNormals()
    if norm is None or recompute:
        normal_gen = vtk.vtkPolyDataNormals()
        normal_gen.SetInputData(polydata)
        normal_gen.ComputePointNormalsOn()
        if splitting is False:
            normal_gen.SplittingOff()

        normal_gen.ComputeCellNormalsOff()
        normal_gen.Update()
        normal_polydata = normal_gen.GetOutput()
        polydata.GetPointData().SetNormals(normal_polydata.GetPointData().GetNormals())
        return normal_polydata
    else:
        if norm.GetNumberOfTuples() == 0:
            normal_gen = vtk.vtkPolyDataNormals()
            normal_gen.SetInputData(polydata)
            normal_gen.ComputePointNormalsOn()
            if splitting is False:
                normal_gen.SplittingOff()

            normal_gen.ComputeCellNormalsOff()
            normal_gen.Update()
            normal_polydata = normal_gen.GetOutput()
            polydata.GetPointData().SetNormals(normal_polydata.GetPointData().GetNormals())
        else:
            pass

        return polydata


def create_sphere(pts, size=None, is_coloring_rainbow=True, color=None):
    """
    pts : [N, 3] array (x,y,z) in order
    return : vtk spheres actor
    """
    size = size or 1.
    spheres = []
    for pt in pts:
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetRadius(size)

        sphereMapper = vtk.vtkDataSetMapper()
        sphereMapper.SetInputConnection(sphereSource.GetOutputPort())

        forwardSphere = vtk.vtkActor()  # vtk.vtkActor()
        forwardSphere.PickableOff()
        forwardSphere.SetMapper(sphereMapper)
        forwardSphere.SetPosition(*pt)
        forwardSphere.GetProperty().SetColor(*np.random.uniform(0, 1, 3))
        spheres.append(forwardSphere)

    if is_coloring_rainbow:
        coloring_rainbow(spheres)
    elif color is not None:
        change_actor_color(spheres, color)

    return spheres


def apply_transform_actor(actor: vtk.vtkActor, t):
    afm = actor.GetUserTransform()
    if afm:
        pd = apply_transform_polydata(actor.GetMapper().GetInput(), afm)
    else:
        pd = actor.GetMapper().GetInput()
    tpd = apply_transform_polydata(pd, t)
    act = polydata2actor(tpd)
    return act


def apply_transform_polydata(polydata, transform):
    if isinstance(transform, np.ndarray):
        transform = myTransform(transform)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(polydata)
    transformFilter.Update()
    return transformFilter.GetOutput()




def _darkening_for_1st_inscisor(color, num):
    return (np.asarray(color) - np.array([DARK_1ST] * 3)).clip(min=0.) if num // 10 in [1, 4] and num % 10 == 1 else color


def _darkening_for_upper_lower(color, num):
    return (np.asarray(color) - np.array([DARK_LOWER] * 3)).clip(min=0.) if num // 10 in [3, 4]  else color


def change_teeth_color(actors, numbers, color_table=None, different_upper_lower=True):
    color_table = get_teeth_color_table() if color_table is None else color_table
    if isinstance(actors, list):
        for a, n in zip(actors, numbers):
            c = color_table[n % 10]
            c = _darkening_for_1st_inscisor(c, n)
            if different_upper_lower:
                c = _darkening_for_upper_lower(c, n)
            change_actor_color(a, c)
            # a.GetProperty().SetColor(*c)
    elif isinstance(actors, vtk.vtkActor) and isinstance(numbers, (int, np.integer)):
        c = color_table[numbers % 10]
        c = _darkening_for_1st_inscisor(c, numbers)
        if different_upper_lower:
            c = _darkening_for_upper_lower(c, numbers)
        change_actor_color(actors, c)

    return actors


def convert_vtkvolume(crop, bbox, tau, radius=1., dev=2.):
    color_table = get_teeth_color_table()

    # for crop, bbox, tau in zip(crop_volume, crop_bbox, crop_taus):
    z1, y1, x1 = bbox[:3]
    # crop = gaussian_filter(crop.astype(np.float), 0.8)
    tempactor = convert_numpy_2_vtkmarching(crop * 255, 123, radius, dev)
    pd = tempactor.GetMapper().GetInput()
    # vtk.vtkTransformPolyDataFilter
    t = vtk.vtkTransform()
    t.Translate(x1, y1, z1)
    pd = apply_transform_polydata(pd, t)
    actor = polydata2actor(pd)
    actor.GetMapper().ScalarVisibilityOff()
    change_teeth_color(actor, tau)
    # c = color_table[tau % 10]
    # # 우측 치아일 경우 좀 더 어둡게 처리
    # c = _darkening_for_1st_inscisor(c, tau)
    # # c = (np.asarray(c) - np.array([.45] * 3)).clip(min=0.) if tau // 10 in [1, 4] and tau % 10 == 1 else c
    # actor.GetProperty().SetColor(*c)
    # actor.GetMapper().ScalarVisibilityOff()

    return actor


def volume_marching_labeling(numpy_array, labels, color_table, offset=5, sigma=0.2, smoothing=False):
    actors = []
    expand = np.array([offset] * 6)
    expand = expand * np.array([-1, -1, -1, 1, 1, 1])
    max_inds = np.array(numpy_array.shape) - 1
    max_inds = np.pad(max_inds, [0, 3], mode='wrap')
    min_inds = np.zeros_like(max_inds)

    for lab in labels:
        zyx = np.where(numpy_array == lab)
        zyx = np.stack(zyx, axis=1)
        if zyx.size > 0:
            p1, p2 = zyx.min(axis=0), zyx.max(axis=0) + 1
            pbox = np.concatenate([p1, p2])
            pbox = pbox + expand
            pbox = np.clip(pbox, min_inds, max_inds)
            z1, y1, x1, z2, y2, x2 = pbox

            crop = (numpy_array[z1:z2, y1:y2, x1:x2] == lab).astype('int32')
            if smoothing:
                crop = gaussian_filter(crop.astype(np.float), sigma)
            actor = convert_numpy_2_vtkmarching(crop * 255, 123)
            t = vtk.vtkTransform()
            t.Translate(x1, y1, z1)
            actor.SetUserTransform(t)
            if lab > 10:
                c = color_table[lab % 10]
            else:
                c = color_table[9]
            actor.GetProperty().SetColor(*c)
            actor.GetMapper().ScalarVisibilityOff()
            actors.append(actor)

    return actors


def convert_auto_volume_labeling(numpy_volumue, smoothing=False):
    """
    :param numpy_volumue: numpy volume array, [d, h, w]
    :return: extract non-zero object using marching cube
    """
    color_table = get_teeth_color_table()
    unique_ids = np.unique(numpy_volumue)
    unique_ids = unique_ids[unique_ids > 0]
    return volume_marching_labeling(numpy_volumue, unique_ids, color_table, smoothing=smoothing)


def convert_volume_labeling(crop_volume, crop_bbox, crop_taus):
    actors = []
    color_table = get_teeth_color_table()

    for crop, bbox, tau in zip(crop_volume, crop_bbox, crop_taus):
        z1, y1, x1 = bbox[:3]
        crop = gaussian_filter(crop.astype(np.float), 0.8)
        actor = convert_numpy_2_vtkmarching(crop * 255, 123)
        t = vtk.vtkTransform()
        t.Translate(x1, y1, z1)
        actor.SetUserTransform(t)
        c = color_table[tau % 10]
        actor.GetProperty().SetColor(*c)
        actor.GetMapper().ScalarVisibilityOff()

        actors.append(actor)

    return actors


def volume_labeling(numpy_array, labels, color_table):
    """
    :param numpy_array: [D, H, W] , uint type
    :param labels: [N], labels, uint type
    :return:
    """
    # numpy_array = gaussian_filter(numpy_array, 0.9)
    factor = 100
    labels = np.asarray(labels)
    color_table = np.asarray(color_table)
    # assert labels.shape[0] == colors.shape[0]
    # numpy_array2 = gaussian_filter(numpy_array, 2.0, mode='mirror')
    vtk_array = numpy_support.numpy_to_vtk(numpy_array.astype(np.uint16).ravel() * factor,
                                           array_type=vtk.VTK_UNSIGNED_SHORT)
    # visulaize(numpy_array2*255, 100)

    pd = vtk.vtkImageData()
    pd.GetPointData().SetScalars(vtk_array)
    pd.SetDimensions(numpy_array.shape[::-1])

    # volumeMapper = vtk.vtkSmartVolumeMapper()
    # volumeMapper.SetInputData(pd)
    # volumeMapper.SetBlendModeToComposite()
    volumeMapper = vtk.vtkFixedPointVolumeRayCastMapper()
    # volumeMapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()
    volumeMapper.SetInputData(pd)

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)

    # property
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.ShadeOn()
    # volumeProperty.SetAutoAdjustSampleDistances(10)

    # volumeProperty.SetInterpolationTypeToLinear()

    # volumeProperty.SetInterpolationType(vtk.VTK_LINEAR_INTERPOLATION)

    thresh = np.unique(labels[labels > 0]).min()
    maxv = np.unique(labels[labels > 0]).max()
    color = vtk.vtkColorTransferFunction()

    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(0, 0.00)
    for v in np.unique(labels):
        value = v if v in color_table else v % 10
        # if not v in color_table:
        #     value = v % 10
        c = color_table[value]
        color.AddRGBPoint(v * factor, *tuple(c))

        # opacity.AddPoint(thresh*factor*0.5, 0.0)
        opacity.AddPoint(thresh * factor, 0.85)
        # opacity.AddPoint(maxv * factor, 1.0)
        volumeProperty.SetScalarOpacity(opacity)
    # color.AddRGBPoint(0, 0, 0, 0)

    volumeProperty.SetColor(color)

    volume.SetProperty(volumeProperty)
    volume.Update()

    return volume


def show_volume_list(volumes_list):
    vols = [numpyvolume2vtkvolume(np.squeeze(v) * 255, 123, division=i + 1) for i, v in
            enumerate(volumes_list)]
    return vols


def create_vector(norm, pts, scale=10, invert=False):
    line_polydata = vtk.vtkPolyData()
    if invert:
        pts = pts[:, ::-1]
        norm = norm[:, ::-1]
    start = pts
    end = pts + norm * scale

    points = vtk.vtkPoints()

    stack_points = np.concatenate([start, end], axis=0)

    vtkpoints = numpy_support.numpy_to_vtk(stack_points, array_type=vtk.VTK_FLOAT)
    points.SetData(vtkpoints)
    N = pts.shape[0]
    lines = vtk.vtkCellArray()
    for i in range(0, pts.shape[0]):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i)
        line.GetPointIds().SetId(1, N + i)
        lines.InsertNextCell(line)

    line_polydata.SetPoints(points)
    line_polydata.SetLines(lines)
    return polydata2actor(line_polydata)
    # return line_polydata


def read_stl(path):
    hanCount = len(re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', path))
    encode_path = path.encode('euc-kr') if hanCount > 0 else path

    reader = vtk.vtkSTLReader()
    reader.SetFileName(encode_path)
    reader.Update()

    if reader.GetErrorCode() or not reader.GetOutput().GetNumberOfPoints():
        reader = vtk.vtkSTLReader()
        reader.SetFileName(path)
        reader.Update()

        assert not reader.GetErrorCode(), "cannot load file:{}".format(path)

    return reader.GetOutput()


def read_vtp(path):
    hanCount = len(re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', path))
    encode_path = path.encode('euc-kr') if hanCount > 0 else path

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(encode_path)
    reader.Update()

    # import vtkmodules.all as vtk
    #
    # vtk.VTK_ERROR
    # python 3.8 & vtk major 9 버전에서는 encoding 할 필요없이 동작된다?
    if reader.GetErrorCode() or reader.GetOutput().GetNumberOfPoints() == 0:
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(path)
        reader.Update()

        assert not reader.GetErrorCode(), "cannot load file:{}".format(path)

    return reader.GetOutput()


def write_stl(filename, polydata_or_actor):
    polydata = polydata_or_actor
    if isinstance(polydata_or_actor, vtk.vtkActor):
        polydata = polydata_or_actor.GetMapper().GetInput()
        # t = myTransform() or
        t0 = polydata_or_actor.GetUserTransform()
        t = myTransform()
        if t0 is not None:
            t.DeepCopy(t0)
        polydata = apply_transform_polydata(polydata, t)

    writer = vtk.vtkSTLWriter()
    writer.SetInputData(polydata)
    writer.SetFileTypeToBinary()
    writer.SetFileName(filename)
    writer.Write()


def write_ply(filename, polydata_or_actor):
    polydata = polydata_or_actor
    if isinstance(polydata_or_actor, vtk.vtkActor):
        polydata = polydata_or_actor.GetMapper().GetInput()
        # t = myTransform() or
        t0 = polydata_or_actor.GetUserTransform()
        t = myTransform()
        if t0 is not None:
            t.DeepCopy(t0)
        polydata = apply_transform_polydata(polydata, t)

    writer = vtk.vtkPLYWriter()
    writer.SetInputData(polydata)
    writer.SetFileTypeToBinary()
    vtk_scalars = polydata.GetPointData().GetScalars()
    if vtk_scalars and vtk_scalars.GetName():
        writer.SetArrayName(vtk_scalars.GetName())
    else:
        writer.SetArrayName('Colors')
    writer.SetFileName(filename)
    writer.Write()


def read_ply(path):
    hanCount = len(re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', path))
    encode_path = path.encode('euc-kr') if hanCount > 0 else path

    reader = vtk.vtkPLYReader()
    reader.SetFileName(encode_path)
    reader.Update()

    if reader.GetErrorCode() or not reader.GetOutput().GetNumberOfPoints():
        reader = vtk.vtkPLYReader()
        reader.SetFileName(path)
        reader.Update()

        assert not reader.GetErrorCode(), "cannot load file:{}".format(path)

    return reader.GetOutput()


def read_dicom(path, **kwargs):
    """

    Parameters
    ----------
    path : str
        pathname
    kwargs :
        normalize : bool optioon voxel-array normalized
        return_windowing : bool option return windowing (min, max) for visualize
        progress_callback : function callback dicom callback as list
    Returns
    -------

    """

    hanCount = len(re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', path))
    # encode_path = path.encode('euc-kr') if hanCount > 0 else path
    encode_path = path.encode('euc-kr') if hanCount > 0 else path

    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(encode_path)

    progress_callback = kwargs.get('progress_callback', [])

    def callback(caller, evetnt):
        for func in progress_callback:
            func(caller.GetProgress())

    if len(progress_callback) > 0:
        reader.AddObserver(vtk.vtkCommand.ProgressEvent, callback)

    reader.Update()

    if reader.GetErrorCode() != vtk.vtkErrorCode().NoError:
        raise ValueError("vtk loading failed")

    # vtk.vtkErrorCode().NoError

    image = reader.GetOutput()
    vol = numpy_support.vtk_to_numpy(image.GetPointData().GetScalars())
    vol_array = vol.reshape([*image.GetDimensions()[::-1]])
    windowing_method = kwargs.get("method", "windowing")

    def get_range_by_windowing():
        # https://vtk.org/pipermail/vtkusers/2017-January/097644.html
        dcm_file = reader.GetProgressText()
        dcm_file = dcm_file.decode('euc-kr') if isinstance(dcm_file, bytes) else dcm_file
        # hanCount = len(re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', dcm_file.decode('euc-kr')))
        # dcm_file = dcm_file.decode('euc-kr') if hanCount > 0 else dcm_file
        ds = pydicom.read_file(dcm_file)
        # ds = pydicom.read_file("D:/dataset/autoplanning/marker_sample_gt/19/CT/김봉녀_07186_0005.dcm")
        ww, wc = ds.WindowWidth, ds.WindowCenter
        dmin = int(wc) - int(ww) / 2
        dmax = int(wc) + int(ww) / 2
        return dmin, dmax

    def get_range_by_fixed_hu():
        offset = reader.GetRescaleOffset()
        slope = reader.GetRescaleSlope()

        hu_ww = 4095
        hu_wc = 1024
        # hu_ww = ww * slope
        # hu_wc = wc * slope + offset
        ww = hu_ww / slope
        wc = (hu_wc - offset) / slope

        dmin = wc - ww / 2
        dmax = wc + ww / 2
        return dmin, dmax

    if windowing_method == "windowing":
        try:
            dmin, dmax = get_range_by_windowing()
        except Exception as e:
            logger = get_runtime_logger()
            logger.error("failed to load dicom-file. then processed fixed range")
            logger.error(e)
            dmin, dmax = get_range_by_fixed_hu()

    elif windowing_method == "fixed_hu":
        dmin, dmax = get_range_by_fixed_hu()

    else:
        raise ValueError(windowing_method)

    if kwargs.get('normalize', True):
        norm_voxel = np.clip((vol_array - dmin) / (dmax - dmin), 0., 1.)
    else:
        norm_voxel = vol_array

    if kwargs.get('return_windowing', False):
        return (norm_voxel, np.array(reader.GetPixelSpacing()), (dmin, dmax))
    else:
        return (norm_voxel, np.array(reader.GetPixelSpacing()))


def apply_transform_tensor(tensor, transform):
    """
    :param tensor:[N, 6]  3pose - 3orientation
    :param transform:
    :return:
    """
    pose, orient = tensor[:, :3], tensor[:, 3:]
    tpose = apply_trasnform_np(pose, transform)
    torient = apply_rotation(orient, transform)
    return np.concatenate([tpose, torient], axis=-1)


def apply_trasnform_np(pts, transform):
    return np.dot(pts, transform[:3, :3].T) + transform[:3, 3]


def apply_rotation(normals, transform):
    """
    :param normals: direction vector [N, 3]
    :param transform:
    :return:
    """
    return np.dot(normals, transform[:3, :3].T)


def numpy2vtkvolume(numpy_array: np.ndarray, threshold: float = None, division=1, opacity=0.8, color=None):
    """
    alias for numpyvolume2vtkvolume(...)
    :param numpy_array: np.ndarray (d, h, w) voxel as numpy array
    :param threshold: threshold for voxel scalar-transferfunction
    :param division:
    :param opacity:
    :param color:
    :return:
    """
    return numpyvolume2vtkvolume(numpy_array, threshold, division, opacity, color, auto)


def numpyvolume2vtkvolume(numpy_array: np.ndarray, threshold: float = None, division=1, opacity=0.8, color=None,
                          spacing=None):
    if threshold is None:
        max_value = np.max(numpy_array)
        min_value = np.min(numpy_array)
        threshold = max_value / 2.0
        scale = 1.0
        if (max_value - min_value) < (1.0 + 1e-3):
            scale = 255.0
        numpy_array = numpy_array * scale
        threshold = threshold * scale

    vtk_array = numpy_support.numpy_to_vtk(numpy_array.astype(np.uint16).ravel(), array_type=vtk.VTK_UNSIGNED_SHORT)

    pd = vtk.vtkImageData()
    pd.GetPointData().SetScalars(vtk_array)
    pd.SetDimensions(numpy_array.shape[::-1])

    if spacing is not None:
        pd.SetSpacing(*spacing)
    # if in_transform is not None:
    #     pd = apply_transform_polydata(pd, in_transform)

    volumeMapper = vtk.vtkSmartVolumeMapper()
    volumeMapper.SetInputData(pd)
    volumeMapper.SetBlendModeToComposite()

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(_get_normal_property(threshold, division=division, opacity=opacity, colors=color))
    volume.Update()
    return volume


def change_opacity_vtkvolume(vtkvolume: vtk.vtkVolume, threshold, opacity=1.0, division=1, color=None):
    opacity_prop = _get_normal_property(threshold, division=division, opacity=opacity, colors=color)
    vtkvolume.SetProperty(opacity_prop)


def visulaize(volume_array, threshold, viewup=-1):
    volume = numpyvolume2vtkvolume(volume_array, threshold)
    # vtk_array = numpy_support.numpy_to_vtk(volume_array.astype(np.uint16).ravel(), array_type=vtk.VTK_UNSIGNED_SHORT)
    #
    # pd = vtk.vtkImageData()
    # pd.GetPointData().SetScalars(vtk_array)
    # pd.SetDimensions(volume_array.shape[::-1])
    #
    # volumeMapper = vtk.vtkSmartVolumeMapper()
    # volumeMapper.SetInputData(pd)
    # volumeMapper.SetBlendModeToComposite()
    #
    # volume = vtk.vtkVolume()
    #
    # volume.SetMapper(volumeMapper)
    # volume.SetProperty(_get_normal_property(threshold))
    # volume.Update()

    ren = vtk.vtkRenderer()
    ren.AddVolume(volume)
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    # ren.GetActiveCamera()->SetViewUp(0, 1, 0);
    ren.GetActiveCamera().SetFocalPoint(*volume.GetCenter())
    # ren.GetActiveCamera()->SetPosition(c[0], c[1], c[2]);

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.AddObserver("KeyPressEvent", keypress_event)
    iren.Initialize()
    iren.Start()


def _normalize(p):
    p[:] = p / np.linalg.norm(p)


def create_arrow(p1, p2, revert=False):
    if revert:
        p1 = p1[::-1]
        p2 = p2[::-1]

    norm_x = p2 - p1
    length = np.linalg.norm(norm_x)
    _normalize(norm_x)

    norm_z = np.cross(norm_x, np.random.randn(3))

    _normalize(norm_z)

    norm_y = np.cross(norm_z, norm_x)

    mat = np.eye(4)
    mat[:3, :3] = np.stack([norm_x, norm_y, norm_z], axis=1)

    tmat = myTransform()
    tmat.set_from_numpy_mat(mat)
    transform = myTransform()
    transform.Translate(*tuple(p1))
    transform.Concatenate(tmat)
    transform.Scale(length, length, length)

    arrow_source = vtk.vtkArrowSource()
    arrow_source.Update()
    arrow = arrow_source.GetOutput()
    arrow_actor = polydata2actor(arrow)
    arrow_actor.SetUserTransform(transform)
    return arrow_actor


def create_arrow_polydata(p1, p2):
    norm_x = p2 - p1
    length = np.linalg.norm(norm_x)
    _normalize(norm_x)

    norm_z = np.cross(norm_x, np.random.randn(3))

    _normalize(norm_z)

    norm_y = np.cross(norm_z, norm_x)

    mat = np.eye(4)
    mat[:3, :3] = np.stack([norm_x, norm_y, norm_z], axis=1)

    tmat = myTransform()
    tmat.set_from_numpy_mat(mat)
    transform = myTransform()
    transform.Translate(*tuple(p1))
    transform.Concatenate(tmat)
    transform.Scale(length, length, length)

    arrow_source = vtk.vtkArrowSource()
    arrow_source.Update()
    arrow = arrow_source.GetOutput()

    return apply_transform_polydata(arrow, transform)


def convert_box_norm2vtk(boxes, invert=True):
    """
    normal format (z1, y1, x1, z2, y2, x2)
    to vtk-format (x1, x2, y1, y2, z1, z2)

    :param boxes:
    :return:
    """
    if isinstance(invert, (bool, str)):
        if isinstance(invert, str):
            assert invert in ['xyz', 'zyx'], 'invert opation as string must to be "xyz" or "zyx"'
            # coord_invert = True if invert == 'zyx' else False
            coord_invert = invert == 'zyx'
        else:
            coord_invert = invert
    else:
        raise ValueError

    boxes_vtk = np.empty_like(boxes)
    # # (x1, x2, y1, y2, z1, z2)
    # boxes_vtk[:, ::2] = boxes[:, :3][:, ::-1]
    # boxes_vtk[:, 1::2] = boxes[:, 3:][:, ::-1]
    if coord_invert:
        boxes_vtk[:, ::2] = boxes[:, :3][:, ::-1]
        boxes_vtk[:, 1::2] = boxes[:, 3:][:, ::-1]
    else:
        boxes_vtk[:, ::2] = boxes[:, :3]
        boxes_vtk[:, 1::2] = boxes[:, 3:]

    return boxes_vtk


def polydata2voxelization_withpad(polydata, spacing,
                                  input_origin=None, input_bounds=None, return_center=True, return_origin=False,
                                  padding=0.0, expandding=1.05, return_bounds=False):
    """
    :param polydata: vtkPolydata
    :param spacing: tuple or list of 3 (float)
    :return: voxel data and voxel_center
    """
    spacing = np.asarray(spacing)

    whiteImg = vtk.vtkImageData()
    actual_bounds = np.array(polydata.GetBounds()).reshape([-1, 2]).T

    # expand bounds, to protect voxel boundary truncation.
    if input_bounds is None:
        bmin, bmax = actual_bounds[0], actual_bounds[1]
        ctr = (bmax + bmin) / 2
        ext = (bmax - bmin) / 2
        fext = ext + padding
        emin, emax = ctr - fext, ctr + fext
        # (3, 2)
        bounds = np.stack([emin, emax], axis=-1)
    else:
        bounds = input_bounds

    # numpy_support.vtk_to_numpy(polydata.Get)

    dim = np.ceil((bounds[:, 1] - bounds[:, 0]) / np.array(spacing)).astype(np.int32)
    ones = np.ones([3], dtype=np.int32)
    dim = dim + ones
    # dim = np.ceil((bounds[:, 1] - bounds[:, 0]) / np.array(spacing)).astype(np.int)
    min_bound = bounds[:, 0]
    origin = bounds[:, 0] if input_origin is None else input_origin
    whiteImg.SetDimensions(dim)
    whiteImg.SetSpacing(*tuple(spacing))
    whiteImg.SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1)
    whiteImg.SetOrigin(origin)

    # count = whiteImg.GetNumberOfPoints()

    np_arry = np.full([np.prod(dim, dtype=np.uint32)], 255, dtype=np.uint8)
    vtk_array = numpy_support.numpy_to_vtk(np_arry, vtk.VTK_UNSIGNED_CHAR)
    whiteImg.GetPointData().SetScalars(vtk_array)

    # whiteImg.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    #
    # for i in range(count):
    #     whiteImg.GetPointData().GetScalars().SetTuple1(i, 255)

    # scalars = whiteImg.GetPointData().GetScalars()

    # voxel = numpy_support.vtk_to_numpy(scalars)

    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(polydata)
    pol2stenc.SetOutputOrigin(origin)
    pol2stenc.SetOutputSpacing(*tuple(spacing))
    pol2stenc.SetOutputWholeExtent(whiteImg.GetExtent())
    pol2stenc.Update()

    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(whiteImg)
    imgstenc.SetStencilData(pol2stenc.GetOutput())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(0)
    imgstenc.Update()

    img = imgstenc.GetOutput()
    temp_vox = numpy_support.vtk_to_numpy(img.GetPointData().GetScalars())
    vox = temp_vox.reshape(dim[::-1])

    ctr = (bounds[:, 0] + bounds[:, 1]) / 2
    ctr = ctr / spacing
    ctr = ctr[::-1]
    # ctr = (ctr[3:] + ctr[:3])/2

    if return_center or return_origin or return_bounds:
        outs = [vox]
        if return_center:
            outs.append(ctr)
        if return_origin:
            voxel_origin = (-origin) / spacing
            outs.append(voxel_origin)
        if return_bounds:
            outs.append(bounds)
        return outs

    return vox


def polydata2voxelization(polydata, spacing, return_center=True, return_origin=False, expand=1.0):
    """
    :param polydata: vtkPolydata
    :param spacing: tuple or list of 3 (float)
    :return: voxel data and voxel_center
    """
    spacing = np.asarray(spacing)

    whiteImg = vtk.vtkImageData()
    actual_bounds = np.array(polydata.GetBounds()).reshape([-1, 2]).T

    # expand bounds, to protect voxel boundary truncation.
    bmin, bmax = actual_bounds[0], actual_bounds[1]
    ctr = (bmax + bmin) / 2
    ext = (bmax - bmin) / 2
    fext = ext * expand
    emin, emax = ctr - fext, ctr + fext
    # (3, 2)
    bounds = np.stack([emin, emax], axis=-1)

    # numpy_support.vtk_to_numpy(polydata.Get)

    dim = np.ceil((bounds[:, 1] - bounds[:, 0]) / np.array(spacing)).astype(np.int32)
    ones = np.ones([3], dtype=np.int32)
    dim = dim + ones
    # dim = np.ceil((bounds[:, 1] - bounds[:, 0]) / np.array(spacing)).astype(np.int)
    min_bound = bounds[:, 0]
    origin = bounds[:, 0]
    whiteImg.SetDimensions(dim)
    whiteImg.SetSpacing(*tuple(spacing))
    whiteImg.SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1)
    whiteImg.SetOrigin(origin)

    # count = whiteImg.GetNumberOfPoints()

    np_arry = np.full([np.prod(dim)], 255, dtype=np.uint8)
    vtk_array = numpy_support.numpy_to_vtk(np_arry, vtk.VTK_UNSIGNED_CHAR)
    whiteImg.GetPointData().SetScalars(vtk_array)

    # whiteImg.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    #
    # for i in range(count):
    #     whiteImg.GetPointData().GetScalars().SetTuple1(i, 255)

    # scalars = whiteImg.GetPointData().GetScalars()

    # voxel = numpy_support.vtk_to_numpy(scalars)

    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(polydata)
    pol2stenc.SetOutputOrigin(origin)
    pol2stenc.SetOutputSpacing(*tuple(spacing))
    pol2stenc.SetOutputWholeExtent(whiteImg.GetExtent())
    pol2stenc.Update()

    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(whiteImg)
    imgstenc.SetStencilData(pol2stenc.GetOutput())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(0)
    imgstenc.Update()

    img = imgstenc.GetOutput()
    temp_vox = numpy_support.vtk_to_numpy(img.GetPointData().GetScalars())
    vox = temp_vox.reshape(dim[::-1])

    ctr = (bounds[:, 0] + bounds[:, 1]) / 2
    ctr = ctr / spacing
    ctr = ctr[::-1]
    # ctr = (ctr[3:] + ctr[:3])/2

    if return_center or return_origin:
        outs = [vox]
        if return_center:
            outs.append(ctr)
        if return_origin:
            voxel_origin = (-min_bound) / spacing
            outs.append(voxel_origin)
        return outs

    return vox


def create_plane_actors(normal, origin, scale, invert=False):
    if invert:
        normal = normal[::-1]
        origin = origin[::-1]
    plane = vtk.vtkPlaneSource()

    # plane.SetCenter(*plane_normal[::-1])
    plane.SetNormal(*normal)
    plane.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(plane.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    # actor.GetProperty().SetOpacity(0.2)
    actor.GetProperty().SetRepresentationToWireframe()
    actor.GetProperty().SetLineWidth(5)
    actor.GetProperty().LightingOff()

    transform = myTransform()
    transform.Translate(*tuple(origin))
    transform.Scale(scale, scale, scale)
    actor.SetUserTransform(transform)
    return actor


# def show(polydta_or_actor_list, **kwargs):
#     actors = [to_actor(poly_or_actor) for poly_or_actor in polydta_or_actor_list]
#     show_actors(actors, **kwargs)


# def test(sdf, adf, dd):
#     """
#
#     Parameters
#     ----------
#     sdf : Callable[[str], str], show me the money
#     adf : str
#     dd : Tuple[float], dsfdsdfdsf
#     Returns
#     -------
#
#     """
#     sdf.


def split_show(items1, items2, item3=[], keypress_callback=None,
               in_window_size=None, next_func=None, render_colors=[], show=True,
               image_save=False,
               savename='',
               cam_direction=(3, -3, 2), view_up=(0, 0, -1),
               focal_dist_weight=1.3,
               spacing=None,
               rgba=True,
               ):
    """

    Parameters
    ----------
    items1 :
    items2 :
    item3 :
    keypress_callback :
    in_window_size :
    next_func : callable[[str], any] 아이템 교체를 위한 callback function
    render_colors :
    show :
    image_save :
    savename : str,
    cam_direction :
    view_up :
    focal_dist_weight :

    Returns
    -------

    """

    ren1 = vtk.vtkRenderer()
    ren2 = vtk.vtkRenderer()
    ctrs1 = add_actors(ren1, items1, spacing=spacing)
    ctrs2 = add_actors(ren2, items2, spacing=spacing)
    if len(item3) > 0:
        ren3 = vtk.vtkRenderer()
        add_actors(ren3, item3, spacing=spacing)

    # ren1 = show_actors(items1, show=False)
    # ren2 = show_actors(items2, show=False)

    if next_func:
        global g_next_items_fun
        g_next_items_fun = next_func

    # https://examples.vtk.org/site/Cxx/Visualization/MultipleViewports/
    # ren1.SetViewport()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren1)
    renWin.AddRenderer(ren2)
    dw = dh = 500
    w = dw * (2 + bool(item3))
    win_size = in_window_size or (w, dh)
    renWin.SetSize(*win_size)

    if len(item3) > 0:
        renWin.AddRenderer(ren3)

    num_view_port = 2 if len(item3) == 0 else 3
    # xmin, ymin, xmax, ymax in vtk format
    view_ports = [
        [0.0, 0, 1 / num_view_port, 1],
        [1. / num_view_port, 0, 2. / num_view_port, 1],
    ]

    if len(item3) > 0:
        view_ports.append([2 / num_view_port, 0, 3 / num_view_port, 1])

    default_colors = [
        # alice_blue,
        # ghost_white,
        # white_smoke,
        # seashell
        black,  # = (0.0000, 0.0000, 0.0000)
        ivory_black,  # = (0.1600, 0.1400, 0.1300)
        lamp_black,  # = (0.1800, 0.2800, 0.2300)
    ]
    view_colors = render_colors or default_colors

    ren1.SetViewport(view_ports[0])
    ren2.SetViewport(view_ports[1])
    ren1.SetBackground(view_colors[0])
    ren2.SetBackground(view_colors[1])

    if ctrs1:
        ctrs = np.mean(ctrs1, axis=0)
    elif ctrs2:
        ctrs = np.mean(ctrs2, axis=0)
    else:
        ctrs = np.zeros([3])
    # item_bounds = gen_items_bounds(ren1)
    # item_ctr = (item_bounds[0] + item_bounds[1])
    cam = ren1.GetActiveCamera()
    # cam.SetPosition(*tuple(item_ctr))
    cam.SetFocalPoint(*tuple(ctrs))
    # sharing camera
    ren2.SetActiveCamera(cam)

    if len(item3) > 0:
        ren3.SetViewport(view_ports[2])
        # if len(view_colors) >= 3:
        c = default_colors[2] if len(render_colors) < 3 else view_colors[2]
        ren3.SetBackground(c)
        ren3.SetActiveCamera(cam)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    if show:
        iren.AddObserver("KeyPressEvent", keypress_callback or keypress_event)
        iren.Initialize()
        renWin.Render()
        iren.Start()
    elif image_save:
        capture_image_from_render_window(renWin, iren,
                                         cam_direction=cam_direction, savename=savename,
                                         focal_dist_weight=focal_dist_weight, view_up=view_up, rgba=rgba)


def volume_coloring(volume_array, **kwargs):
    """
    volume_integer_color=volume_integer_color, volume_floating_color=
    """

    # threshold = kwargs.get('threshold')
    # division = kwargs.get('division', 7)
    # if threshold is None:
    threshold = kwargs.get('threshold', 0.30)
    if np.issubdtype(volume_array.dtype, np.integer):
        scale = volume_array.max()
        threshold = 0.5  # np.min(volume_array[volume_array>0]) / scale
        norm_vol = volume_array
    else:
        scale = 1000
        vmin, vmax = np.min(volume_array), np.max(volume_array)
        norm_vol = (volume_array - vmin) / (vmax - vmin)
        norm_vol = scale * norm_vol
        if np.issubdtype(volume_array.dtype, np.integer):
            norm_vol = norm_vol.astype(volume_array.dtype)

    #     denorm_value = 1000
    #     denorm_volume = norm_voume * denorm_value
    #     threshold = denorm_value / 2
    # else:
    #     denorm_volume = norm_voume
    kwargs.update({
        'max_value': scale,
        'threshold': scale * threshold
    })

    volume = numpyvolume2vtkvolume(norm_vol, scale / 2, spacing=kwargs.get('spacing'))
    return set_volume_prop(volume, **kwargs)


# def to_vtkvolu
def get_color_table(name: str, number: int):
    if name == 'rainbow':
        return get_rainbow_color_table(number)
    elif name == 'normal':
        color_table = [

            colors.indian_red,
            colors.flesh,
            # (0.0, 0.0, 0.0),
            colors.ivory,
            colors.antique_white,
        ]
        # antique_white
        return color_table
    else:
        return get_rainbow_color_table(7)


def set_volume_prop(volume, **kwargs):
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.ShadeOn()
    volumeProperty.SetInterpolationType(vtk.VTK_LINEAR_INTERPOLATION)
    threshold = kwargs.get('threshold', 500)
    division = kwargs.get('division', 7)
    max_value = kwargs.get('max_value', 1000)
    pallete = kwargs.get('pallete', 'normal')
    # threshold = 140, max_value = 255, division = 3

    color = vtk.vtkColorTransferFunction()
    # color.AddRGBPoint(threshold, *blue_light)
    colors = get_color_table(pallete, division)

    values = np.linspace(threshold, max_value, len(colors))
    for v, c in zip(values, colors):
        color.AddRGBPoint(v, *c)
        # color.AddHSVPoint(v, c)
    # color.AddRGBPoint(threshold, *antique_white)
    # color.AddRGBPoint(threshold, *colors_list[0])

    opacity = vtk.vtkPiecewiseFunction()
    # opacity.AddPoint(-3024, 0)
    # opacity.AddPoint(0, 0)
    op_value = np.linspace(0, max_value, division) / max_value

    opacity.AddPoint(0, 0.00)
    # for v in op_value:
    for v, op in zip(values, op_value):
        opacity.AddPoint(v, op)
    # opacity.AddPoint(threshold, opacity_scalar)

    volumeProperty.SetColor(color)
    volumeProperty.SetScalarOpacity(opacity)
    volume.SetProperty(volumeProperty)
    return volume


def add_actors(renderer: vtk.vtkRenderer, actors, **kwargs):
    ren = renderer
    ctrs = []

    # if isinstance(actors, dict):
    #     actors = list(actors.values())
    # if isinstance(actors, dict):
    #
    # else:
    for act in actors:
        if isinstance(act, dict):
            act_values = list(act.values())
            ctr_act_values = add_actors(renderer, act_values)
            ctrs.extend(ctr_act_values)
        elif isinstance(act, list):
            ctr_values = add_actors(renderer, act)
            ctrs.extend(ctr_values)
        elif isinstance(act, vtk.vtkVolume):
            ren.AddVolume(act)
        elif isinstance(act, vtk.vtkActor):
            ren.AddActor(act)
        elif isinstance(act, vtk.vtkPolyData):
            ren.AddActor(polydata2actor(act))
        elif isinstance(act, np.ndarray):
            if act.ndim == 1 and act.size == 3:
                act = create_sphere(act[None])[0]
                ren.AddActor(act)
            elif act.ndim == 2:
                pts_act = create_points_actor(act)
                ren.AddActor(pts_act)
                act = pts_act
            elif act.ndim == 3 or (act.ndim == 4 and act.shape[0] == 1):
                act = np.squeeze(act)
                # assumption 0~1 normalize value
                # https://numpy.org/doc/stable/reference/generated/numpy.issubdtype.html
                if np.issubdtype(act.dtype, np.floating):
                    div = 1
                elif np.issubdtype(act.dtype, np.integer):
                    div = 3
                elif np.issubdtype(act.dtype, np.bool_):
                    div = 1
                    act = act.astype(np.float32)
                else:
                    raise ValueError(act.dtype)
                if np.issubdtype(act.dtype, np.integer):
                    act = numpyvolume2vtkvolume(act, 0.5, division=div, spacing=kwargs.get('spacing'))
                else:
                    act = volume_coloring(act, **kwargs)
                # act = volume_coloring(act, **kwargs)
                ren.AddActor(act)
                # raise ValueError(act.shape)
        else:
            ren.AddActor(act)

        if hasattr(act, "GetCenter"):
            ctrs.append(act.GetCenter())

    return ctrs


def capture_image_from_render_window(render_window: vtk.vtkRenderWindow, iren: vtk.vtkRenderWindowInteractor,
                                     savename='',
                                     cam_direction=(1e-2, -1, 0), view_up=(0, 0, 1),
                                     focal_dist_weight=1.3, rgba=True):
    """

    Parameters
    ----------
    render_window :
    iren :
    cam_direction :
    savename : str,
    focal_dist_weight : float, weights from camera-positon to focal-points.
    view_up : tuple[3], camera view-up vector

    Returns
    -------

    """
    ren = render_window.GetRenderers().GetFirstRenderer()
    renWin = render_window

    bounds = []
    # 각각의 렌더러는 카메라를 공유한다고 가정하에서, 모든 렌더러에 있는 object bounds 계산
    for renderer in renWin.GetRenderers():
        for act in renderer.GetViewProps():
            bounds.append(_get_bounds(act))

    if bounds:
        # (n, 2, 3)
        bound_stack = np.stack(bounds, axis=0)
        # compute the box that containes all-items
        bmin = np.min(bound_stack[:, 0], axis=0)
        bmax = np.max(bound_stack[:, 1], axis=0)
    else:
        bmin = np.zeros([3])
        bmax = np.ones([3])

    ctr = (bmin + bmax) / 2.
    box_ext = bmax - bmin
    # 가장 falt한 영역으로 focusing out 처리
    # flat_index = np.argmin(box_ext[max_box_arg])
    # d0 = np.zeros([3])
    # d0[flat_index] = 1.0
    # dist_d0 = np.linalg.norm(box_ext)

    # unit_direct = box_ext / np.linalg.norm(box_ext)
    # 카메라 방향이 기본단우벡터(0, 1, 00 일 경우 렌더러가 정상적으로 처리 되지 않는것으로 보임
    direction = np.asarray(cam_direction)
    direction = direction / np.linalg.norm(direction)
    if np.min(np.abs(direction)) < 1e-3:
        direction[np.argmin(direction)] = 1e-1
        direction = direction / np.linalg.norm(direction)

    cam = ren.GetActiveCamera()
    # focal_dist_w = 1.8
    # cam_pose = ctr - direction * (1.5 * dist_d0)
    d0 = np.max(box_ext) / np.tan(np.deg2rad(cam.GetViewAngle())) * focal_dist_weight
    cam_pose = ctr - d0 * direction  # * (1.5 * dist_d0)

    # cam_pose = (0, 0, 0)
    # ctr + unit_direct * dist_d0 * 10
    savename = savename or f'{time.strftime("%Y%m%d%H%M%S")}.png'
    # savename = f'{time.strftime("%Y%m%d%H%M%S")}.png'
    renWin.SetOffScreenRendering(True)

    cam.SetFocalPoint(*tuple(ctr))
    ren.GetActiveCamera().SetPosition(*tuple(cam_pose))
    ren.GetActiveCamera().SetViewUp(*tuple(view_up))
    # view_up
    # time.sleep(1)
    renWin.Render()
    iren.MouseWheelForwardEvent()
    iren.MouseWheelBackwardEvent()
    renWin.Render()


    save_path = os.path.dirname(savename)

    if save_path:
        os.makedirs(save_path, exist_ok=True)

    imagefilter = vtk.vtkWindowToImageFilter()
    imagefilter.SetInput(renWin)
    # imagefilter.SetM
    imagefilter.ReadFrontBufferOff()
    if rgba:
        imagefilter.SetInputBufferTypeToRGBA()
    else:
        imagefilter.SetInputBufferTypeToRGB()

    imagefilter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(savename)
    writer.SetInputConnection(imagefilter.GetOutputPort())
    writer.Write()

    logger = get_runtime_logger()
    logger.info('save complete:{}'.format(savename))


def show_actors(actors, keypress_callback=None, in_window_size=None, show=True, next_func=None,
                volume_floating_color=None,
                volume_integer_color=None, image_save=False, savename='', cam_direction=(2, -3, 1), view_up=(0, 0, -1),
                focal_dist_weight=1.3,
                spacing=None, rgba=True) -> vtk.vtkRenderer:
    ren = vtk.vtkRenderer()

    if next_func is not None:
        global g_next_items_fun
        g_next_items_fun = next_func
    # ctrs = []

    ctrs = add_actors(ren, actors,
                      volume_integer_color=volume_integer_color,
                      volume_floating_color=volume_floating_color,
                      spacing=spacing)

    win_size = in_window_size or (500, 500)

    if show or image_save:
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)
        # win_size = in_window_size or (500, 500)
        renWin.SetSize(*win_size)

        if len(ctrs) > 0:

            ctrs = np.mean(ctrs, axis=0)
            # ren.GetActiveCamera()->SetViewUp(0, 1, 0);
            ren.GetActiveCamera().SetFocalPoint(*tuple(ctrs))
        else:
            logging.warning('emtpy actors')

        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)

        if image_save:
            capture_image_from_render_window(renWin, iren,
                                             cam_direction=cam_direction, savename=savename,
                                             focal_dist_weight=focal_dist_weight, view_up=view_up, rgba=rgba)

        elif show:
            # iren = vtk.vtkRenderWindowInteractor()
            # iren.SetRenderWindow(renWin)

            # iren = vtk.vtkRenderWindowInteractor()
            # iren.SetRenderWindow(renWin)
            # if show:
            iren.AddObserver("KeyPressEvent", keypress_callback or keypress_event)
            iren.Initialize()
            renWin.Render()
            iren.Start()
        else:
            pass

    return ren

# alias
show = show_actors

def to_actor(polydata_or_actor):
    """

    :param polydata_or_actor: vtk.vtkPolyData or vtk.vtkActor
    :return:
    """
    if isinstance(polydata_or_actor, vtk.vtkPolyData):
        actor = polydata2actor(polydata_or_actor)
    elif issubclass(type(polydata_or_actor), vtk.vtkProp3D):
        actor = polydata_or_actor
    return actor


def polydata2actor(polydata):
    if isinstance(polydata, vtk.vtkPolyData):
        volumeMapper = vtk.vtkPolyDataMapper()
        volumeMapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(volumeMapper)
        return actor
    elif isinstance(polydata, (list, tuple)):
        return [polydata2actor(p) for p in polydata]
    elif isinstance(polydata, dict):
        res = dict()
        for k, v in polydata.items():
            res[k] = polydata2actor(v)
        return res


def actor2polydata(actors):
    if isinstance(actors, vtk.vtkActor):
        return actors.GetMapper().GetInput()
    elif isinstance(actors, (list, tuple)):
        return [actor2polydata(p) for p in actors]
    elif isinstance(actors, dict):
        res = dict()
        for k, v in actors.items():
            res[k] = actor2polydata(v)
        return res


# def show_actors(actors):
#
# def show(polydata):
#     volumeMapper = vtk.vtkPolyDataMapper()
#     volumeMapper.SetInputData(polydata)
#     # volumeMapper.SetBlendModeToComposite()
#
#     actor = vtk.vtkActor()
#     actor.SetMapper(volumeMapper)
#
#     # volume.SetMapper(volumeMapper)
#     # volume.SetProperty(_get_normal_property())
#     # volume.Update()
#
#     # volume = vtk.vtkVolume()
#     # volume.Set
#
#     ren = vtk.vtkRenderer()
#     ren.AddVolume(actor)
#     renWin = vtk.vtkRenderWindow()
#     renWin.AddRenderer(ren)
#
#     # ren.GetActiveCamera()->SetViewUp(0, 1, 0);
#     ren.GetActiveCamera().SetFocalPoint(*actor.GetCenter())
#     # ren.GetActiveCamera()->SetPosition(c[0], c[1], c[2]);
#
#     iren = vtk.vtkRenderWindowInteractor()
#     iren.SetRenderWindow(renWin)
#     iren.Initialize()
#     iren.Start()


def convert_voxel_to_polydata(vtkimg, threshold, radius=1., stddev=2.):
    if isinstance(vtkimg, np.ndarray):
        vtkimg = convert_numpy_vtkimag(vtkimg)
    boneExtractor = vtk.vtkMarchingCubes()
    gaussianRadius = radius
    gaussianStandardDeviation = stddev
    gaussian = vtk.vtkImageGaussianSmooth()
    gaussian.SetStandardDeviations(gaussianStandardDeviation, gaussianStandardDeviation, gaussianStandardDeviation)
    gaussian.SetRadiusFactors(gaussianRadius, gaussianRadius, gaussianRadius)
    # gaussian.SetInputConnection(selectTissue.GetOutputPort())
    gaussian.SetInputData(vtkimg)

    boneExtractor.SetInputConnection(gaussian.GetOutputPort())
    boneExtractor.SetValue(0, threshold)
    boneExtractor.Update()

    return boneExtractor.GetOutput()


def smoothingPolydata(polydata, iterration=15, factor=0.6, edge_smoothing=False):
    smoothFilter = vtk.vtkSmoothPolyDataFilter()
    smoothFilter.SetInputData(polydata)
    smoothFilter.SetNumberOfIterations(iterration)
    smoothFilter.SetRelaxationFactor(factor)
    if not edge_smoothing:
        smoothFilter.FeatureEdgeSmoothingOff()
    else:
        smoothFilter.FeatureEdgeSmoothingOn()
    smoothFilter.BoundarySmoothingOn()
    smoothFilter.Update()
    return smoothFilter.GetOutput()


class myTransform(vtk.vtkTransform):
    def __init__(self, ndarray=None):
        super(myTransform, self).__init__()

        if isinstance(ndarray, np.ndarray) and ndarray.shape == (4, 4):
            self.set_from_numpy_mat(ndarray)

    def getRigidTransform(self):
        out = myTransform()
        out.Translate(*self.GetPosition())
        out.RotateWXYZ(*self.GetOrientationWXYZ())
        return out

    def convert_np_mat(self):
        mat = self.GetMatrix()
        np_mat = np.zeros([4, 4], dtype=np.float64)
        for i in range(4):
            for j in range(4):
                np_mat[i, j] = mat.GetElement(i, j)
        return np_mat

    def GetInverse(self, vtkMatrix4x4=None):
        # inverse_t = super(myTransform, self).GetInverse()
        mat4x4 = self.convert_np_mat()
        t = myTransform()
        t.set_from_numpy_mat(np.linalg.inv(mat4x4))
        return t

    def set_from_numpy_rotatewxyz(self, orientWXYZ=None, trans=None, scales=None):
        if orientWXYZ is not None:
            self.RotateWXYZ(*tuple(orientWXYZ))

        if trans is not None:
            self.Translate(*tuple(trans))

        if scales is not None:
            self.Scale(scales, scales, scales)

    def set_from_numpy_mat(self, np_mat, invert=False):
        if invert:
            invmat = np.eye(4)
            invmat[:3, :3] = np_mat[:3, :3][::-1, ::-1]
            invmat[:3, 3] = np_mat[:3, 3][::-1]
            np_mat = invmat

        mat = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                mat.SetElement(i, j, np_mat[i, j])
        self.SetMatrix(mat)

    def set_from_numpy(self, orient=None, trans=None, scales=None):
        """
        :param orient: 3 array
        :param trans:  3 array
        :param scales: 3 array
        :return:
        """
        if orient is not None:
            self.RotateZ(orient[0])
            self.RotateX(orient[1])
            self.RotateY(orient[2])

        if trans is not None:
            self.Translate(*tuple(trans))

        if scales is not None:
            self.Scale(*tuple(scales))

    def transfrom_numpy(self, np_pts):
        np_mat = self.convert_np_mat()
        ex_pts = np.ones([np_pts.shape[0], 4])
        ex_pts[:, :3] = np_pts
        out = np.dot(np_mat, ex_pts.T).T[:, :3]
        return out

    def transform_only_rotate(self, np_pts):
        np_mat = self.convert_np_mat()
        np_mat[:, 3] = np.array([0, 0, 0, 1])
        ex_pts = np.ones([np_pts.shape[0], 4])
        ex_pts[:, :3] = np_pts
        out = np.dot(np_mat, ex_pts.T).T[:, :3]
        return out


def reconstruct_polydata(points, polys):
    vtk_points_array = numpy_support.numpy_to_vtk(points, vtk.VTK_FLOAT)
    vtkpoints = vtk.vtkPoints()
    vtkpoints.SetData(vtk_points_array)

    cell_array = numpy_support.numpy_to_vtk(polys.reshape([-1, 4]), array_type=vtk.VTK_ID_TYPE)

    cells = vtk.vtkCellArray()
    polys_reshape = polys.reshape([-1, 4])
    cells.SetCells(polys_reshape.shape[0], cell_array)

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(vtkpoints)
    polyData.SetPolys(cells)

    return polyData


def get_rainbow_color_table(number):
    length = len(rainbow_table)
    np_rainbow_table = np.array(rainbow_table)
    if number < np_rainbow_table.shape[0]:
        return np_rainbow_table / 255.
    landmark_color_table = []
    for i in range(number):
        value = (i / (number - 1) * (length - 1))
        upper = np.ceil(value)
        lower = np.floor(value)
        x = abs(value - upper)
        index = int(lower)
        # print(i, value, x, 1-x , index)

        if index == length - 1:
            color = np_rainbow_table[index]
        else:

            if x < 1e-10:
                x = 1

            color = x * np_rainbow_table[index] + (1 - x) * np_rainbow_table[index + 1]

        landmark_color_table.append(tuple(color / 255.0))
    return landmark_color_table


def coloring_rainbow(acts):
    color_table = get_rainbow_color_table(len(acts))
    for i, act in enumerate(acts):
        act.GetProperty().SetColor(*color_table[i])


def get_vert_indices(polydata: vtk.vtkPolyData, points):
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(polydata)
    loc.BuildLocator()
    inds = []
    for pt in points:
        i = loc.FindClosestPoint(*pt)
        inds.append(i)
    return np.asarray(inds)


# numpy converting

def vtk_2_vf(polydata_or_actor):
    if issubclass(type(polydata_or_actor), vtk.vtkActor):
        pts, polys = _actor_2_vf(polydata_or_actor)
    elif issubclass(type(polydata_or_actor), vtk.vtkPolyData):
        pts = numpy_support.vtk_to_numpy(polydata_or_actor.GetPoints().GetData())
        polys = numpy_support.vtk_to_numpy(polydata_or_actor.GetPolys().GetData())
    else:
        raise ValueError
    return pts, polys.reshape([-1, 4])


def create_trasnform(basis: np.ndarray, origin: np.ndarray):
    """
    :param basis: (3, 3) orthornomal basis
    :type basis:
    :param origin: (3,) origin position
    :type origin:
    :return:
    :rtype:
    """

    transform_mat = np.eye(4)
    transform_mat[:3, :3] = basis
    transform_mat[:3, 3] = -np.dot(basis, origin)
    return transform_mat


def create_obb_cube(center: np.ndarray, size: np.ndarray, basis: np.ndarray):
    """
    save with 'create_general_cube(...)'
    :param center: center of bbox [3]
    :param size: bbox size [3]
    :param basis: [3,3] orthornoaml basis
    :return:
    """
    return create_general_cube(center, size, basis)


def create_general_cube(center: np.ndarray, size: np.ndarray, basis: np.ndarray):
    """
    :param center: center of bbox [3]
    :param size: bbox size [3]
    :param basis: [3,3] orthornoaml basis
    :return:
    """
    # get bounding bbox
    # fixed dhw
    # dhw = np.array([32, 15, 15])
    p1 = center - size / 2
    p2 = center + size / 2
    bbox = np.concatenate([p1, p2])
    vtk_bbox = convert_box_norm2vtk(bbox[np.newaxis])
    cube = get_cube(vtk_bbox[0])

    # shift from center to origin
    rot = create_trasnform(basis, np.zeros([3]))
    # rot = create_trasnform(pca.components_[::-1, ::-1].T, np.zeros([3]))
    rot_vtk = myTransform()
    rot_vtk.set_from_numpy_mat(rot)

    vtk_origin = center[::-1]
    t1 = myTransform()
    t1.Translate(*tuple(-vtk_origin))
    t2 = myTransform()
    t2.Translate(*tuple(vtk_origin))

    concat_t = myTransform()
    concat_t.Concatenate(t2)
    concat_t.Concatenate(rot_vtk)
    concat_t.Concatenate(t1)

    cube.SetUserTransform(concat_t)
    return cube


def _actor_2_vf(act):
    v = _actor_2_numpy(act)
    f = _actor_2_numpy_polys(act)
    return v, f


def _actor_2_numpy(act):
    return numpy_support.vtk_to_numpy(act.GetMapper().GetInput().GetPoints().GetData())


def _actor_2_numpy_polys(act):
    return numpy_support.vtk_to_numpy(act.GetMapper().GetInput().GetPolys().GetData()).reshape([-1, 4])


def main():
    path = "D:/PersonalFolder/LJY/SVN/AutoPlanning_branch_new/Server/temp/20200715163528_dio"
    assert os.path.exists(path), "empty file:{}".format(path)
    src_volume, src_spacing = read_dicom(path)
    vol_vtk = numpyvolume2vtkvolume(src_volume * 255, 123)
    show_actors([vol_vtk])


def read_vtk(filename):
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(filename)
    reader.Update()

    loaddata = reader.GetOutput()

    np_voxel_array = numpy_support.vtk_to_numpy(loaddata.GetPointData().GetScalars())
    segment_voxels_array = np_voxel_array.reshape(loaddata.GetDimensions()[::-1])
    return segment_voxels_array


def read_vtk_itksnapped(filename):
    segment_voxels_array = read_vtk(filename)
    segment_voxels_array = segment_voxels_array[::-1, ::-1, :]
    return segment_voxels_array


def show_point_cloud(poly_or_ndarray,
                     scalars=None, without_zero=True, coloring="tooth", point_size=None,
                     scalar_color_table: dict = None, with_scalarbar=True, otehr_actors=[], is_show=True):
    assert coloring in ["auto", "tooth"], "invalid type coloring"
    if isinstance(poly_or_ndarray, vtk.vtkPolyData):
        pass
        in_polydata = poly_or_ndarray
    elif isinstance(poly_or_ndarray, np.ndarray):
        to_polydata = lambda x: create_points_actor(x).GetMapper().GetInput()
        in_polydata = to_polydata(poly_or_ndarray)
    else:
        raise ValueError('invalid type:{}'.format(type(poly_or_ndarray)))

    polydata = vtk.vtkPolyData()
    polydata.DeepCopy(in_polydata)
    used_cell_scalars = False

    if scalars is not None:
        if scalars.dtype == 'bool':
            scalars = scalars.astype(int)
        vtk_scalars = numpy_support.numpy_to_vtk(scalars)
        if polydata.GetNumberOfPoints() == scalars.size:
            polydata.GetPointData().SetScalars(vtk_scalars)
        elif polydata.GetNumberOfPolys() == scalars.size:
            polydata.GetCellData().SetScalars(vtk_scalars)
            used_cell_scalars = True
        else:
            raise ValueError('invalid scalar size')
    else:
        vtk_scalars = polydata.GetPointData().GetScalars() or polydata.GetCellData().GetScalars()
        assert scalars, 'empty scalars'
        scalars = numpy_support.vtk_to_numpy(vtk_scalars)

    if coloring == "tooth":
        if not used_cell_scalars:
            vtk_scalars = polydata.GetPointData().GetScalars()
        else:
            vtk_scalars = polydata.GetCellData().GetScalars()

        scalars_array = numpy_support.vtk_to_numpy(vtk_scalars)
        # 0, 1, 2, ... 로 scalars 로 변경
        scalars_array = scalars_array % 10
        v_scalars = numpy_support.numpy_to_vtk(scalars_array)
        if not used_cell_scalars and polydata.GetNumberOfPoints() == scalars_array.size:
            polydata.GetPointData().SetScalars(v_scalars)
        elif polydata.GetNumberOfPolys() == scalars_array.size:
            polydata.GetCellData().SetScalars(v_scalars)
        else:
            raise ValueError()
    else:
        scalars_array = scalars

        # if polydata.GetPointData().GetScalars() or
    actor = polydata2actor(polydata)

    if np.issubdtype(scalars_array.dtype, np.integer):

        unique_scalar = np.unique(scalars_array)

        color_table = vtk.vtkColorTransferFunction()

        colors = get_rainbow_color_table(len(unique_scalar))

        tables = get_teeth_color_table()

        for i, c in enumerate(unique_scalar):
            if scalar_color_table is None:
                if coloring == "auto":
                    if without_zero:
                        if c == 0:
                            color_table.AddRGBPoint(c, 1, 1, 1)
                        else:
                            color_table.AddRGBPoint(c, *colors[i - 1])
                    else:
                        color_table.AddRGBPoint(c, *colors[i])
                elif coloring == "tooth":
                    o = c % 10
                    color = tables[o]
                    color = _darkening_for_1st_inscisor(color, o)
                    color_table.AddRGBPoint(c, *color)
                else:
                    raise ValueError("error:" + coloring)
            else:
                # if c in scalar_color_table:
                color = scalar_color_table.get(c, (1, 1, 1))
                color_table.AddRGBPoint(c, *color)

        actor.GetMapper().SetLookupTable(color_table)
    else:

        smin, smax = np.min(scalars), np.max(scalars)
        color_table = vtk.vtkColorTransferFunction()

        num_color = 20
        colors = get_rainbow_color_table(num_color)

        vals = np.linspace(smin, smax, num_color)
        # num = 20

        for i, c in enumerate(vals):
            # if coloring == "auto":
            if without_zero:
                if c == 0:
                    color_table.AddRGBPoint(c, 1, 1, 1)
                else:
                    color_table.AddRGBPoint(c, *colors[i - 1])
            else:
                color_table.AddRGBPoint(c, *colors[i])
            # else:
            #     raise ValueError("error:"+coloring)

        actor.GetMapper().SetLookupTable(color_table)

    if point_size:
        actor.GetProperty().SetPointSize(point_size)

    lookuptable = actor.GetMapper().GetLookupTable()
    scalarBar = vtk.vtkScalarBarActor()
    scalarBar.SetLookupTable(lookuptable)

    scalarBar.SetNumberOfLabels(5)
    actors = [actor]
    if with_scalarbar:
        actors.append(scalarBar)

    if actor.GetMapper().GetInput().GetCellData().GetScalars() and used_cell_scalars:
        actor.GetMapper().SetScalarModeToUseCellData()

    for a in otehr_actors:
        actors.append(a)
    if is_show:
        show_actors(actors)
    return actors


def polydata_coloring(polydata, color_value):
    if isinstance(color_value, np.ndarray):
        color_value = color_value.tolist()
    color_value = color_value or (1, 1, 1)
    color_value = np.asarray(color_value)
    assert color_value.size == 3 and np.all(color_value < 3)
    cuint8 = (color_value * 255).astype(np.uint8)
    num = polydata.GetNumberOfPoints()
    color = np.repeat(cuint8[None], num, axis=0)
    vtk_array = numpy_support.numpy_to_vtk(color)
    polydata.GetPointData().SetScalars(vtk_array)


def create_tube(points, radius=1.0, color=None) -> vtk.vtkPolyData:
    """

    Parameters
    ----------
    points : np.ndarray (N, 3) points array
    color : (3,) color 0~1 value in rgb order

    Returns
    -------

    """
    curve_actor = create_curve_actor(points)
    curve_data = curve_actor.GetMapper().GetInput()

    tubes = vtk.vtkTubeFilter()
    tubes.SetNumberOfSides(16)
    tubes.SetInputData(curve_data)
    tubes.SetRadius(radius)
    tubes.Update()
    tube_poly = tubes.GetOutput()
    if color is not None:
        polydata_coloring(tube_poly, color)
    return tube_poly


def tri_filter(polydata):
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(polydata)
    tri.Update()
    return tri.GetOutput()


class ColorTransfer(vtk.vtkColorTransferFunction):
    def __init__(self, scalars_tables: Dict[float, Union[np.ndarray, List[float]]]):
        super(ColorTransfer, self).__init__()

        self.out_of_range = True
        self.scalars_tables = scalars_tables
        for scalar, color in scalars_tables.items():
            assert len(color) == 3, '3 length values for color-r,g,b'
            self.AddRGBPoint(scalar, *color)

    def mapping(self, scalars):

        # color_range = self.GetRange()
        colors = []
        for s in scalars:
            c = self.GetColor(s)
            colors.append(c)
        return np.array(colors)


def show_target_with_scalars(pd_or_numpy: vtk.vtkPolyData, numpy_scalars=None, show=True, inrange=None,
                             color_tables: Dict[float, Union[np.ndarray, List[float]]] = None, input_actors=[], scalar_bar_name=''):
    if isinstance(pd_or_numpy, np.ndarray):
        pts_actor = create_points_actor(pd_or_numpy)
        pd = pts_actor.GetMapper().GetInput()
    else:
        # _actor_2_vf()
        pd = to_polydata(pd_or_numpy)

    copy_pd = vtk.vtkPolyData()
    copy_pd.DeepCopy(pd)
    used_cell_scalars = False
    if numpy_scalars is not None:
        if numpy_scalars.dtype == 'bool':
            numpy_scalars = numpy_scalars.astype('float')
        if numpy_scalars.shape[0] == pd.GetNumberOfPoints():
            vtk_scalars = numpy_support.numpy_to_vtk(numpy_scalars)
            copy_pd.GetPointData().SetScalars(vtk_scalars)
        elif numpy_scalars.shape[0] == pd.GetNumberOfPolys():
            vtk_scalars = numpy_support.numpy_to_vtk(numpy_scalars, array_type=vtk.VTK_FLOAT)
            copy_pd.GetCellData().SetScalars(vtk_scalars)
            used_cell_scalars = True
        else:
            raise ValueError("scalar shape {}".format(numpy_scalars.shape))

    elif copy_pd.GetPointData().GetScalars() is not None:
        vtk_scalars = copy_pd.GetPointData().GetScalars()
        assert vtk_scalars, "empty scalars and invalid scalar array"

        numpy_scalars = numpy_support.vtk_to_numpy(vtk_scalars)
    elif copy_pd.GetCellData().GetScalars() is not None:
        vtk_scalars = copy_pd.GetCellData().GetScalars()
        numpy_scalars = numpy_support.vtk_to_numpy(vtk_scalars)

    if color_tables is None:
        if inrange is None:
            dmin, dmax = numpy_scalars.min(), numpy_scalars.max()
        else:
            dmin, dmax = inrange
    else:
        vals = np.array([v for v in color_tables.keys()])
        dmin, dmax = np.min(vals), np.max(vals)

    copy_actor = polydata2actor(copy_pd)
    copy_actor.GetMapper().SetScalarRange(dmin, dmax)
    lookuptable = copy_actor.GetMapper().GetLookupTable()

    if color_tables is not None:
        transfer = ColorTransfer(color_tables)
        colors_array = transfer.mapping(numpy_scalars)
        vtk_color_array = numpy_support.numpy_to_vtk((colors_array * 255).astype(np.uint8))
        vtk_color_array.SetName("Colors")
        copy_pd.GetPointData().AddArray(vtk_color_array)
        # https://vtkusers.public.kitware.narkive.com/pwyOi07Y/multiple-scalars-in-vtkpolydata
        copy_pd.GetPointData().SetActiveScalars("Colors")
    else:
        lookuptable.Build()
        from scipy.interpolate import RegularGridInterpolator
        look_up = numpy_support.vtk_to_numpy(lookuptable.GetTable())
        look_up_array = np.array(look_up)[:, :3] / 255
        numpy_range = np.linspace(dmin, dmax, look_up_array.shape[0])
        interp = RegularGridInterpolator((numpy_range,), look_up_array)
        colors_array = interp(np.clip(numpy_scalars, dmin, dmax))
        vtk_color_array = numpy_support.numpy_to_vtk((colors_array * 255).astype(np.uint8))
        vtk_color_array.SetName("Colors")

        copy_pd.GetPointData().AddArray(vtk_color_array)
        # https://vtkusers.public.kitware.narkive.com/pwyOi07Y/multiple-scalars-in-vtkpolydata
        copy_pd.GetPointData().SetActiveScalars("Colors")

    scalarBar = vtk.vtkScalarBarActor()

    scalarBar.SetTitle(scalar_bar_name)
    if color_tables is None:
        label_size = 5
        scalarBar.SetNumberOfLabels(label_size)
        scalarBar.SetLookupTable(lookuptable)
    else:
        labels_vtk_array = numpy_support.numpy_to_vtk(np.array([v for v in color_tables.keys()]).astype(np.float64))
        scalarBar.SetCustomLabels(labels_vtk_array)
        scalarBar.SetLookupTable(transfer)

    if copy_pd.GetCellData().GetScalars() and used_cell_scalars:
        copy_actor.GetMapper().SetScalarModeToUseCellData()
    items = [copy_actor, scalarBar]
    if show:
        show_actors([copy_actor, scalarBar, *input_actors])
    return items


def to_polydata(polydata_or_actor):
    if issubclass(type(polydata_or_actor), vtk.vtkActor):
        return polydata_or_actor.GetMapper().GetInput()
    elif issubclass(type(polydata_or_actor), vtk.vtkPolyData):
        return polydata_or_actor
    else:
        raise ValueError('not implemented')


def point_pair(p1: np.ndarray, p2: np.ndarray, invert=False, color=None):
    """

    Parameters
    ----------
    p1 : np.ndarray (N, 3)
    p2 : np.ndarray (N, 3

    Returns
    -------

    """
    assert p1.shape == p2.shape

    d12 = p2 - p1

    actor = create_vector(d12, p1, 1, invert=invert)
    change_actor_color(actor, color)
    return actor


def clip_polydata_by_plane(polydata, plane_origin, plane_unit_normal):
    plane = vtk.vtkPlane()
    plane.SetOrigin(tuple(plane_origin))
    plane.SetNormal(tuple(plane_unit_normal))

    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(polydata)
    clipper.SetClipFunction(plane)
    clipper.Update()
    return clipper.GetOutput(), plane


def cutter_polydata_by_plane(polydata, plane_origin, plane_unit_normal):
    plane = vtk.vtkPlane()
    plane.SetOrigin(tuple(plane_origin))
    plane.SetNormal(tuple(plane_unit_normal))

    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(polydata)
    cutter.Update()
    return cutter.GetOutput()


def get_axis(ctr, basis, size=10) -> List[vtk.vtkActor]:
    """
    default color
    x - red
    y = green
    z = blue
    p0 = ctr
    p1 = ctr + basis * size
    create axis

    Parameters
    ----------
    ctr :
        (3,)
    basis :
        (3, 3)
    size : float

    Returns
    -------

    """
    assert ctr.size == 3
    assert basis.shape == (3, 3)

    p1 = np.expand_dims(ctr, axis=0) + basis * size
    p0 = np.expand_dims(ctr, axis=0).repeat(3, axis=0)
    color = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    ]
    props = []
    for a, b, c in zip(p0, p1, color):
        arrow = create_arrow(a, b)
        arrow.GetProperty().SetColor(c)
        props.append(arrow)
    return props


def _get_bounds(act: vtk.vtkActor):
    if act.GetBounds():
        return np.array(act.GetBounds()).reshape([-1, 2]).T
    else:
        return np.zeros([2, 3], dtype='float32')


def gen_items_bounds(ren: vtk.vtkRenderer) -> np.ndarray:
    bounds = []
    for act in ren.GetViewProps():
        bounds.append(_get_bounds(act))

    # (n, 2, 3)
    bound_stack = np.stack(bounds, axis=0)
    # compute the box that containes all-items
    bmin = np.min(bound_stack[:, 0], axis=0)
    bmax = np.min(bound_stack[:, 1], axis=0)
    return np.stack([bmin, bmax], axis=0)


def capture_image(vtk_items, save_path, image_size=(500, 500)):
    ren = show_actors(vtk_items, show=False)

    bounds = []
    for act in ren.GetViewProps():
        bounds.append(_get_bounds(act))

    # (n, 2, 3)
    bound_stack = np.stack(bounds, axis=0)
    # compute the box that containes all-items
    bmin = np.min(bound_stack[:, 0], axis=0)
    bmax = np.max(bound_stack[:, 1], axis=0)
    # box_ext = bound_stack[:, 1] - bound_stack[:, 0]
    # box_ext = bound_stack[]

    ctr = (bmin + bmax) / 2.
    box_ext = bmax - bmin
    # 가장 falt한 영역으로 focusing out 처리
    # flat_index = np.argmin(box_ext[max_box_arg])
    # d0 = np.zeros([3])
    # d0[flat_index] = 1.0
    dist_d0 = np.linalg.norm(box_ext)

    unit_direct = box_ext / np.linalg.norm(box_ext)
    focus = ctr + unit_direct * dist_d0 * 10

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(*image_size)
    renWin.SetOffScreenRendering(True)
    # renWin.GetRendere
    # ctrs = np.mean(ctrs, axis=0)
    # ren.GetActiveCamera()->SetViewUp(0, 1, 0);
    ren.GetActiveCamera().SetFocalPoint(*tuple(ctr))
    ren.GetActiveCamera().SetPosition(*tuple(focus))

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    # iren.AddObserver("KeyPressEvent", keypress_callback or keypress_event)
    iren.MouseWheelForwardEvent()
    iren.MouseWheelBackwardEvent()
    renWin.Render()

    # iren.Initialize()
    # iren.Start()

    # save_dir = "test_image"
    # if not os.path.exists(save_dir):
    os.makedirs(save_path, exist_ok=True)

    imagefilter = vtk.vtkWindowToImageFilter()

    imagefilter.SetInput(renWin)
    # imagefilter.SetM
    imagefilter.ReadFrontBufferOff()
    imagefilter.SetInputBufferTypeToRGBA()
    imagefilter.Update()

    lt = time.localtime()

    str_time = "{}{}{}.png".format(lt.tm_hour, lt.tm_min, lt.tm_sec)
    savename = os.path.join(save_path, str_time)
    # savename = filename.format(str_time)
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(savename)
    writer.SetInputConnection(imagefilter.GetOutputPort())
    writer.Write()

    logger = get_runtime_logger()
    logger.info('save complete:{}'.format(savename))


def append_polydata(polys: List[vtk.vtkPolyData]):
    appender = vtk.vtkAppendPolyData()
    for poly in polys:
        appender.AddInputData(poly)
    appender.Update()
    return appender.GetOutput()


def show_point_cloud(in_polydata, scalars=None, without_zero=True, coloring="tooth", point_size=None,
                     scalar_color_table: dict = None, with_scalarbar=True, otehr_actors=[], is_show=True):
    assert coloring in ["auto", "tooth"], "invalid type coloring"
    if isinstance(in_polydata, vtk.vtkPolyData):
        pass
        # in_polydata = in_polydata
    # elif isinstance(in_polydata, vedo.Mesh):
    #     in_polydata = in_polydata.polydata()
    else:
        raise ValueError('invalid type:{}'.format(type(in_polydata)))
    polydata = vtk.vtkPolyData()
    polydata.DeepCopy(in_polydata)
    used_cell_scalars = False

    if scalars is not None:
        if scalars.dtype == 'bool':
            scalars = scalars.astype(int)
        vtk_scalars = numpy_support.numpy_to_vtk(scalars)
        if polydata.GetNumberOfPoints() == scalars.size:
            polydata.GetPointData().SetScalars(vtk_scalars)
        elif polydata.GetNumberOfPolys() == scalars.size:
            polydata.GetCellData().SetScalars(vtk_scalars)
            used_cell_scalars = True
        else:
            raise ValueError('invalid scalar size')
    else:
        vtk_scalars = polydata.GetPointData().GetScalars() or polydata.GetCellData().GetScalars()
        assert scalars, 'empty scalars'
        scalars = numpy_support.vtk_to_numpy(vtk_scalars)

    if coloring == "tooth":
        if not used_cell_scalars:
            vtk_scalars = polydata.GetPointData().GetScalars()
        else:
            vtk_scalars = polydata.GetCellData().GetScalars()

        scalars_array = numpy_support.vtk_to_numpy(vtk_scalars)
        # 0, 1, 2, ... 로 scalars 로 변경
        scalars_array = scalars_array % 10
        v_scalars = numpy_support.numpy_to_vtk(scalars_array)
        if not used_cell_scalars and polydata.GetNumberOfPoints() == scalars_array.size:
            polydata.GetPointData().SetScalars(v_scalars)
        elif polydata.GetNumberOfPolys() == scalars_array.size:
            polydata.GetCellData().SetScalars(v_scalars)
        else:
            raise ValueError()
    else:
        scalars_array = scalars

        # if polydata.GetPointData().GetScalars() or
    actor = polydata2actor(polydata)

    if np.issubdtype(scalars_array.dtype, np.integer):

        unique_scalar = np.unique(scalars_array)

        color_table = vtk.vtkColorTransferFunction()

        colors = get_rainbow_color_table(len(unique_scalar))

        tables = get_teeth_color_table()

        # if scalar_color_table is None:

        for i, c in enumerate(unique_scalar):
            if scalar_color_table is None:
                if coloring == "auto":
                    if without_zero:
                        if c == 0:
                            color_table.AddRGBPoint(c, 1, 1, 1)
                        else:
                            color_table.AddRGBPoint(c, *colors[i - 1])
                    else:
                        color_table.AddRGBPoint(c, *colors[i])
                elif coloring == "tooth":
                    o = c % 10
                    color = tables[o]
                    color_table.AddRGBPoint(c, *color)
                else:
                    raise ValueError("error:" + coloring)
            else:
                # if c in scalar_color_table:
                color = scalar_color_table.get(c, (1, 1, 1))
                color_table.AddRGBPoint(c, *color)

        actor.GetMapper().SetLookupTable(color_table)
    else:

        smin, smax = np.min(scalars), np.max(scalars)
        color_table = vtk.vtkColorTransferFunction()

        num_color = 20
        colors = get_rainbow_color_table(num_color)

        vals = np.linspace(smin, smax, num_color)
        # num = 20

        for i, c in enumerate(vals):
            # if coloring == "auto":
            if without_zero:
                if c == 0:
                    color_table.AddRGBPoint(c, 1, 1, 1)
                else:
                    color_table.AddRGBPoint(c, *colors[i - 1])
            else:
                color_table.AddRGBPoint(c, *colors[i])
            # else:
            #     raise ValueError("error:"+coloring)

        actor.GetMapper().SetLookupTable(color_table)

    if point_size:
        actor.GetProperty().SetPointSize(point_size)

    lookuptable = actor.GetMapper().GetLookupTable()
    scalarBar = vtk.vtkScalarBarActor()
    scalarBar.SetLookupTable(lookuptable)

    scalarBar.SetNumberOfLabels(5)
    actors = [actor]
    if with_scalarbar:
        actors.append(scalarBar)

    if actor.GetMapper().GetInput().GetCellData().GetScalars() and used_cell_scalars:
        actor.GetMapper().SetScalarModeToUseCellData()

    for a in otehr_actors:
        actors.append(a)
    if is_show:
        show_actors(actors)
    return actors


def show_pc(points, scalars, point_size=None, otehr_actors=[], coloring="auto", is_show=True):
    """

    Args:
        points ():
        scalars ():
        point_size ():
        otehr_actors ():
        coloring (): 'auto' or 'tooth'

    Returns:

    """

    assert coloring in ["tooth", "auto"]
    to_polydata = lambda x: create_points_actor(x).GetMapper().GetInput()
    return show_point_cloud(to_polydata(points), scalars, point_size=point_size, otehr_actors=otehr_actors,
                            coloring=coloring, is_show=is_show)


def clear_polydata(poly):
    poly.GetPointData().Reset()
    poly.GetCellData().Reset()


def render_polydata(polydata_or_actor) -> vtk.vtkPolyData:
    """
    입려깅 actor일경우, transform 정보가 잇는지 확인 후 해당 변환을 적용을 시킨 후 polydata로 반환
    Parameters
    ----------
    polydata_or_actor :

    Returns
    -------

    """
    if isinstance(polydata_or_actor, vtk.vtkActor):
        polydata = polydata_or_actor.GetMapper().GetInput()
        # t = myTransform() or
        t0 = polydata_or_actor.GetUserTransform()
        t = myTransform()
        if t0 is not None:
            t.DeepCopy(t0)
        polydata = apply_transform_polydata(polydata, t)
    else:
        polydata = polydata_or_actor
    return polydata


def read_medit_obj(filename):
    import igl
    igl_load = igl.read_obj(filename)
    assert len(igl_load) == 6
    verts, faces = igl_load[0], igl_load[3]
    points = verts[:, :3]
    colors = verts[:, 3:] if verts.shape[1] > 3 else np.zeros([verts.shape[0]], dtype=verts.dtype)
    scale = 255.0 if colors.max() < (1.0 + 1e-3) else 1.0
    colors_uint8 = (colors * scale).astype(np.uint8)
    num_faces = faces.shape[0]
    faces_vtk = np.concatenate([np.full([num_faces, 1], 3), faces], axis=-1)
    poly = reconstruct_polydata(points, faces_vtk)
    poly.GetPointData().SetScalars(numpy_support.numpy_to_vtk(colors_uint8))
    return poly, colors_uint8


if __name__ == "__main__":
    main()

from ctypes import *
import numpy as np
import imageio

renderlib = cdll.LoadLibrary("./rd.so")
# int rd_init_gl_context(int width, int height)
renderlib.rd_init_gl_context.restype = c_int
renderlib.rd_init_gl_context.argtypes = [c_int, c_int]

# int rd_init_renderer(int context, const char *path)
renderlib.rd_init_renderer.restype = c_int
renderlib.rd_init_renderer.argtypes = [c_int, c_char_p]

# int rd_destroy_renderer(int context)
renderlib.rd_destroy_renderer.restype = c_int
renderlib.rd_destroy_renderer.argtypes = [c_int]

# int rd_destroy_gl_context(int context)
renderlib.rd_destroy_gl_context.restype = c_int
renderlib.rd_destroy_gl_context.argtypes = [c_int]

# unsigned char *rd_renderd(int context, float *extrinsic, float *intrinsic, size_t sze, size_t szi)
renderlib.rd_render.restype = c_int
renderlib.rd_render.argtypes = [c_int, POINTER(c_float), c_size_t, 
        POINTER(c_float), POINTER(c_float), c_size_t, c_size_t]

#############################################
c = renderlib.rd_init_gl_context(1280, 960)
print(c)
print(renderlib.rd_init_renderer(c, b"./data/test_male.obj"))

# extracted from data/parameters/0001
w2c = np.array([[0., 0., 9.9999999999999978e-01, 0.],
        [0., -9.9999999999999978e-01, 0., 7.4999999999999978e-01],
        [9.9999999999999978e-01, 0., 0., 1.7100000381469722e+00],
        [0., 0., 0., 1.]])
c2w = np.linalg.inv(w2c)
extrinsic = (c_float*16)(*c2w.reshape(-1).tolist())
intrinsic = (c_float*9)(
        7.29575195e+02, 0., 6.49034302e+02, 
        0., 8.19848328e+02, 4.76053406e+02, 
        0., 0., 1.)
depth_buffer = np.zeros((960, 1280), dtype=np.float32)
print(renderlib.rd_render(c, depth_buffer.ctypes.data_as(POINTER(c_float)), depth_buffer.size * 4,
        extrinsic, intrinsic, len(extrinsic), len(intrinsic)))
imageio.imwrite("rd_0001.png", depth_buffer)

# extracted from data/parameters/0002
w2c = np.array([[-8.1932550941111393e-01, 0., 5.7332862271843588e-01, 3.1660829668464396e-02],
        [0., -9.9999999999999989e-01, 0., 7.4999999999999989e-01],
        [5.7332862271843588e-01, 0., 8.1932550941111393e-01, 1.7720038642343678e+00],
        [0., 0., 0., 1.]])
c2w = np.linalg.inv(w2c)
extrinsic = (c_float*16)(*c2w.reshape(-1).tolist())
print(renderlib.rd_render(c, depth_buffer.ctypes.data_as(POINTER(c_float)), depth_buffer.size * 4,
        extrinsic, intrinsic, len(extrinsic), len(intrinsic)))
imageio.imwrite("rd_0002.png", depth_buffer)

print(renderlib.rd_destroy_renderer(c))
print(renderlib.rd_destroy_gl_context(c))

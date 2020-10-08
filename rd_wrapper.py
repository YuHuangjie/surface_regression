from ctypes import *
import numpy as np
import imageio

class rd_wrapper:
        def __init__(self, height, width, obj_path):
                renderlib = cdll.LoadLibrary("./rd_wrapper/rd.so")
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

                c = renderlib.rd_init_gl_context(width, height)
                if c < 0:
                        raise Exception('rd_init_gl_context fail')

                r = renderlib.rd_init_renderer(c, obj_path.encode('utf8'))
                if r < 0:
                        raise Exception('rd_init_renderer fail')

                self.renderlib = renderlib
                self.context = c

        def clean(self):
                self.renderlib.rd_destroy_renderer(self.context)
                self.renderlib.rd_destroy_gl_context(self.context)

        def render_depth(self, c2w, K, buffer):
                if not buffer.dtype == np.float32:
                        raise Exception('need float32 buffer')
                extrinsic = (c_float*16)(*c2w.reshape(-1).tolist())
                intrinsic = (c_float*9)(*K.reshape(-1).tolist())
                r = self.renderlib.rd_render(self.context, buffer.ctypes.data_as(POINTER(c_float)), 
                        buffer.size * 4, extrinsic, intrinsic, len(extrinsic), len(intrinsic))
                if r < 0:
                        raise Exception('rd_render fail')


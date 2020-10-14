/*
 * A simple tool for rendering depth from a .obj file. Note that this tool 
 * supports only one instance, i.e., you can't call InitGLContext twice without
 * releasing one of them.
 *
 * Yu Huangjie - 2020/7/3
 */
#include <stdexcept>
#include <stdio.h>
#include <mutex>
#include <algorithm>
#include <glad/glad_egl.h>
#include <glad/glad.h>
#include "Renderer.h"

// struct rd_context {
//         int CAMERA_WIDTH = 0;
//         int CAMERA_HEIGHT = 0;
//         EGLContext context;
//         EGLSurface surface;
//         EGLDisplay display;
//         Renderer *renderer;
// };

// static const int MAX_CONTEXTS = 3;
// static struct rd_context rd_contexts[MAX_CONTEXTS];
// static bool rd_busy[MAX_CONTEXTS];
// static std::mutex lck;     // protect rd_busy

// static struct rd_context gcontext;

extern "C" {

// static struct rd_context* acquire_context(int &i, bool acquire_new = false)
// {
//         struct rd_context *c = nullptr;
//         int cid = -1;

//         if (!acquire_new && (i < 0 || i >= MAX_CONTEXTS)) {
//                 fprintf(stderr, "invalid context\n");
//                 return nullptr;
//         }
//         if (acquire_new) {
//                 std::lock_guard<std::mutex> lk(lck);
//                 for (cid = 0; cid < MAX_CONTEXTS; cid++)
//                         if (!rd_busy[cid])
//                                 break;
//                 if (cid == MAX_CONTEXTS) {
//                         fprintf(stderr, "no free contexts\n");
//                         return nullptr;
//                 }
//                 rd_busy[cid] = true;
//                 i = cid;
//                 return &rd_contexts[cid];
//         } else {
//                 std::lock_guard<std::mutex> lk(lck);
//                 if (!rd_busy[i]) {
//                         fprintf(stderr, "invalid context\n");
//                         return nullptr;
//                 }
//                 return &rd_contexts[i];
//         }
// }

// static void release_context(int i)
// {
//         struct rd_context *c;

//         if (!(c = acquire_context(i)))
//                 return;
//         std::lock_guard<std::mutex> lk(lck);
//         rd_busy[i] = false;
// }

static int CAMERA_WIDTH = 0;
static int CAMERA_HEIGHT = 0;
static EGLDisplay s_display = EGL_NO_DISPLAY;
static EGLContext s_context = EGL_NO_CONTEXT;
static EGLSurface s_surface = EGL_NO_SURFACE;
static std::mutex s_ctx_lck;    // protect context across threads

static Renderer *s_renderer;    // depth renderer 

static EGLDisplay query_display()
{
        const int max_devices = 32;
        EGLDeviceEXT devices[max_devices];
        EGLint num_devices = 0;
        EGLint i;
        EGLint major, minor;
        EGLDisplay display = EGL_NO_DISPLAY;

        // query native devices
        if (!eglQueryDevicesEXT(max_devices, devices, &num_devices) || num_devices <= 0) {
                fprintf(stderr, "eglQueryDevicesEXT fail\n");
                return EGL_NO_DISPLAY;
        }
        for (i = 1; i < num_devices; i++) {
                // get an EGL display connection
                if ((display = eglGetPlatformDisplay(EGL_PLATFORM_DEVICE_EXT,
				devices[i], NULL)) == EGL_NO_DISPLAY)
                        continue;
                // initialize the EGL display connection
                if (!eglInitialize(display, &major, &minor))
                        continue;
                break;
        }
        if (i == num_devices) {
                fprintf(stderr, "eglGetDisplay returns EGL_NO_DISPLAY\n");
                return EGL_NO_DISPLAY;
        }
        return display;
}

static EGLContext query_context(EGLDisplay display, EGLConfig config)
{
        EGLContext context = EGL_NO_CONTEXT;

        context = eglCreateContext(display, config, EGL_NO_CONTEXT, NULL);
        return context;
}

static EGLContext query_surface(EGLDisplay display, EGLConfig config,
        EGLint pbuffer_attrib_list[])
{
        EGLSurface surface = EGL_NO_SURFACE;

        surface = eglCreatePbufferSurface(display, config, pbuffer_attrib_list);
        return surface;
}

/* 
 * Initialize global EGL context. It must be called in the main thread.
 */
int rd_init_gl_context(int width, int height)
{
        EGLint egl_err;
        EGLint num_config;
        EGLConfig config;
        EGLint const attrib_list[] = {
                EGL_RED_SIZE, 8,
                EGL_GREEN_SIZE, 8,
                EGL_BLUE_SIZE, 8,
                EGL_ALPHA_SIZE, 8,
                EGL_DEPTH_SIZE, 8,
                EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                EGL_NONE
        };
        EGLint pbuffer_attrib_list[] = {
                EGL_WIDTH, width,
                EGL_HEIGHT, height,
                EGL_NONE,
        };
        EGLint i;
        EGLDisplay display;
        EGLContext context;
        EGLSurface surface;
        // int free_c = -1;
        // struct rd_context *c = nullptr;

        // look for a free context slot
        // if (!(c = acquire_context(free_c, true)))
        //         return -1;
        // c->context = EGL_NO_CONTEXT;
        // c->surface = EGL_NO_SURFACE;
        // c->display = EGL_NO_DISPLAY;

        if (!gladLoadEGL()) {
                printf("gladLoadEGL fail\n");
                // goto rel_ctx;
                return -1;
        }
        if ((display = query_display()) == EGL_NO_DISPLAY)
                goto gl_err;
        if (!eglBindAPI(EGL_OPENGL_API)) {
                fprintf(stderr, "eglBindAPI fail\n");
                goto gl_err;
        }
        // get an appropriate EGL frame buffer configuration
        if (!eglChooseConfig(display, attrib_list, &config, 1, &num_config)) {
                fprintf(stderr, "eglChooseConfig fail\n");
                goto gl_err;
        }
        // create an EGL rendering context
        if ((context = query_context(display, config)) == EGL_NO_CONTEXT) {
                fprintf(stderr, "query_context fail\n");
                goto gl_err;
        }
        // create a pbuffer surface
        if ((surface = query_surface(display, config, pbuffer_attrib_list)) == EGL_NO_SURFACE) {
                fprintf(stderr, "query_surface fail\n");
                goto gl_err;
        }
        // connect the context to the surface
        if (!eglMakeCurrent(display, surface, surface, context)) {
                fprintf(stderr, "eglMakeCurrent fail. Try another EGLdisplay.\n");
                goto gl_err;
        }
        // load GL functions
        if (!gladLoadGL()) {
                fprintf(stderr, "gladLoadGLLoader fail\n");
                // goto rel_ctx;
                goto gl_err;
        }
        // test OpenGL context is successfully created
        glDeleteShader(glCreateShader(GL_VERTEX_SHADER));
        if (glGetError() != 0) {
                fprintf(stderr, "GL was not successfully setup\n");
                goto gl_err;
        }
        // c->display = display;
        // c->context = context;
        // c->surface = surface;
        // c->CAMERA_HEIGHT = height;
        // c->CAMERA_WIDTH = width;
        s_display = display;
        s_context = context;
        s_surface = surface;
        CAMERA_WIDTH = width;
        CAMERA_HEIGHT = height;
        return 0;

gl_err:
        egl_err = eglGetError();
        printf("eglGetError returns %d\n", egl_err);

        if (surface != EGL_NO_SURFACE)
                eglDestroySurface(display, surface);
        if (context != EGL_NO_CONTEXT)
                eglDestroyContext(display, context);
        if (display != EGL_NO_DISPLAY)
                eglTerminate(display);
        return -1;
// rel_ctx:
//         release_context(free_c);
//         return -1;
}

/* 
 * This function must be called in the main thread and after rd_init_gl_context
 * is called.
 */
int rd_init_renderer(int context, const char *path)
{
        vector<Geometry> geometry;
        GLenum err;
        // struct rd_context *c = nullptr;

        // if (!(c = acquire_context(context)))
        //         return -1;

        while ((err = glGetError() != GL_NO_ERROR))
                fprintf(stderr, "gl error on entry, continue\n");
        try {
                geometry = Geometry::FromObj(path);
        }
        catch (std::runtime_error &e) {
                fprintf(stderr, "caught runtime_error: %s\n", e.what());
                return -1;
        }
        if (geometry.empty()) {
                fprintf(stderr, "mesh empty\n");
                return -1;
        }
        s_renderer = new Renderer();
        s_renderer->SetGeometries(geometry);
        if ((err = glGetError() != GL_NO_ERROR)) {
                fprintf(stderr, "create renderer fail\n");
                delete s_renderer;
                s_renderer = nullptr;
                return -1;
        }
        return 0;
}

int rd_destroy_renderer(int context)
{
        // struct rd_context *c = nullptr;

        // if (!(c = acquire_context(context)))
        //         return -1;
        delete s_renderer;
        s_renderer = nullptr;
        return 0;
}

int rd_destroy_gl_context(int context)
{
        EGLint egl_err;
        // struct rd_context *c = nullptr;

        // if (!(c = acquire_context(context)))
        //         return -1;
        eglDestroySurface(s_display, s_surface);
        s_surface = EGL_NO_SURFACE;
        eglDestroyContext(s_display, s_context);
        s_context = EGL_NO_CONTEXT;
        eglTerminate(s_display);
        s_display = EGL_NO_DISPLAY;
        rd_destroy_renderer(context);
        // release_context(context);
        return 0;
}

/*
 * The egl context is to be shared across threads by eglMakeCurrent. This must
 * be protected by mutual exclusion.
 */
int rd_render(int context, float *buf, size_t szbuf, 
        float *c2w, float *K, size_t sze, size_t szi)
{
        Extrinsic e;
        Intrinsic i;
        GLenum err;
        glm::vec3 pos, target, up;
        float *row0, *row1, *row2;
        // struct rd_context *c = nullptr;
        uint8_t *src = reinterpret_cast<uint8_t*>(buf);
        float *oldbuf = buf;

        // if (!(c = acquire_context(context)))
        //         return -1;
        if (sze != 4*4 || szi != 3*3) {
                fprintf(stderr, "camera parameters sizes are incorrect\n"); 
                return -1;
        }
        if (szbuf != CAMERA_WIDTH * CAMERA_HEIGHT * 4) {
                fprintf(stderr, "buffer size is incorrect\n");
                return -1;
        }
        
        // do a bit decoding
        row0 = c2w;
        row1 = c2w + 4;
        row2 = c2w + 8;
        pos = glm::vec3(row0[3], row1[3], row2[3]);
        up = -glm::vec3(row0[1], row1[1], row2[1]);
        target = pos + glm::vec3(row0[2], row1[2], row2[2]);
        e = Extrinsic(pos, target, up);

        row0 = K;
        row1 = K + 3;
        i = Intrinsic(row0[2], row1[2], row0[0], row1[1], CAMERA_WIDTH, CAMERA_HEIGHT);

        {
                // acquire mutual-excluded context
                std::lock_guard<std::mutex> lck(s_ctx_lck);

                glClearColor(0.f, 0.f, 0.f, 0.f);
                glClear(GL_COLOR_BUFFER_BIT);
                s_renderer->SetCamera(Camera(e, i));
                s_renderer->Render();
                if ((err = glGetError()) != GL_NO_ERROR) {
                        fprintf(stderr, "Render depth fail\n");
                        fprintf(stderr, "GL Error code: %d\n", err);
                }
                // depth into RGBA
                memset(buf, 0, szbuf);
                s_renderer->ScreenShot(src, 0, 0, CAMERA_WIDTH, CAMERA_HEIGHT);
                if ((err = glGetError() != GL_NO_ERROR)) {
                        fprintf(stderr, "Read framebuffer fail\n");
                        fprintf(stderr, "GL Error code: %d\n", err);
                }
                // release context lock so that other threads can render
        }
        // decode to float
        for (int i = 0; i < CAMERA_WIDTH * CAMERA_HEIGHT; i++) {
                // the 1000.f is hardcoded in depth shader
                *buf = static_cast<float>(*reinterpret_cast<int*>(src)) / 1000.f;
                buf++;
                src += 4;
        }
        // flip upside-down
        for(int line = 0; line != CAMERA_HEIGHT / 2; ++line) {
                std::swap_ranges(oldbuf + CAMERA_WIDTH*line, 
                        oldbuf + CAMERA_WIDTH*(line+1),
                        oldbuf + CAMERA_WIDTH*(CAMERA_HEIGHT-line-1));
        }
        return 0;
}

}
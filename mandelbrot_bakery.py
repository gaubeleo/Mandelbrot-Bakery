
import numpy as np
#from scipy.interpolate import interp1d as interpolate

import pyopencl as cl

import os
import cv2 as cv
from math import sqrt

import time


class MandelbrotBackery:
    def __init__(self):
        self.init_opencl()


    def init_opencl(self):
        #self.ctx = cl.create_some_context()

        platform = cl.get_platforms()
        my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
        
        self.ctx = cl.Context(devices=my_gpu_devices)

        self.queue = cl.CommandQueue(self.ctx)
        self.mf = cl.mem_flags

        self.prg = cl.Program(self.ctx, r"""
            #pragma OPENCL EXTENSION cl_khr_fp64 : enable

            __kernel void mandelbrot(const ushort max_iter,
                                     const ushort height, const ushort width,
                                     const double re0, const double re1, 
                                     const double im0, const double im1,
                                     __global const uchar *gradient, 
                                     __global uchar *result,
                                     __global uchar *img)
            {
                int gid = get_global_id(0);

                int2 pos;
                pos.x = gid % width;
                pos.y = gid / width;

                //assert(width*height == length(result_img))
                //assert(pos.x <= width && pos.y <= height);
                //assert(re0 < re1 && im0 < im1);

                double2 c;
                c.x = re0 + pos.x * (re1 - re0) / width;
                c.y = im0 + pos.y * (im1 - im0) / height;

                double2 z = c;
                double temp = 0;

                for(uint iter = 0; iter < max_iter; iter++) {
                    temp = (z.x*z.x - z.y*z.y) + c.x;
                    z.y = (2*z.x*z.y) + c.y;
                    z.x = temp;

                    if (z.x*z.x + z.y*z.y > 4.0f){
                        result[pos.y*width+pos.x] = iter;
                        for (unsigned i=0; i < 3; i++)
                            img[(pos.y*width+pos.x)*3 + i] = gradient[iter*3 + i];
                        return;
                    }
                }
                result[pos.y*width+pos.x] = 0;
                for (unsigned i=0; i < 3; i++)
                    img[(pos.y*width+pos.x)*3 + i] = 0;
            }
            """).build()



    def mandelbrot_opencl(self, max_iter, height, width, boundries):
        assert (len(boundries) == 4)
        re0, re1, im0, im1 = boundries

        gradient = create_gradient(max_iter)
        gradient_opencl = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=gradient)

        img = np.zeros((height*width*3,), dtype=np.uint8)
        img_opencl = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, img.nbytes)

        result = np.empty(width*height, dtype=np.uint8)
        result_opencl = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, result.nbytes)

        self.prg.mandelbrot(self.queue, result.shape, None, 
            np.uint16(max_iter), 
            np.uint16(height), np.uint16(width),
            np.float64(re0), np.float64(re1), 
            np.float64(im0), np.float64(im1), 
            gradient_opencl,
            result_opencl,
            img_opencl)

        cl.enqueue_copy(self.queue, result, result_opencl).wait()
        cl.enqueue_copy(self.queue, img, img_opencl).wait()

        img = img.reshape((height, width, 3))

        return result, img


def mandelbrot_cpu(max_iter, height, width, boundries):
    assert (len(boundries) == 4)
    re0, re1, im0, im1 = boundries

    gradient = create_gradient(max_iter)
    gradient = gradient.reshape((max_iter, 3))

    img = np.zeros((height, width, 3), dtype=np.uint8)
    result = np.empty(width*height, dtype=np.uint8)

    for y in range(height):
        print float(y)/height
        for x in range(width):
            c = (re0 + x*(re1 - re0) / width) + (im0 + y*(im1 - im0) / height)*1j
            z = c

            result[y*width+x] = 0;
            img[y, x] = [0, 0, 0]

            for i in range(max_iter):
                z = z**2 +c

                if abs(z)**2 > 4.:
                    result[y*width+x] = i;
                    img[y, x] = gradient[i]
                    break

    return result, img



def calc_boundries(re_center, im_center, re_diameter, ratio):
    assert(type(re_center) == float and type(im_center) == float and type(re_diameter) == float)

    im_diameter = re_diameter/ratio

    re0 = re_center - re_diameter/2
    re1 = re_center + re_diameter/2
    im0 = im_center - im_diameter/2
    im1 = im_center + im_diameter/2

    return np.array([re0, re1, im0, im1])


def create_gradient(count):
    gradient = np.zeros((count, 3))

    colors = np.array([[203, 107, 32],
        [255, 255, 237],
        [0, 170, 255],
        [48, 2, 49],
        [100, 7, 0]])

    #colors = np.array([[100, 100, 100], [200, 200, 200]])

    for i in xrange(count):
        sqrt_i = sqrt(i)/7
        float_i = sqrt_i-int(sqrt_i)
        gradient[i] = (1.-float_i)*colors[int(sqrt_i) % len(colors)] + float_i*colors[(int(sqrt_i)+1) % len(colors)]

    return gradient.reshape((count*3, )).astype(np.uint8)

def save_img(img, show=True):
    if not os.path.isdir(".\\results"):
        os.mkdir(".\\results\\")
        
    i = 1
    while os.path.isfile("results\\mandelbrot_%02i.png"%i):
        i += 1
    cv.imwrite("results\\mandelbrot_%02i.png"%i, img)

    if show:
        cv.namedWindow("Mandelbrot Bakery", cv.WND_PROP_FULLSCREEN)
        cv.setWindowProperty("Mandelbrot Bakery", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.imshow("Mandelbrot Bakery", img)
        cv.waitKey()
        cv.destroyAllWindows()

def create_animation(max_iter, height, width, re_center, im_center, re_diameter, steps=1000, fps=30.):

    #fourcc = cv.VideoWriter_fourcc(*'XVID')
    video = cv.VideoWriter('results\\zoom_animation.avi', -1, fps, (width, height))

    #boundries = start_boundries
    #boundries_step = 1./steps * (end_boundries - start_boundries)

    bakery = MandelbrotBackery()

    for frame in range(steps):
        boundries = calc_boundries(re_center, im_center, re_diameter, float(width)/height)

        result, img = bakery.mandelbrot_opencl(max_iter, height, width, boundries)
        video.write(img)

        re_diameter *= 1.025

        print frame

    video.release()


#mandelbrot = mandelbrot_opencl
#mandelbrot = mandelbrot_cpu

max_iter = 4000

width = 1920/2
height = 1200/2

re_center = -0.45
im_center = 0.0
re_diameter = 3.7

start_boundries = calc_boundries(re_center, im_center, re_diameter, float(width)/height)

#re_center = -0.7435669
#im_center = 0.1314023
#re_diameter = 0.0028

#re_center = -0.7436447
#im_center = 0.1318251
#re_diameter = 0.0000036


re_center = -0.743643887037151
im_center = 0.131825904205330
re_diameter = 0.000000000051299

end_boundries = calc_boundries(re_center, im_center, re_diameter, float(width)/height)


if __name__ == "__main__":
    #boundries = calc_boundries(re_center, im_center, re_diameter, float(width)/height)
    #result, img = mandelbrot(max_iter, height, width, boundries)
    #save_img(img, show=True)

    create_animation(max_iter, height, width, re_center, im_center, re_diameter)
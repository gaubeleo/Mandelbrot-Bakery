
import numpy as np
#from scipy.interpolate import interp1d as interpolate

import pyopencl as cl

import os
import cv2 as cv
from math import sqrt

import time


def mandelbrot_opencl(max_iter, height, width, boundries):
    assert (len(boundries) == 4)
    re0, re1, im0, im1 = boundries

    #ctx = cl.create_some_context()
    platform = cl.get_platforms()
    my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(devices=my_gpu_devices)

    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    prg = cl.Program(ctx, r"""
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

    gradient = create_gradient(max_iter)
    gradient_opencl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gradient)

    img = np.zeros((height*width*3,), dtype=np.uint8)
    img_opencl = cl.Buffer(ctx, mf.WRITE_ONLY, img.nbytes)

    result = np.empty(width*height, dtype=np.uint8)
    result_opencl = cl.Buffer(ctx, mf.WRITE_ONLY, result.nbytes)

    prg.mandelbrot(queue, result.shape, None, 
        np.uint16(max_iter), 
        np.uint16(height), np.uint16(width),
        np.float64(re0), np.float64(re1), 
        np.float64(im0), np.float64(im1), 
        gradient_opencl,
        result_opencl,
        img_opencl)

    cl.enqueue_copy(queue, result, result_opencl).wait()
    cl.enqueue_copy(queue, img, img_opencl).wait()

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


def calc_boundries(re_center, im_center, re_diameter, ratio):
    im_diameter = re_diameter/ratio

    re0 = re_center - re_diameter/2
    re1 = re_center + re_diameter/2
    im0 = im_center - im_diameter/2
    im1 = im_center + im_diameter/2

    return (re0, re1, im0, im1)


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


mandelbrot = mandelbrot_opencl
mandelbrot = mandelbrot_cpu

max_iter = 4000

width = 1920/10
height = 1200/10

re_center = -0.45
im_center = 0.0
re_diameter = 3.7

re_center = -0.7435669
im_center = 0.1314023
re_diameter = 0.0028

re_center = -0.7436447
im_center = 0.1318251
re_diameter = 0.0000036

#re_center = -0.743643887037151
#im_center = 0.131825904205330
#re_diameter = 0.000000000051299

assert(type(re_center) == float and type(im_center) == float and type(re_diameter) == float)

boundries = calc_boundries(re_center, im_center, re_diameter, float(width)/height)

if __name__ == "__main__":
    result, img = mandelbrot(max_iter, height, width, boundries)

    print "hi"
    #print list(img)

    #img = np.zeros((height, width, 3), np.uint8)
    #for i, iteration in enumerate(result):
    #    img[i//width][i%width] = choose_color(iteration, max_iter, gradient)

    #filename = "mandebrot_%i_%i_iter_%i"

    save_img(img, show=True)
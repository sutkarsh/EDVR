
"""
Image resizing and degradation models for ref-guided X tasks.
"""

import numpy
from PIL import Image
import skimage.filters
import skimage.transform
from scipy.ndimage.filters import convolve
import time

target_res_vimeo = (512, 288)

def resize_pil(I):
    """
    I: PIL image
    Result: PIL image
    """
    return I.resize(target_res_vimeo)

def resize_numpy(I):
    """
    This is unnecessarily slow, but uses float numpy input and output, so it can be useful as a visualization function.
    """
    return skimage.img_as_float(numpy.asarray(resize_pil(Image.fromarray(skimage.img_as_ubyte(I)))))

def make_noise(sigma):
    def noise(I):
        """
        I: numpy array in float format, range [0, 1]
        Result: numpy array.
        """
        assert len(I.shape) == 3
        assert I.dtype in ['float32', 'float64']
        return I + numpy.random.normal(size=I.shape) * sigma
    return noise

noise = make_noise(0.3)
noise_1 = make_noise(0.1)

def downsample_4x_upsample(a):
    assert len(a.shape) == 3
    assert a.dtype in ['float32', 'float64']
    I = skimage.transform.resize(a, [a.shape[0]//4, a.shape[1]//4], order=3)
    return skimage.transform.resize(I, a.shape, order=1)

def blur_8x(I):
    assert len(I.shape) == 3
    assert I.dtype in ['float32', 'float64']
    return skimage.filters.gaussian(I, 4.5, multichannel=True)

def blur_4x(I):
    """
    This is very close to downsample_4x_upsample. The PSNR between them is 36.9 on a uniform random image and 40+ on real images.
    """
    assert len(I.shape) == 3
    assert I.dtype in ['float32', 'float64']
    return skimage.filters.gaussian(I, 2.25, multichannel=True)

def motion_blur(I):
    # In the average case, this is about 4x slower than noise/blur
    # I guess it is hard to get significantly faster without a C extension
    
    assert len(I.shape) == 3
    assert I.dtype in ['float32', 'float64']
    
    theta = numpy.random.uniform(0.0, 2*numpy.pi)
    
    # Construct the kernel
    sigma_x = 5.0
    sigma_y = 0.5
    stds = 3
    window = int(stds*sigma_x)
    window += (window % 2 == 0)
    X, Y = numpy.meshgrid(numpy.arange(-window, window+1), numpy.arange(-window, window+1))
    c = numpy.cos(theta)
    s = numpy.sin(theta)
    X, Y = (X * c - Y * s, X * s + Y * c)
    G = numpy.exp(-(X**2) / (2*sigma_x**2)) * numpy.exp(-(Y**2) / (2*sigma_y**2))

    # Discard ends of the kernel if there is not enough energy in them
    min_falloff = numpy.exp(-(stds*sigma_x)**2/(2*sigma_x**2))
    G_sum = G.sum()
    while G.shape[0] >= 3 and G[0,:].sum() < min_falloff*G_sum and G[-1,:].sum() < min_falloff*G_sum:
        G = G[1:-1,:]
        G_sum = G.sum()
    while G.shape[1] >= 3 and G[:,0].sum() < min_falloff*G_sum and G[:,-1].sum() < min_falloff*G_sum:
        G = G[:,1:-1]
        G_sum = G.sum()
    G = G/G_sum
    
    return numpy.dstack(tuple([convolve(I[:,:,channel], G) for channel in range(I.shape[2])]))

def test():
    import pylab
    a=skimage.io.imread('../vimeo_90k_pairs_v3/train/disparity0/121585991_00001_00009_a.png')
    ap=resize_numpy(a)
    #ap = numpy.random.uniform(size=(4000,4000,3))

    def bench(method_str, f):
        T0 = time.time()
        n = 100
        for i in range(n):
            res = f(ap)
        time_per = (time.time()-T0) / n
        print('%s: %f' % (method_str, time_per))

    #bench('noise', noise)
    #bench('blur', blur)
    #bench('motion_blur', motion_blur)

    #b1 = blur_4x(ap)
    #b2 = downsample_4x_upsample(ap)
    """
    b1 = blur(ap)
    b2 = downsample_8x_upsample(ap)
    rms = ((b2-b1)**2).mean() ** 0.5
    psnr = 20 * numpy.log(1.0 / rms) / numpy.log(10)
    print('b1 range:', b1.min(), b1.max())
    print('rms(b2, b1):', rms)
    print('psnr:', psnr)
    """
    
    pylab.imshow(ap)
    pylab.figure()
    pylab.imshow(noise(ap))
    pylab.figure()
    pylab.imshow(blur(ap))
    #pylab.figure()
    #pylab.imshow(downsample_8x_upsample(ap))
    
    for i in range(0):
        pylab.figure()
        pylab.imshow(motion_blur(ap))
    pylab.show()

if __name__ == '__main__':
    test()



from gan.inception_score import get_inception_score
from gan.fid import calculate_fid_given_arrays
import numpy as np
from tqdm import tqdm


def compute_scores(epoch, image_shape, generator, dataset, number_of_images=50000, compute_inception=True, compute_fid=True,
                   log_file=None, additional_info=""):
    if not (compute_inception or compute_fid):
        return
    images = np.empty((number_of_images, ) + image_shape)
    previous_batsh_size = dataset._batch_size
    dataset._batch_size = 100
    for i in tqdm(range(0, 50000, 100)):
        g_s = dataset.next_generator_sample_test()
        images[i:(i+100)] = generator.predict(g_s)
    images *= 127.5
    images += 127.5
    dataset._batch_size = previous_batsh_size

    if compute_inception:
        str = "INCEPTION SCORE: %s, %s" % get_inception_score(images)
        print (str)
        if log_file is not None:
            with open(log_file, 'a') as f:
                print >>f, ("Epoch %s " % (epoch, )) + str + " " + additional_info
    if compute_fid:
        true_images = 127.5 * dataset._X + 127.5
        str = "FID SCORE: %s" % calculate_fid_given_arrays([true_images, images])
        print (str)
        if log_file is not None:
            with open(log_file, 'a') as f:
                print >>f, ("Epoch %s " % (epoch, )) + str + " " + additional_info

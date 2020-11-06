import tensorflow as tf
import kerastuner as kt
from utils.generic_utils import print_log
import utils.config as config
import os
import datetime
import time
from utils.analyzing_data import multiclass_analysis
import pickle


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    # cast to float32 for one_hot encode (otherwise TRUE/FALSE tensor)
    return tf.cast(parts[-2] == config.CLASS_NAMES, tf.float32)


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=config.CHANNELS)  # tf.image.decode_jpeg(img, channels=CHANNELS)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [config.IMG_DIM, config.IMG_DIM])


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def prepare_for_training(ds, batch_size, cache=True, shuffle_buffer_size=1000, loop=False):
    """
    cache:  If isinstance(cache, str), then represents the name of a
            directory on the filesystem to use for caching elements in this Dataset.
            Otherwise, the dataset will be cached in memory.
    """
    # IF it is a small dataset, only load it once and keep it in memory.
    # OTHERWISE use `.cache(filename)` to cache preprocessing work for datasets that don't fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    if loop:
        ds = ds.repeat()

    ds = ds.batch(batch_size)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=config.AUTOTUNE)

    return ds


def prepare_ds(caching, my_set, cache_name, batch_train):
    if caching:
        # delete previous cache files and store for this execution
        caching_file_base = config.main_path + "temp/"
        for f in os.listdir(caching_file_base):
            if "{}.tfcache".format(cache_name) in f:
                os.remove(caching_file_base + f)
        set_ds = prepare_for_training(my_set, batch_size=batch_train,
                                      cache=caching_file_base + "{}.tfcache".format(cache_name))

    else:
        set_ds = prepare_for_training(my_set, batch_size=batch_train)

    return set_ds


def get_ds(name_ds, ds_info):
    # Load filepaths
    file_paths = ds_info[name_ds]
    # Create tf.Dataset from filepaths
    file_paths_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    return file_paths_ds


def initialization(arguments, class_info, ds_info):

    # GLOBAL SETTINGS
    config.AUTOTUNE = tf.data.experimental.AUTOTUNE
    config.CHANNELS = arguments.channels
    config.IMG_DIM = arguments.image_size
    config.CLASS_NAMES = class_info['class_names']
    config.BATCH_SIZE = arguments.batch_size

    print("LOADING AND PRE-PROCESSING DATA")

    dataset_base = config.main_path + arguments.dataset

    # STATS
    size_train, size_val, size_test = class_info['train_size'], class_info['val_size'], class_info['test_size']
    class_names, nclasses = class_info['class_names'], class_info['n_classes']

    # Print information on log
    # EXECUTION Info
    mode_info = "load_model = {}".format(arguments.load_model) if arguments.load_model is not None \
        else \
        "tuning = {}".format(arguments.tuning) if arguments.tuning is not None else "mode = {}".format(arguments.mode)
    print_log("INFO EXECUTION:"
              "\n{}\nmodel = {}\ndataset = {}"
              "\noutput_model = {}\nepochs = {}\nbatch_size = {}\ncaching = {}"
              "\n----------------"
              .format(mode_info, arguments.model, arguments.dataset,
                      arguments.output_model, arguments.epochs, arguments.batch_size, arguments.caching))

    # DATA Info
    print_log("INFO DATA:"
              "\num_classes = {}\nclass_names= {}\nnsize_img= {}x{}\nSize train-val-test= {}-{}-{}"
              .format(nclasses, class_names, arguments.image_size, arguments.channels, size_train, size_val, size_test))
    for ds_class in class_names:
        print_log("{} : {}-{}-{} -> {}".format(ds_class, class_info['info'][ds_class]['TRAIN'],
                                               class_info['info'][ds_class]['VAL'],
                                               class_info['info'][ds_class]['TEST'],
                                               class_info['info'][ds_class]['TOT']))
    print_log("----------------")


def train_val(arguments, model, ds_info):

    # Create tf.Dataset from ds_info e filepaths
    train_paths_ds, val_paths_ds = get_ds('train_paths', ds_info), get_ds('val_paths', ds_info)

    # --------------  TRAINING and VALIDATION part  --------------------

    #  Use Dataset.map to create a dataset of image, label pairs
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    lab_train_ds = train_paths_ds.map(process_path, num_parallel_calls=config.AUTOTUNE)
    lab_val_ds = val_paths_ds.map(process_path, num_parallel_calls=config.AUTOTUNE)

    # Caching dataset in memory for big dataset (IF arguments.caching is set)
    train_ds, val_ds = prepare_ds(arguments.caching, lab_train_ds, "train", arguments.batch_size),\
                       prepare_ds(arguments.caching, lab_val_ds, "val", arguments.batch_size)

    print_log('Start Training for {} epochs  '.format(arguments.epochs), print_on_screen=True)

    # Initialize callbacks for Tensorboard
    log_fit = "tensorboard_logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback_fit = tf.keras.callbacks.TensorBoard(log_dir=log_fit, histogram_freq=1)

    train_results = model.fit(x=train_ds, batch_size=arguments.batch_size, epochs=arguments.epochs,
                              validation_data=val_ds, callbacks=[tensorboard_callback_fit])

    print_log("\ttraining loss: {} \n\ttraining acc:{} \n\tvalidation loss:{} \n\tvalidation acc:{}"
              .format(train_results.history['loss'], train_results.history['acc'],
                      train_results.history['val_loss'], train_results.history['val_acc']))

    del train_ds, val_ds


def train_test(arguments, model, class_info, ds_info):

    # Create tf.Dataset from ds_info e filepaths
    final_training_paths_ds = get_ds('final_training_paths', ds_info)

    # --------------  FINAL TRAINING and TEST part  --------------------

    #  Use Dataset.map to create a dataset of image, label pairs
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    lab_final_train_ds = final_training_paths_ds.map(process_path, num_parallel_calls=config.AUTOTUNE)

    # NB The batch_size for testing is set to 1 to make easier the calculation of the performance results
    fin_train_ds = prepare_ds(arguments.caching, lab_final_train_ds, "fin_tr", arguments.batch_size)

    # Train the model over the entire total_training set and then test
    print_log('Start Final Training for {} epochs  '.format(arguments.epochs), print_on_screen=True)
    start_training = time.perf_counter()
    final_train_results = model.fit(x=fin_train_ds, batch_size=arguments.batch_size, epochs=arguments.epochs)
    end_training = time.perf_counter()
    print_log("\ttraining loss: {} \n\ttraining acc:{}".format(final_train_results.history['loss'],
                                                               final_train_results.history['acc']))
    print_log("FINAL TRAINING TIME: {} ".format(str(datetime.timedelta(seconds=end_training - start_training))))

    del fin_train_ds

    # Test the trained model over the test set
    test(arguments, model, class_info, ds_info)


def test(arguments, model, class_info, ds_info):

    # Create tf.Dataset from ds_info e filepaths
    test_paths_ds = get_ds('test_paths', ds_info)

    # --------------  TEST part  --------------------

    #  Use Dataset.map to create a dataset of image, label pairs
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    lab_test_ds = test_paths_ds.map(process_path, num_parallel_calls=config.AUTOTUNE)

    # NB The batch size for test is set to 1 to make easier the calculation of the performance results
    test_ds = prepare_ds(arguments.caching, lab_test_ds, "test", 1)

    # Test the trained model over the test set
    print_log('Start Test', print_on_screen=True)
    results = model.evaluate(test_ds)
    print_log("\ttest loss: {} \n\ttest accuracy: {}".format(results[0], results[1]), print_on_screen=True)
    print_log("\tPrec: {} \n\tRecall: {}".format(results[2], results[3]), print_on_screen=True)
    # F-measure calculated as (2 * Prec * Recall)/(Prec + Recall)
    print_log("\tF-Measure: {} \n\tAUC: {}"
              .format((2 * results[2] * results[3]) / (results[2] + results[3]), results[4]), print_on_screen=True)

    # TODO: split evaluation and prediction in two phases -> at the moment, the test set is first used by model.evaluate
    # to get cumulative information, and then is again used by model.predict to get per class information, thus, the
    # test process is repeated two times!
    print("Calculating performances per class, it may take a while...")
    cm, results_classes, to_print = multiclass_analysis(model, test_ds, class_info['class_names'],
                                                        save_fig=config.main_path + "results/figures/CM_{}"
                                                        .format(config.timeExec))
    print_log("Results per classes", print_on_screen=True)
    print_log(to_print, print_on_screen=True)

    del test_ds


def save_model(arguments, model):
    model_path = config.main_path + 'model_saved/{}_m{}_i{}x{}' \
        .format(arguments.output_model, arguments.model, arguments.image_size, arguments.channels)

    # save model and architecture to single file
    if arguments.output_model is not None:
        tf.keras.models.save_model(model, model_path, overwrite=False)

        with open(model_path + '.info', 'wb') \
                as filehandle:
            store_data = {"CLASS_NAMES": config.CLASS_NAMES, "CHANNELS": config.CHANNELS, "IMG_DIM": config.IMG_DIM}
            pickle.dump(store_data, filehandle)

        print_log("Model, Weights and Info saved to 'model_saved/{}_m{}_i{}x{}[.info]'"
                  .format(arguments.output_model, arguments.model, arguments.image_size, arguments.channels),
                  print_on_screen=True)


def load_model(arguments):
    print("LOADING MODEL")
    model_path = config.main_path + 'model_saved/{}_m{}_i{}x{}'\
        .format(arguments.load_model, arguments.model, arguments.image_size, arguments.channels)
    if not os.path.isdir(model_path):
        print("Model not found in {}, exiting...".format(model_path))
        exit()

    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['acc', tf.keras.metrics.Precision(name="prec"),
                           tf.keras.metrics.Recall(name="rec"), tf.keras.metrics.AUC(name='auc')])
    with open(model_path + ".info", 'rb') \
            as filehandle:
        stored_data = pickle.load(filehandle)
        _ = stored_data["CLASS_NAMES"]
        C = stored_data["CHANNELS"]
        ID = stored_data["IMG_DIM"]
        if C != config.CHANNELS or ID != config.IMG_DIM:
            # TODO: Handle this case, the model should work in any case even if trained on different data, isn't it?
            print("CHANNELS and IMG_DIM of the loaded model DIFFERS from the required! Exiting...")
            print("Asking {} {} but found {} {}".format(config.CHANNELS, config.IMG_DIM, C, ID))
            exit()
    return model


def save_weights(arguments, model):
    print("SAVING WEIGHTS")
    model.save_weights(config.main_path + 'model_saved/{}_m{}_weights'.format(arguments.output_model, arguments.model))


def load_weights(arguments, model):
    print("LOADING WEIGHTS")
    model.load_weights(config.main_path + 'model_saved/{}_m{}_weights'.format(arguments.load_model, arguments.model))
    return model


def tuning(arguments, model_class, ds_info):

    # Create tf.Dataset from ds_info e filepaths
    train_paths_ds, val_paths_ds = get_ds('train_paths', ds_info), get_ds('val_paths', ds_info)

    #  Use Dataset.map to create a dataset of image, label pairs
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    lab_train_ds = train_paths_ds.map(process_path, num_parallel_calls=config.AUTOTUNE)
    lab_val_ds = val_paths_ds.map(process_path, num_parallel_calls=config.AUTOTUNE)

    # Caching dataset in memory for big dataset (IF arguments.caching is set)
    train_ds, val_ds = prepare_ds(arguments.caching, lab_train_ds, "train", arguments.batch_size), \
                       prepare_ds(arguments.caching, lab_val_ds, "val", arguments.batch_size)

    print("TUNING MODEL HYPER-PARAMETERS with '{}'".format(arguments.tuning))
    tuner = None
    # https://github.com/keras-team/keras-tuner
    if arguments.tuning == "hyperband":
        tuner = kt.Hyperband(
            model_class.build_tuning,
            tune_new_entries=True,  # =False prevents unlisted parameters from being tuned
            objective='val_acc',
            max_epochs=30,
            hyperband_iterations=2,
            directory='tuning',
            project_name='wt_hyperband'
        )
    elif arguments.tuning == "bayesian":
        tuner = kt.tuners.BayesianOptimization(
            model_class.build_tuning,
            objective='val_acc',
            max_trials=50,
            directory='tuning',
            project_name='wt_bayesian'
        )
    elif arguments.tuning == "random":
        tuner = kt.tuners.RandomSearch(
            model_class.build_tuning,
            objective='val_acc',
            max_trials=5,
            executions_per_trial=3,
            directory='tuning',
            project_name='wt_random'
        )

    # # """Case #4:
    # # - We restrict the search space
    # # - This means that default values are being used for params that are left out
    # # """
    #
    # hp = HyperParameters()
    # hp.Choice('learning_rate', [1e-1, 1e-3])
    #
    # tuner = RandomSearch(
    #     build_model,
    #     max_trials=5,
    #     hyperparameters=hp,
    #     tune_new_entries=False,
    #     objective='val_accuracy')

    tuner.search_space_summary()
    tuner.search(train_ds, validation_data=val_ds)
    tuner.results_summary()
    # best_model = tuner.get_best_models(1)[0]
    # best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    del train_ds, val_ds
# -*- coding: utf-8 -*-
"""
Integrate a model with the DEEP API
"""

import json
import argparse
import pkg_resources
# import project's config.py
import flowersegmodelprediction.config as cfg
from aiohttp.web import HTTPBadRequest

from functools import wraps

## Authorization
from flaat import Flaat
flaat = Flaat()


def _catch_error(f):
    """Decorate function to return an error as HTTPBadRequest, in case
    """
    @wraps(f)
    def wrap(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            raise HTTPBadRequest(reason=e)
    return wrap


def _fields_to_dict(fields_in):
    """
    Example function to convert mashmallow fields to dict()
    """
    dict_out = {}
    
    for key, val in fields_in.items():
        param = {}
        param['default'] = val.missing
        param['type'] = type(val.missing)
        if key == 'files' or key == 'urls':
            param['type'] = str

        val_help = val.metadata['description']
        if 'enum' in val.metadata.keys():
            val_help = "{}. Choices: {}".format(val_help, 
                                                val.metadata['enum'])
        param['help'] = val_help

        try:
            val_req = val.required
        except:
            val_req = False
        param['required'] = val_req

        dict_out[key] = param
    return dict_out


def get_metadata():
    """
    Function to read metadata
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.get_metadata
    :return:
    """

    module = __name__.split('.', 1)

    try:
        pkg = pkg_resources.get_distribution(module[0])
    except pkg_resources.RequirementParseError:
        # if called from CLI, try to get pkg from the path
        distros = list(pkg_resources.find_distributions(cfg.BASE_DIR, 
                                                        only=True))
        if len(distros) == 1:
            pkg = distros[0]
    except Exception as e:
        raise HTTPBadRequest(reason=e)

    ### One can include arguments for train() in the metadata
    train_args = _fields_to_dict(get_train_args())
    # make 'type' JSON serializable
    for key, val in train_args.items():
        train_args[key]['type'] = str(val['type'])

    ### One can include arguments for predict() in the metadata
    predict_args = _fields_to_dict(get_predict_args())
    # make 'type' JSON serializable
    for key, val in predict_args.items():
        predict_args[key]['type'] = str(val['type'])

    meta = {
        'name': None,
        'version': None,
        'summary': None,
        'home-page': None,
        'author': None,
        'author-email': None,
        'license': None,
        'help-train' : train_args,
        'help-predict' : predict_args
    }

    for line in pkg.get_metadata_lines("PKG-INFO"):
        line_low = line.lower() # to avoid inconsistency due to letter cases
        for par in meta:
            if line_low.startswith(par.lower() + ":"):
                _, value = line.split(": ", 1)
                meta[par] = value

    return meta


def warm():
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.warm
    :return:
    """
    # e.g. prepare the data


def get_predict_args():
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.get_predict_args
    :return:
    """
    return cfg.PredictArgsSchema().fields


@_catch_error
def predict(**kwargs):
    """
    Function to execute prediction
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.predict
    :param kwargs:
    :return:
    """

    if (not any([kwargs['urls'], kwargs['files']]) or
            all([kwargs['urls'], kwargs['files']])):
        raise Exception("You must provide either 'url' or 'data' in the payload")

    if kwargs['files']:
        kwargs['files'] = [kwargs['files']]  # patch until list is available
        return _predict_data(kwargs)
    elif kwargs['urls']:
        kwargs['urls'] = [kwargs['urls']]  # patch until list is available
        return _predict_url(kwargs)


def _predict_data(*args):
    """
    (Optional) Helper function to make prediction on an uploaded file
    """
    image_list = [image.filename for image in args['files']]
    #Fonctions utiles
    def redimension(image):
        X = np.zeros((1,256,256,3),dtype=np.uint8)
        #img = cv2.imread(image,cv2.COLOR_BGR2RGB)
        img = imread(image)[:,:,:3]
        size_ = img.shape
        img = img[:,:,:3]
        X[0] = resize(img, (256, 256), mode='constant', preserve_range=True)
        return X,size_

    def dice_coefficient(y_true, y_pred):
        eps = 1e-6
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + eps) #eps pour éviter la division par 0 
    
    try:
        #Step 1 : Reshaped image
        image_sample_reshaped = []
        size_original = []

        for sample in image_list:
            image_reshaped,size_ = redimension(sample)
            image_sample_reshaped.append(image_reshaped)
            x,y,z = size_
            size_original.append((x,y))
        #End Step 1
        #Step 2 : Prediction
        model_new = tf.keras.models.load_model("best_model_FL_BCE_0_5.h5",custom_objects={'dice_coefficient': dice_coefficient})
        #Model imported

        image_prediction = []
        for sample in image_sample_reshaped:
            prediction = model_new.predict(sample)

            #Application du seuil de segmentation optimisé
            preds_test_t = (prediction > 0.30000000000000004)
            image_prediction.append(preds_test_t)
        #End Step 2
        #Step 3 : Exportation
        image_output = []
        for sample,i,name in zip(image_prediction,range(len(size_original)),image_sample):
            preds_test_t = resize(sample[0,:,:,0], size_original[i], mode='constant', preserve_range=True)
            cv2.imwrite(name+"_output",preds_test_t*255)
        return {'Information':'Files exported'}
        #End Step 3
    message = 'Not implemented (predict_data())'
    message = {"Error": message}
    return message



def _predict_url(*args):
    """
    (Optional) Helper function to make prediction on an URL
    """
    message = 'Not implemented (predict_url())'
    message = {"Error": message}
    return message


def get_train_args():
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.get_train_args
    :param kwargs:
    :return:
    """
    return cfg.TrainArgsSchema().fields


###
# @flaat.login_required() line is to limit access for only authorized people
# Comment this line, if you open training for everybody
# More info: see https://github.com/indigo-dc/flaat
###
@flaat.login_required() # Allows only authorized people to train
def train(**kwargs):
    """
    Train network
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.train
    :param kwargs:
    :return:
    """

    message = { "status": "ok",
                "training": [],
              }

    # use the schema
    schema = cfg.TrainArgsSchema()
    # deserialize key-word arguments
    train_args = schema.load(kwargs)
    
    # 1. implement your training here
    # 2. update "message"
    
    train_results = { "Error": "No model implemented for training (train())" }
    message["training"].append(train_results)

    return message


# during development it might be practical 
# to check your code from CLI (command line interface)
def main():
    """
    Runs above-described methods from CLI
    (see below an example)
    """

    if args.method == 'get_metadata':
        meta = get_metadata()
        print(json.dumps(meta))
        return meta
    elif args.method == 'predict':
        # [!] you may need to take special care in the case of args.files [!]
        results = predict(**vars(args))
        print(json.dumps(results))
        return results
    elif args.method == 'train':
        results = train(**vars(args))
        print(json.dumps(results))
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model parameters', 
                                     add_help=False)

    cmd_parser = argparse.ArgumentParser()
    subparsers = cmd_parser.add_subparsers(
                            help='methods. Use \"deep_api.py method --help\" to get more info', 
                            dest='method')

    ## configure parser to call get_metadata()
    get_metadata_parser = subparsers.add_parser('get_metadata', 
                                         help='get_metadata method',
                                         parents=[parser])
    # normally there are no arguments to configure for get_metadata()

    ## configure arguments for predict()
    predict_parser = subparsers.add_parser('predict', 
                                           help='commands for prediction',
                                           parents=[parser]) 
    # one should convert get_predict_args() to add them in predict_parser
    # For example:
    predict_args = _fields_to_dict(get_predict_args())
    for key, val in predict_args.items():
        predict_parser.add_argument('--%s' % key,
                               default=val['default'],
                               type=val['type'],
                               help=val['help'],
                               required=val['required'])

    ## configure arguments for train()
    train_parser = subparsers.add_parser('train', 
                                         help='commands for training',
                                         parents=[parser]) 
    # one should convert get_train_args() to add them in train_parser
    # For example:
    train_args = _fields_to_dict(get_train_args())
    for key, val in train_args.items():
        train_parser.add_argument('--%s' % key,
                               default=val['default'],
                               type=val['type'],
                               help=val['help'],
                               required=val['required'])

    args = cmd_parser.parse_args()
    
    main()

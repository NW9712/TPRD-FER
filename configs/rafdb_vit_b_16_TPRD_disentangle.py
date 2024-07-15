from easydict import EasyDict as edict

config = edict()
# weights for testing
# config.checkpoints = '/home/niewei/pythonWork/FER/results_ouput/weights_TPRD/RAFDB/TPRD/CLIP-ViT-B-16/epoch21_iter4224_acc0.9289000034332275.pth'
# config.occlusion = False
# config.headpose = False

# config.checkpoints = '/home/niewei/pythonWork/FER/results_ouput/weights_TPRD/RAFDB-Occlu/TPRD/CLIP-ViT-B-16/epoch21_iter4224_acc0.9289000034332275.pth'
# config.occlusion = True
# config.headpose = False

# config.checkpoints = '/home/niewei/pythonWork/FER/results_ouput/weights_TPRD/RAFDB-Pose30/TPRD/CLIP-ViT-B-16/epoch21_iter4224_acc0.9289000034332275.pth'
# config.occlusion = False
# config.headpose = 30

config.checkpoints = '/home/niewei/pythonWork/FER/results_ouput/weights_TPRD/RAFDB-Pose45/TPRD/CLIP-ViT-B-16/epoch21_iter4224_acc0.9289000034332275.pth'
config.occlusion = False
config.headpose = 45

# model
config.model = 'TPRD_disentangle'
config.clip_model = 'ViT-B/16'
config.expression_prompts = ['surprise','fear','disgust','happy','sad','anger','neutral']
# config.expression_prompts = ['a surprise face','a fear face','a disgust face','a happy face','a sad face','a anger face','a neutral face']
# config.expression_prompts = ['a photo of a surprise face','a photo of a fear face','a photo of a disgust face',
#                              'a photo of a happy face','a photo of a sad face','a photo of a anger face','a photo of a neutral face']

config.region_prompts = ['forehead','brow','eye','nose','mouth']
# config.region_prompts = ['the forehead in a face','the brow in a face','the eye in a face','the nose in a face','the mouth in a face']
# config.region_prompts = ['a photo of forehead','a photo of brow','a photo of eye','a photo of nose','a photo of mouth']
# config.region_prompts = ['a photo of a person\'s forehead','a photo of a person\'s brow','a photo of a person\'s eye','a photo of a person\'s nose','a photo of a person\'s mouth']
# config.region_prompts = ['a face highlighting the forehead','a face highlighting the brow','a face highlighting the eye','a face highlighting the nose','a face highlighting the mouth']

# config.region_prompts = ['forehead','brow','lid','eye','nose','cheek','mouth','lip','chin','jaw']
# config.region_prompts = ['inner brow','outer brow','brow','upper lid','lid', 'eye','nose','cheek',
#                          'mouth','upper lip','lower lip','lip corner','lip','chin','jaw']
# config.region_prompts = ['inner brow raiser','outer brow raiser','brow lowerer','upper lid raiser','lid tightener', 'eye closed',
#                          'nose wrinkler','cheek raiser','mouth stretch','upper lip raiser','lower lip depressor','lip corner puller',
#                          'lip puckerer','chin raiser','jaw drop']

config.expression_contexts_number = 0
config.region_contexts_number = 0
config.alpha = 1.0
config.load_and_tune_prompt_learner = True
config.class_token_position = 'end'
config.class_specific_contexts = False
config.onehot = False
config.requires_grad_namelist = ['image_encoder','text_encoder','cat_head','cross_head','logit_scale']
# config.onehot = True
# config.requires_grad_namelist = ['image_encoder','text_encoder','cat_head','cross_head','fc']

# trainer
config.trainer = 'TPRA_disentangle'
config.seed = 1234
config.batchsize = 64
config.output = '/home/niewei/pythonWork/FER/results_ouput/TPRA'
config.device = 0
config.resume = False
config.verbose = -1
config.test = ['none'] # drop, norm, none

# loss
config.criterion_sup = 'CrossEntropyLoss'
config.reverse = False

# dataloader
config.dataset = 'rafdb'
config.dataset_root = '/home/gaoyu/FER/datasets/rafdb_apvit/RAF-DB/basic/'
config.drop_last = False
config.num_workers = 8
config.input_size = 224

# optimizer
config.optimizer = 'SGD'
config.momentum =0.9
config.weight_decay = 1e-4

# scheduler
config.scheduler = 'ExponentialLR'
config.epochs = 100
config.initial_lr = 0.0001
config.initial_lr_prompts = 0.0001
config.lr_Exponential_gamma = 0.9


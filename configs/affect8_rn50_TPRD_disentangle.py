from easydict import EasyDict as edict

config = edict()

# weights for testing
config.checkpoints = '/home/niewei/pythonWork/FER/results_ouput/weights_TPRD/AffectNet8/TPRD/CLIP-RN50/epoch3_iter17200_acc0.614799976348877.pth'
config.occlusion = False
config.headpose = False

# model
config.model = 'TPRD_disentangle'
config.clip_model = 'RN50'
config.expression_prompts = ['surprise','fear','disgust','happy','sad','anger','neutral','contempt']
# config.expression_prompts = ['surprise','fear','disgust','happy','sad','anger','neutral']
config.region_prompts = ['forehead','brow','eye','nose','mouth']
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
config.verbose = 100
config.test = ['none'] # drop, norm, none

# loss
config.criterion_sup = 'LabelSmoothing_CrossEntropy'
# config.criterion_sup = 'CrossEntropyLoss'
config.reverse = False

# dataloader
config.dataset = 'affect'
config.dataset_root = '/home/niewei/pythonWork/FER/Datasets/AffectNet_Manually_Annotated_Images'
config.drop_last = False
config.num_workers = 8
config.input_size = 224
config.sampler = True

# optimizer
config.optimizer = 'AdamW'
config.weight_decay = 0

# scheduler
config.scheduler = 'ExponentialLR'
config.epochs = 100
config.initial_lr = 0.00001
config.initial_lr_text = 0.0001
config.lr_Exponential_gamma = 0.95


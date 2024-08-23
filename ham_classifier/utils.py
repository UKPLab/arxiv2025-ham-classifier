import json


class KWArgsMixin:
    '''
    Enforces the presence of a kwargs attribute

    Usage: at the end of init, just call KWArgsMixin.__init__(self, arg_name1=arg_value1, ...)
    Important: args need to be passed explicitly, not as **kwargs
    '''
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class UpdateMixin:
    '''
    Adds an update method to the class
    '''
    def update(self):
        '''
        To be implemented by the subclass
        '''
        pass

    def to(self, device):
        '''
        To be implemented by the subclass
        '''
        pass


class DotDict:
    '''
    Dot notation access to dictionary attributes
    '''
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def read_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)
    

class DatasetSetup:
    def __init__(self, n_classes, criterion, train_loader, dev_loader, test_loader):
        self.n_classes = n_classes
        self.criterion = criterion
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
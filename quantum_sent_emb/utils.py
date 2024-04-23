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
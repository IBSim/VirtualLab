class Method_base():
    def __init__(self):
        self.Data = {}
    def __call__(self,*args,**kwargs):
        return self.Run(*args,**kwargs)

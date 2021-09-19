import matplotlib

def ensure_backend(backend):
    '''
    Utility to make sure the plots use the correct backend.
    
    Parameters
    ----------
    backend : str
        one of 'inline' or 'notebook'
    '''
    if backend == 'inline':
        for i in range(100):
            get_ipython().run_line_magic('matplotlib', 'inline')
            bk = matplotlib.get_backend()
            if 'inline' in bk:
                break
    elif backend == 'notebook':
        for i in range(100):
            get_ipython().run_line_magic('matplotlib', 'notebook')
            bk = matplotlib.get_backend()
            if 'agg' in bk:
                break
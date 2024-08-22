from vectordb.api.local import LocalAPI

def Client(**kwargs):
    return LocalAPI(**kwargs)

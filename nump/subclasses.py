#%%
import numpy as np
#%%
class NamedArray(np.ndarray):
    def __new__(cls, array, name = None):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.name = getattr(obj, 'name', None)
    def __repr__(self):
        return f" {self.name}: {super().__repr__()} "
    def __str__(self) :
        return  self.name + ": " + super().__str__() 
#%%
a=NamedArray(np.array([1,2,3,4]),'cool name')
a
#%%
#View casting is the standard ndarray mechanism by which you take an ndarray of any subclass,
# and return a view of the array as another (specified) subclass:
a = np.arange(0,4).reshape(2,2)       
# %%
b = a.view(np.matrix)
# %%
type(a), type(b)
# %%
#There are other points in the use of ndarrays where we need such views, such as copying arrays (c_arr.copy()),
# creating ufunc output arrays (see also __array_wrap__ for ufuncs),
# and reducing methods (like c_arr.mean().
#######
#View casting means you have created a new instance of your array type from any potential subclass of ndarray.
#Creating new from template means you have created a new instance of your class from a pre-existing instance,
# allowing you - for example - to copy across attributes that are particular to your subclass.
###########
#The first is the use of the ndarray.__new__ method for the main work of object initialization,
# rather then the more usual __init__ method.
#The second is the use of the __array_finalize__ method 
# to allow subclasses to clean up after the creation of views and new instances from templates.
#########
#As you can see, the object can be initialized in the __new__ method or the __init__ method, or both,
# and in fact ndarray does not have an __init__ method, because all the initialization is done in the __new__ method.
#__array_finalize__ is the mechanism that numpy provides
# to allow subclasses to handle the various ways that new instances get created.
#Remember that subclass instances can come about in these three ways:
#I: explicit constructor call (obj = MySubClass(params)). 
# This will call the usual sequence of MySubClass.__new__ then (if it exists) MySubClass.__init__.
#II: View casting
#III: Creating new from template

#The signature of __array_finalize__ is:
###
# def __array_finalize__(self, obj):
#%%
class Mat(np.matrix):
    def __new__(cls,array) :
        obj = np.asarray(array).view(cls)
        obj = np.transpose(obj) * obj
        obj= np.pad(obj, pad_width = ((1,0),(0,0)), mode = 'mean')
        return obj
    def __repr__(self):
        return f" {super().__repr__()} "
# %%
a = Mat(np.arange(1,5).reshape(2,2))
# %%
a
# %%
b=np.array([1,2,3,4,5,6])
Mat(b)
# %%
print(Mat(b))
# %%

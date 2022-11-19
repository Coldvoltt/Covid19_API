#%%
import pickle

with open('./model.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)


pred()
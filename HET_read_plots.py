
import pickle

fig_path = "HET_figs/"
fig_name = "MFAM_givene_IDs"
fig_object = pickle.load(open(fig_path+fig_name+'.pickle','rb'))
fig_object.show()

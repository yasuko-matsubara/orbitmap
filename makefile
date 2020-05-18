NAME=orbitmap

SRC= \
fit.py \
kf.py \
main_om.py \
main_om_viz.py \
om_mdb.py \
om_trans_fit.py \
orbitmap.py \
graphviz.py \
main_norm.py \
main_om.sh \
make_viz.sh \
nlds.py \
om_trans.py \
om_multiscale.py \
om_viz.py \
tool.py \
X_demo.sh \



REST= \
README.md \
makefile \
_dat/ \
_out/ \

# demo 
demo: 	$(SRC)
	sh X_demo.sh

sample: $(SRC)
	open ./_out/demo/syn/scan/_viz/result_L.html

samplea: $(SRC)
	open ./_out/demo/syn/scan/_viz/result_L_a.html


clean:
	\rm -r -f *.pyc _out/tmp/* *~ $(NAME).zip
	\rm -r -f *.pyc ./__pycache__/ nohup.out _out/tmp/* *~ all.tar all.zip
zip: $(SRC) $(REST)
	make clean
	zip -r $(NAME).zip $(SRC) $(REST)






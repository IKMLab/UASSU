# sudo apt-get install libxml-parser-perl
# sudo apt-get install -y python-setuptools

cur_dir=$("pwd")

git clone https://github.com/andersjo/pyrouge.git
mv pyrouge/tools/ROUGE-1.5.5 ./
rm -rf pyrouge
export ROUGE_EVAL_HOME="${cur_dir}/ROUGE-1.5.5/data/"

cd ROUGE-1.5.5/data/WordNet-2.0-Exceptions/
./buildExeptionDB.pl . exc WordNet-2.0.exc.db

cd ../
rm -rf WordNet-2.0.exc.db
ln -s WordNet-2.0-Exceptions/WordNet-2.0.exc.db WordNet-2.0.exc.db

git clone https://github.com/bheinzerling/pyrouge.git
cd pyrouge
python setup.py install
pyrouge_set_rouge_path ${cur_dir}/ROUGE-1.5.5/

# Test if pyrouge is working
python -m pyrouge.test
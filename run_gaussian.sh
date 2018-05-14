image="input2.png"
echo ${image}
python segment.py ${image} 10 --gaussian
python segment.py ${image} 20 --gaussian
python segment.py ${image} 30 --gaussian
python segment.py ${image} 40 --gaussian
python segment.py ${image} 50 --gaussian
python segment.py ${image} 60 --gaussian
python segment.py ${image} 70 --gaussian
python segment.py ${image} 80 --gaussian
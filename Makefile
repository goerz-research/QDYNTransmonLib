install:
	python setup.py install

clean:
	@rm -rf build
	@rm -f QDYNTransmonLib/*.pyc
	@rm -f QDYNTransmonLib/PE/*.pyc

.PHONY: install clean

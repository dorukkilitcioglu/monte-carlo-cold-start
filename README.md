# A Monte Carlo Algorithm for Cold Start Recommendation

The purpose of this project is to implement the methods proposed by [Yu Rong, Xiao Wen, and Hong Cheng](http://wwwconference.org/proceedings/www2014/proceedings/p327.pdf) at WWW '14, and if possible, extend it. We will be using the [Movie Lens 1M Dataset](http://grouplens.org/datasets/movielens/) to test the results of the method, as per the paper.

This project uses Python 3, and uses NumPy as a dependency.

Majority of the implementation is done on the Jupyter Notebook, after which it was ported to a Python document to form a coherent class structure. You can use our improved algorithm by passing `alternative = True` to the Predicter class initializer in `paper.py`.

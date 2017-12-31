import numpy as np 
import matplotlib.pyplot as plt 

from util import get_data

X,Y = get_data()
label = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def main():
	while True:
		for n in range(7):
			x,y = X[Y==n],Y[Y==n]
			N =len(y)
			i = np.random.choice(N)
			plt.imshow(x[i].reshape(48,48),cmap='gray')
			plt.title(label[y[i]])
			plt.show()

			stop_choice = input('Quit or Comtinue? enter Q to quit \n')
			if stop_choice.upper() == 'Q':
				break


if __name__ == '__main__':
	main()
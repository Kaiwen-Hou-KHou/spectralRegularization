from char_lang_model import CharLanguageModel
from spectral_reg import SpectralRegularization
import torch
import sys

torch.set_printoptions(linewidth=200)
model = CharLanguageModel(vocab_size = 4, embed_size = 4, hidden_size=50, nlayers=1, 
						rnn_type='RNN', nonlinearity='tanh') 
specreg = SpectralRegularization()

size_cap = int(sys.argv[1])
# H = specreg.forward(model,VOCAB_SIZE=4, stopProb=0.00001, hankelSizeCap=size_cap, verbose=1)
# print(H.item())
for i in range(2,size_cap):
	H = specreg.forward(model,VOCAB_SIZE=4, stopProb=0.00001, hankelSizeCap=i, 
		russian_roulette_type='block_diag', verbose=1)
	print("trace norm:",H.item())
	# H = specreg.forward(model,VOCAB_SIZE=4, stopProb=0.00001, hankelSizeCap=i, verbose=1)
	# print("trace norm:",H.item())
	print()
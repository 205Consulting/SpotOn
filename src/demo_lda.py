# Script: demo_lda
# ----------------
# provides a demo/test for finding topics using LDA
from SpotOn import SpotOn
from util import *

if __name__ == "__main__":
	print_header ("LDA DEMONSTRATION")

	#=====[ Step 1: construct SpotOn	]=====
	so = SpotOn ()

	#=====[ Step 2: load/train semantic analysis	]=====
	# so.semantic_analysis.load ()
	so.train_semantic_analysis ()

	#=====[ Step 3: print lda topics	]=====
	so.print_lda_topics ()

	#=====[ Step 4: save the model	]=====
	so.semantic_analysis.save ()

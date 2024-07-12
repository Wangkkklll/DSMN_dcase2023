class config: 
    def __init__(self):
        

        # wav to mel
        self.sample_rate = 44100 
        self.n_fft = 4096 
        self.win_length = self.n_fft
        self.hop_length = int(self.win_length/4) 
        self.n_mels = 256
        
        # when you use npdata which saved before
        self.reuse = False
        self.reusefolder = 'fs44.1k_bin256_frame4096_hop0.25/'
        
        #when you train with all dataset
        self.include_val = False
        
        self.epochs = 100
        self.batch_size = 32
        
        #augmentation
        self.DIFF_FREQ = True
        self.MIXUP = True
        self.SPEC_AUG = True
        
        # Mixing rate of original 10class and extended 10class
        self.mix_rate = 0.8
        
        
        
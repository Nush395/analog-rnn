### This repo ###
This repo is a work in progress, if you stumble across this repo I'm by no means a tensorflow expert so if you spot something amiss I would appreciate any feedback!

# Analog RNN paper # 
Cool interpretation of the wave equation reformulated as the RNN update equations. This is an attempt to replicate the work thought about
in https://arxiv.org/pdf/1904.12831.pdf. (This isn't my original work! Just trying to have a play from a cool paper and their code found at - https://github.com/fancompute/wavetorch).

## Summary ##
The general idea of the paper is by reformulating the wave equation (written out using finite difference methods) as the
RNN update equations, a physical wave system can be trained to do a similar task to that of the computations in a traditional RNN. Here
the trainable parameter is taken to be the wave speed and the non-linearity provided by activation functions in the RNN are replaced
by the wave speed depending on the field as well as non-linearities introduced when taking intensity measurements to give an output. *The outcome of this is that you could have a physical manifold custom
designed (using an inverse method) for different tasks such as the vowel classification described in the paper without needing a computer whatsoever!* It would be a lens that would direct different vowels to different points! I wanted to see how effective this was on some data myself so this repository is my attempt to replicate the work

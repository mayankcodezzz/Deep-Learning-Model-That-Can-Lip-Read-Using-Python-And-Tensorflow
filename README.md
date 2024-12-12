# Deep Learning Model That Can Lip Read Using Python And Tensorflow

Refer pdf for code Explaination

This project leverages deep learning to perform video-to-text transcription using a Connectionist Temporal Classification (CTC) model. The model is trained to recognize speech or actions from video frames and output a corresponding text transcription. The process begins by preprocessing video data into frame sequences, which are then passed into a recurrent neural network (RNN) model designed for sequence-to-sequence tasks. 

The project uses TensorFlow and Keras for model implementation, with the CTC loss function employed to handle the alignment between input frames and output transcriptions. A custom callback function is integrated to visualize predictions during training and to save model weights. The model is optimized using the Adam optimizer with a learning rate scheduler to adjust the learning rate as training progresses. 

Once trained, the model can process unseen video samples and predict corresponding transcriptions using CTC decoding. This approach is robust to varying sequence lengths and allows the model to predict text from videos where the alignment between input and output is not explicitly available. The model is evaluated on multiple video samples, showing its capability to transcribe speech or actions with notable accuracy, offering potential applications in automated captioning, video indexing, and accessibility tools.

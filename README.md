# EE656-Project
Welcome to our project on intelligent condition-based monitoring of air compressors using acoustic signals. This work was carried out as part of our coursework for EE-656 at IIT Kanpur, and is inspired by the methodologies discussed in the IEEE paper by Prof. Nishchal K. Verma et al.. Our goal was simple but ambitious: build an accurate, fast, and interpretable system that can detect faults in industrial air compressors just by listening to how they sound.

Air compressors are everywhere in industrial setups—from chemical plants to manufacturing units—and any unexpected failure can cause costly downtime, damage, or even safety hazards. We decided to explore acoustic signal analysis as a non-intrusive and intelligent way to detect such faults early. Our system listens to sound recordings of air compressors and tells you whether it’s running smoothly or suffering from one of seven common faults (like valve issues, piston wear, belt problems, etc.).

We worked with a high-resolution dataset recorded right here at IIT Kanpur. Each sample is a 5-second long recording captured at 50 kHz, and the dataset includes eight different classes: one healthy condition and seven types of faults. Every class has 225 examples, which makes the dataset nicely balanced.

Our approach was to keep the system practical yet effective. We didn’t go all-in on heavy signal processing methods like EMD or PCA from the paper. Instead, we leaned into more intuitive yet powerful techniques—like Fast Fourier Transform (FFT), Discrete Cosine Transform (DCT), Wavelet Packet Decomposition, Short-Time Fourier Transform (STFT), and basic time-domain statistics. These collectively gave us a rich 70-dimensional feature set per recording.

To prevent overfitting and improve efficiency, we used mutual information to select the top 25 most relevant features. Then we trained a Support Vector Machine (SVM) classifier with an RBF kernel to distinguish between the eight conditions. The results were incredibly promising. On our test set, the model achieved a classification accuracy of 99.2%, with near-perfect precision and recall across all classes. Even the computational performance was solid—training and prediction took just a few seconds.

We also compared our implementation with the IEEE methodology. While the paper emphasized techniques like EMD and PCA, our simpler pipeline managed to match or even exceed performance using more computationally efficient methods. This shows that with thoughtful feature design and tuning, even relatively lightweight models can perform extremely well on real-world diagnostic tasks.

If you're a researcher, engineer, or student looking to explore machine learning in mechanical fault diagnosis—or you're just curious about how machine learning can "hear" what's wrong with a machine—this project is a great starting point. All the code is available in a Jupyter Notebook, and we’ve made sure it’s modular, readable, and easy to adapt or extend.

Thanks for checking out our work! We hope it helps you learn something new or inspires your next project.

—
Rutul Bhanushali, Kaushal Mehra, Abhay Tripathi, Rhythm Agrawal
EE-656 | IIT Kanpur
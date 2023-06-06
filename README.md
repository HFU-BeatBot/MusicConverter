# MusicConverter
## Description
<p>This library is created because of the discrepancy between jLibrosa and Librosa.
As of right now, our AI uses a .csv file to train the AI, and the .csv created using features generated
with librosa differs from the ones generated with jLibrosa, sometimes by 10%, and other times by more than 50%
This library creates a .csv file with the features generated with jLibrosa. </p>

## Usage:
an input and output directories are needed where the program will run
the input directory should have subdirectories which are the genres of the songs inside of them
eg. <br>
+input/ <br>
+++blues/ <br>
+++rock/ <br>
+++classical/ <br>
+++... <br>
All subdirectories (genres) and will get copied to the output/ directory. All .wav files will get copied to their respective subdirectories
and all .mp3 files will get converted and then saved as .wav in the output directory

# GRETCHEN ORACLE

Deep within the urban landscapes that we are all ever so used to, there is a world of mystery and divination. Tarot cards have been a tool that has been used for centuries, evolving from a trick-taking card game called Tarocchi in Northern Italy to instruments of fortune telling in France. Whether one believes in the power of Tarot, its charm remains unmatched against the testament of time. Interpreting the cards is an art form that requires a tremendous amount of patience and experience to be able to do it on its own. As a group we wanted to see if we were able to make Gretchen interpret the Tarot cards for you so that we will be able to make Tarot readings more accessible to the general public.

## Installation

1. Clone the repository:
    ```bash
   git clone https://teaching.csap.snu.ac.kr/lesimp/sigmagirls67.git
   cd sigmagirls67
   ```
2. Create and activate virtual environment (recommended):
* **Windows:**
  ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
* **macOS / Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the required dependencies:
    ```bash
   pip install -r requirements.txt
   ```
You don't need to actually install the entire dataset we created, as the project already uses a model trained on this data. However, if you're interested in viewing the source data, you can download our dataset from the link below:
https://drive.google.com/file/d/1blPF8maJf5K1ecAT4vHB3OqxEf52CoXI/view?usp=drive_link

## Usage

To run the program, run the command:
* **Windows:**
    ```bash
   python main.py
   ```
* **macOS / Linux:**
    ```bash
   python3 main.py
   ```
  
To select the prediction mode, show the required number of fingers to the camera:
1. Secret song to lift your spirits <3;
2. Personality fortune telling;
3. Career fortune telling:
4. Love fortune telling;
5. Yes or No fortune telling.

After selection the prediction mode, show the drawn tarot cards (1 for Yes/No and Personality modes, 3 for Career and Love modes) to the camera and wait for the program to record them.
Wait for Gretchen to tell you your fate and rejoice! (Or cry, depending on your fate) :)

To exit the mode, show your fist to the camera. To exit the program, press Q.

*Just have fun and remember that your life is not controlled by tarot cards, gods or stars. Everything is in your hands, and wow, fortune tells me everything will definitely be wonderful <3*
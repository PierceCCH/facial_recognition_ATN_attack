# Real-time Impersonation Attack on Facial Recognition Systems

## Setup

1. git clone the repository
    ```
    git clone https://github.com/PierceCCH/facial_recognition_ATN_attack.git
    ```
2. Create a new conda or python virtual environment (this was tested with python3.10)
3. Install the required packages
    ```
    pip install -r requirements.txt
    ```

## Data Collection and Training

1. Run the data collection script to collect face data for each person
    ```
    python collect_data.py --person_label <person_label>
    ```
    Replace `<person_label>` with the name of the person you are collecting data for. This will create a folder in the `data` directory with the person's name and collect 240 training and 60 validation images for that person.

2. Train the victim model (FaceNet)
    ```
    python train_facenet_recognition.py
    ```
    This will train the FaceNet model on the collected data and save the trained model as `facenet_4class.pth`.

3. Train the attack network
    ```
    python train_reface_atn.py
    ```
    This trains the attack network to impersonate one person as another. The trained models are saved in the `models` directory. Each model is saved as `attack_<person_A>_to_<person_B>.pth`, where `person_A` is the attacker and `person_B` is being impersonated.

## Real-time Impersonation Attack Demo

1. Run the real-time victim model inference script
    ```
    python live_inference_victim_model.py
    ```
    This will open a webcam window and display the recognized person's name in real-time.

2. Run the real-time impersonation attack script
    ```
    python live_reface_attack.py
    ```
    Similar to the previous script, this will open a webcam window and display the recognized person's name in real-time. However, the attack network can be toggled on or off using `A`. Toggle on-off JPEG compression as a defense using `D`. 

## Evaluation

`evaluation.ipynb` contains the code we used to evaluate the success rates of the attacks.



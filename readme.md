1. 6 code files needed
collect_data.py -> collected face data
dataset_utils.py -> formated data collected

train_facenet_recognition.py -> train the face recognize model
train_reface_atn.py -> train the attack network for model

live_inference_victim_model.py -> realtime recognize people face
live_reface_attack.py ->attack people face in realtime

2. file genrated
facenet_4class.pth -> trained result @ train_facenet_recognition.py
attack_A_to_B.pth -> trained result people to attack another people @ train_reface_atn.py

3. might rubbish by testing don't care
face_recognition_model.pkl
victim_mobilenet.pth

4. **how to run**
- activate the virtual environment: .\.venv\Scripts\activate
- test train for recognition face: python .\live_inference_victim_model.py
- test attack result: python .\live_reface_attack.py

for live_reface_attack.py

toggle attack on/off: press 'a'
toggle attack different person: while attack mode on(press: 'Al':0, 'Nhat':1, 'Pierce':2, 'Yaqi':3)
zoom in/out attack intensity: i think it is unuseful when I testing....
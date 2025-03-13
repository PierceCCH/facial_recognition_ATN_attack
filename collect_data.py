import cv2
import os
import argparse
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="data",
                        help="root output: train/val/person_label folder creation")
    parser.add_argument("--person_label", type=str, default="PersonA",
                        help="name, e.g. PersonA, PersonB")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="How long save the image")
    args = parser.parse_args()

    # === we set: first 60 for val，next 240 for train，total 300 pictures ===
    MAX_VAL = 60
    MAX_TRAIN = 240

    # create train/val output path
    val_dir   = os.path.join(args.output_root, "val",   args.person_label)
    train_dir = os.path.join(args.output_root, "train", args.person_label)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    frame_count = 0
    val_count   = 0
    train_count = 0

    print(f"Begin collect {args.person_label} face...")
    print("press Q to quit。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(50,50)
        )

        # picture face on detected
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        cv2.imshow("Collect Data - Press Q to Quit", frame)

        frame_count += 1

        # judge whether need to save
        if len(faces) > 0 and (frame_count % args.save_interval == 0):
            x, y, w, h = faces[0]  # save only one face
            face_roi = frame[y:y+h, x:x+w]

            # decide val or train
            if val_count < MAX_VAL:
                # save to val
                timestamp = int(time.time())
                filename = os.path.join(val_dir, f"{args.person_label}_val_{timestamp}_{val_count}.jpg")
                cv2.imwrite(filename, face_roi)
                print(f"[SAVE to val] {filename}")
                val_count += 1

            elif train_count < MAX_TRAIN:
                # save to train
                timestamp = int(time.time())
                filename = os.path.join(train_dir, f"{args.person_label}_train_{timestamp}_{train_count}.jpg")
                cv2.imwrite(filename, face_roi)
                print(f"[SAVE to train] {filename}")
                train_count += 1

            else:
                # pass 300 pictures(60 + 240)，not save more
                print("Already detected 300 pictures(60 val + 240 train). Can press Q to quit")

        # Press Q to quit
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Collecting finsh")
    print(f"Totally saved val={val_count} pictures, train={train_count} pictures.")


if __name__ == "__main__":
    main()

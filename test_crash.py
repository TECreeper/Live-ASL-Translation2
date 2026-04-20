import cv2
import ASL

translator = ASL.ASLClassifier()
features = [0.1]*46
ASL.log_data("c:\\PYASL\\asl_data_real.csv", 'Z', features)
try:
    translator.train_model("c:\\PYASL\\asl_data_real.csv")
    print("Train model successful")
except Exception as e:
    import traceback
    traceback.print_exc()
